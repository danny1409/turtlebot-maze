"""
Zenoh-based YOLOv8 object detector with CLIP embedding extraction.

Subscribes to RGB, depth, odometry, and TF through zenoh-bridge-ros2dds.
Pairs RGB keyframes with the nearest ROS-stamped depth/odometry samples,
runs YOLOv8 and batched CLIP, and publishes object coordinates in map.
The reference main() and nested-callback structure is retained.

Task 2 frame handling: odometry gives parent <- child. For an odom parent,
map <- odom comes from the bridged Nav2 /tf stream, not a guessed offset.
If the child is base_footprint, its fixed base_link edge comes from
/tf_static. Dynamic edges use the nearest sample to the synchronized
odometry timestamp within 200 ms. Missing transforms reject the frame.
The supplied D435i calibration assumes registered RGB/depth at 320x240.
Restart the detector after resetting Gazebo's simulation clock.

Based on:
- https://github.com/eclipse-zenoh/zenoh-demos/tree/main/ROS2/zenoh-python-lidar-plot
- https://github.com/eclipse-zenoh/zenoh-demos/tree/main/computer-vision/face-recog
"""

import argparse
import base64
import json
import math
import socket
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
import zenoh
from pycdr2 import IdlStruct
from pycdr2.types import array, sequence, uint8, uint32, int32, float64
from dataclasses import dataclass
from typing import List
from ultralytics import YOLO
from PIL import Image as PILImage

# Task 2 choices: central half of each box side, at least 9 valid pixels,
# and no more than 200 ms between an RGB frame and either sensor sample.
MAX_SYNC_NS = 200_000_000
SYNC_WAIT_SECONDS = 2.0  # Arrival grace; the ROS timestamp limit remains 200 ms.
CENTRAL_FRACTION = 0.5
MIN_VALID_SAMPLES = 9
FX = FY = 277.13
CX, CY = 160.0, 120.0

R_CAMERA_FROM_OPTICAL = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=float)
T_BASE_FROM_CAMERA = np.eye(4)
T_BASE_FROM_CAMERA[:3, 3] = [0.064, -0.065, 0.094]


# CDR struct matching sensor_msgs/msg/Image
# See: https://docs.ros2.org/latest/api/sensor_msgs/msg/Image.html
@dataclass
class Time(IdlStruct, typename="builtin_interfaces/msg/Time"):
    sec: int32
    nanosec: uint32


@dataclass
class Header(IdlStruct, typename="std_msgs/msg/Header"):
    stamp: Time
    frame_id: str


@dataclass
class Image(IdlStruct, typename="sensor_msgs/msg/Image"):
    header: Header
    height: uint32
    width: uint32
    encoding: str
    is_bigendian: uint8
    step: uint32
    data: sequence[uint8]


@dataclass
class Point(IdlStruct, typename="geometry_msgs/msg/Point"):
    x: float64
    y: float64
    z: float64


@dataclass
class Quaternion(IdlStruct, typename="geometry_msgs/msg/Quaternion"):
    x: float64
    y: float64
    z: float64
    w: float64


@dataclass
class Pose(IdlStruct, typename="geometry_msgs/msg/Pose"):
    position: Point
    orientation: Quaternion


@dataclass
class PoseWithCovariance(IdlStruct, typename="geometry_msgs/msg/PoseWithCovariance"):
    pose: Pose
    covariance: array[float64, 36]


@dataclass
class Twist(IdlStruct, typename="geometry_msgs/msg/Twist"):
    linear: Point
    angular: Point


@dataclass
class TwistWithCovariance(IdlStruct, typename="geometry_msgs/msg/TwistWithCovariance"):
    twist: Twist
    covariance: array[float64, 36]


@dataclass
class Odometry(IdlStruct, typename="nav_msgs/msg/Odometry"):
    header: Header
    child_frame_id: str
    pose: PoseWithCovariance
    twist: TwistWithCovariance


@dataclass
class Transform(IdlStruct, typename="geometry_msgs/msg/Transform"):
    translation: Point
    rotation: Quaternion


@dataclass
class TransformStamped(IdlStruct, typename="geometry_msgs/msg/TransformStamped"):
    header: Header
    child_frame_id: str
    transform: Transform


@dataclass
class TFMessage(IdlStruct, typename="tf2_msgs/msg/TFMessage"):
    transforms: List[TransformStamped]


def timestamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def nearest_sample(samples, wanted_ns):
    """Search by ROS timestamp, including messages arriving out of order."""
    if not samples:
        return None
    match = min(samples, key=lambda item: abs(item[0] - wanted_ns))
    return match if abs(match[0] - wanted_ns) <= MAX_SYNC_NS else None


def lookup_transform(target, source, wanted_ns, dynamic_tf, static_tf):
    """The two required TF edges: map/odom and odometry-child/base_link.

    Use the nearest dynamic sample within 200 ms, or a timeless static edge.
    A missing edge is an error; never assume map and odom are identical.
    """
    target, source = target.lstrip("/"), source.lstrip("/")
    if target == source:
        return np.eye(4)
    for edge, inverse in (((target, source), False), ((source, target), True)):
        matrix = static_tf.get(edge)
        if matrix is None:
            match = nearest_sample(dynamic_tf.get(edge, []), wanted_ns)
            if match is not None:
                matrix = match[1]
        if matrix is not None:
            return np.linalg.inv(matrix) if inverse else matrix
    raise LookupError(f"Missing time-valid TF: {target} <- {source}")


def pose_matrix(position, quaternion):
    """Homogeneous transform from a translation and xyzw quaternion."""
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-12:
        raise ValueError("Invalid quaternion")
    x, y, z, w = q / norm
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ]
    matrix[:3, 3] = position
    if not np.isfinite(matrix).all():
        raise ValueError("Invalid transform")
    return matrix


def transform_matrix(message):
    t, q = message.translation, message.rotation
    return pose_matrix((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


def decode_depth(message):
    """Decode 32FC1 metres, respecting row padding and byte order."""
    if message.encoding != "32FC1":
        raise ValueError(f"Expected 32FC1 depth, got {message.encoding}")
    h, w, step = message.height, message.width, message.step
    data = bytes(message.data)
    if h <= 0 or w <= 0 or step < w * 4 or len(data) < h * step:
        raise ValueError("Invalid depth dimensions or byte count")
    dtype = ">f4" if message.is_bigendian else "<f4"
    return np.ndarray((h, w), dtype=dtype, buffer=data, strides=(step, 4)).astype(np.float32)


def median_box_depth(depth, bbox):
    """Central 50% of width/height = 25% of area; require 9 valid pixels."""
    x1, y1, x2, y2 = bbox
    h, w = depth.shape
    if not np.isfinite(bbox).all() or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("Invalid bounding box")
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    half_width = (x2 - x1) * CENTRAL_FRACTION / 2
    half_height = (y2 - y1) * CENTRAL_FRACTION / 2
    a, b = max(0, math.floor(u-half_width)), max(0, math.floor(v-half_height))
    c, d = min(w, math.ceil(u+half_width)), min(h, math.ceil(v+half_height))
    interior = depth[b:d, a:c]
    valid = interior[np.isfinite(interior) & (interior > 0)]
    if valid.size < MIN_VALID_SAMPLES:
        raise ValueError("Too few finite positive depth pixels")
    return float(np.median(valid)), int(valid.size), [a, b, c, d]


def optical_point(u, v, depth):
    if not np.isfinite([u, v, depth]).all() or depth <= 0:
        raise ValueError("Pixel coordinates and positive depth must be finite")
    return np.array([(u-CX)*depth/FX, (v-CY)*depth/FY, depth, 1.0])


def back_project(u, v, depth, map_from_base):
    # optical -> camera body axes -> base_link -> map, right to left.
    return (map_from_base @ T_BASE_FROM_CAMERA @ R_CAMERA_FROM_OPTICAL
            @ optical_point(u, v, depth))[:3]


def camera_quaternion(map_from_base):
    """Compose the robot quaternion with the fixed optical-frame rotation."""
    # This also accepts map/base matrices containing roll and pitch.
    r = (map_from_base @ R_CAMERA_FROM_OPTICAL)[:3, :3]
    trace = np.trace(r)
    if trace > 0:
        s = math.sqrt(trace + 1) * 2
        q = [(r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s,
             (r[1, 0]-r[0, 1])/s, s/4]
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(1+r[0, 0]-r[1, 1]-r[2, 2]) * 2
            q = [s/4, (r[0, 1]+r[1, 0])/s, (r[0, 2]+r[2, 0])/s, (r[2, 1]-r[1, 2])/s]
        elif i == 1:
            s = math.sqrt(1+r[1, 1]-r[0, 0]-r[2, 2]) * 2
            q = [(r[0, 1]+r[1, 0])/s, s/4, (r[1, 2]+r[2, 1])/s, (r[0, 2]-r[2, 0])/s]
        else:
            s = math.sqrt(1+r[2, 2]-r[0, 0]-r[1, 1]) * 2
            q = [(r[0, 2]+r[2, 0])/s, (r[1, 2]+r[2, 1])/s, s/4, (r[1, 0]-r[0, 1])/s]
    return np.asarray(q) / np.linalg.norm(q)


def quaternion_to_yaw(q: Quaternion) -> float:
    """Extract yaw angle from a quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def main():
    parser = argparse.ArgumentParser(description="Zenoh YOLOv8 Object Detector")
    parser.add_argument(
        "-e",
        "--connect",
        type=str,
        default="",
        help="Zenoh endpoint to connect to (empty = multicast)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics model name or path",
    )
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum detection confidence",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="camera/color/image_raw",
        help="Zenoh key for camera images (D435i: camera/color/image_raw via zenoh-bridge-ros2dds)",
    )
    parser.add_argument(
        "--detection-key",
        type=str,
        default="tb/detections",
        help="Zenoh key to publish detections to",
    )
    parser.add_argument(
        "--max-fps", type=float, default=10.0, help="Incoming image limit in Hz (maximum 10)"
    )
    parser.add_argument(
        "--enable-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute CLIP embeddings for each detection crop (default: enabled)",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B-32",
        help="open_clip model architecture",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="open_clip pretrained weights tag",
    )
    parser.add_argument(
        "--odom-key",
        type=str,
        default="odom",
        help="Zenoh key for odometry (bridged from /odom)",
    )
    parser.add_argument("--depth-key", default="camera/depth/image_rect_raw")
    parser.add_argument("--tf-key", default="tf")
    parser.add_argument("--tf-static-key", default="tf_static")
    parser.add_argument(
        "--keyframe-dist",
        type=float,
        default=0.5,
        help="Minimum translation (m) to trigger a keyframe",
    )
    parser.add_argument(
        "--keyframe-angle",
        type=float,
        default=15.0,
        help="Minimum rotation (degrees) to trigger a keyframe",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Session identifier (default: auto-generated as hostname-timestamp)",
    )
    parser.add_argument(
        "--save-keyframes", type=Path,
        help="Optional directory for accepted RGB images used in physical-object labeling",
    )
    args = parser.parse_args()
    if args.save_keyframes:
        args.save_keyframes.mkdir(parents=True, exist_ok=True)
    if not 0 < args.max_fps <= 10:
        parser.error("max-fps must be greater than zero and at most 10")

    if args.run_id:
        run_id = args.run_id
    else:
        run_id = f"{socket.gethostname()}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    print(f"Run ID: {run_id}")

    # Load YOLO model
    model = YOLO(args.model)
    print(f"Loaded model: {args.model}")
    print(f"Subscribing to: {args.image_key}")
    print(f"Publishing to:  {args.detection_key}")

    # Load CLIP model for embedding extraction
    clip_model = None
    clip_preprocess = None
    clip_device = None
    if args.enable_embeddings:
        import open_clip

        clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            args.clip_model, pretrained=args.clip_pretrained
        )
        clip_model = clip_model.to(clip_device).eval()
        clip_embedding_dim = clip_model.visual.output_dim
        if clip_embedding_dim != 512:
            raise ValueError("Task 1 requires 512-dimensional CLIP embeddings")
        clip_tag = f"{args.clip_model}/{args.clip_pretrained}"
        print(
            f"CLIP embeddings enabled: {clip_tag} "
            f"({clip_embedding_dim}-dim) on {clip_device}"
        )

    min_interval = 1.0 / args.max_fps
    last_image_time = -math.inf

    # Callbacks collect samples without waiting for inference. Main searches
    # the histories by ROS time, not by the order in which callbacks ran.
    sensor_lock = Lock()
    depth_samples = deque(maxlen=128)
    odom_samples = deque(maxlen=256)
    color_frames = deque(maxlen=32)
    dynamic_tf = {}
    static_tf = {}

    # Keyframe state
    keyframe_dist_thresh = args.keyframe_dist
    keyframe_angle_thresh = math.radians(args.keyframe_angle)
    last_keyframe_pose = None  # (x, y, yaw)
    keyframe_id = 0
    last_processed_ns = -1
    last_skip_log = -math.inf

    print(
        f"Keyframe gating: dist={keyframe_dist_thresh}m, "
        f"angle={args.keyframe_angle}deg"
    )
    print(f"Subscribing to odom: {args.odom_key}")
    print(f"Subscribing to depth: {args.depth_key}; sync limit=200ms")
    print(f"Subscribing to TF: {args.tf_key}, {args.tf_static_key}")

    # Open Zenoh session
    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", json.dumps([args.connect]))

    session = zenoh.open(conf)
    pub = session.declare_publisher(args.detection_key)

    def odom_callback(sample):
        try:
            odom_msg = Odometry.deserialize(bytes(sample.payload))
        except Exception as e:
            print(f"Odom CDR deserialize error: {e}")
            return
        with sensor_lock:
            odom_samples.append((timestamp_ns(odom_msg.header), odom_msg))

    def depth_callback(sample):
        try:
            message = Image.deserialize(bytes(sample.payload))
            with sensor_lock:
                depth_samples.append((timestamp_ns(message.header), message))
        except Exception as e:
            print(f"Depth CDR deserialize error: {e}")

    def tf_callback(sample, is_static=False):
        try:
            message = TFMessage.deserialize(bytes(sample.payload))
            with sensor_lock:
                for item in message.transforms:
                    edge = (item.header.frame_id.lstrip("/"), item.child_frame_id.lstrip("/"))
                    matrix = transform_matrix(item.transform)
                    if is_static:
                        static_tf[edge] = matrix
                    else:
                        dynamic_tf.setdefault(edge, deque(maxlen=256)).append(
                            (timestamp_ns(item.header), matrix))
        except Exception as e:
            print(f"TF CDR deserialize error: {e}")

    def is_keyframe(pose) -> bool:
        # Without odometry we cannot check movement: wait rather than infer.
        if pose is None:
            return False
        if last_keyframe_pose is None:
            return True
        dx = pose[0] - last_keyframe_pose[0]
        dy = pose[1] - last_keyframe_pose[1]
        dist = math.hypot(dx, dy)
        dyaw = pose[2] - last_keyframe_pose[2]
        # Wrapped difference avoids a large jump between +pi and -pi.
        dyaw = abs(math.atan2(math.sin(dyaw), math.cos(dyaw)))
        return dist >= keyframe_dist_thresh or dyaw >= keyframe_angle_thresh

    def image_callback(sample):
        nonlocal last_image_time

        # Rate limit
        now = time.monotonic()
        with sensor_lock:
            if now - last_image_time < min_interval:
                return
            last_image_time = now

        # Deserialize CDR-encoded sensor_msgs/Image
        try:
            img_msg = Image.deserialize(bytes(sample.payload))
        except Exception as e:
            print(f"CDR deserialize error: {e}")
            return

        stamp = timestamp_ns(img_msg.header)
        with sensor_lock:
            color_frames.append((stamp, now, img_msg))

    def process_keyframe(img_msg, depth_match, odom_match, map_from_base, received_at):
        nonlocal last_keyframe_pose, keyframe_id
        processing_started = time.monotonic()
        stamp = timestamp_ns(img_msg.header)
        depth_stamp, depth_msg = depth_match
        odom_stamp, odom_msg = odom_match
        p = odom_msg.pose.pose.position
        pose = (p.x, p.y, quaternion_to_yaw(odom_msg.pose.pose.orientation))
        depth = decode_depth(depth_msg)
        if (img_msg.width, img_msg.height) != (320, 240) or depth.shape != (240, 320):
            raise ValueError("Task 2 assumes registered RGB/depth at 320x240")

        # Convert to numpy array
        if img_msg.encoding not in ("rgb8", "bgr8"):
            print(f"Unsupported encoding: {img_msg.encoding}")
            return
        data = bytes(img_msg.data)
        if (img_msg.height <= 0 or img_msg.width <= 0 or img_msg.step < img_msg.width * 3
                or len(data) < img_msg.height * img_msg.step):
            print("Invalid image dimensions, row size, or data length")
            return
        # step is the number of bytes per row, including any padding.
        frame = np.frombuffer(data[:img_msg.height * img_msg.step], dtype=np.uint8).reshape(
            img_msg.height, img_msg.step
        )[:, :img_msg.width * 3].reshape(img_msg.height, img_msg.width, 3).copy()
        if img_msg.encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run YOLOv8 inference
        results = model.predict(frame, conf=args.confidence, verbose=False)

        # Build detections list
        detections = []
        crops = []
        rejected_depth = 0
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                bbox = [float(x) for x in box.xyxy[0].tolist()]
                det = {
                    "class": model.names[cls_id],
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": bbox,
                }
                # Clip the box to the image, then estimate depth from its interior.
                bbox = [max(0.0, bbox[0]), max(0.0, bbox[1]),
                        min(float(img_msg.width), bbox[2]), min(float(img_msg.height), bbox[3])]
                det["bbox"] = bbox
                try:
                    depth_m, valid_count, roi = median_box_depth(depth, bbox)
                except ValueError:
                    rejected_depth += 1
                    continue
                u, v = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                point = back_project(u, v, depth_m, map_from_base)
                camera_point = optical_point(u, v, depth_m)[:3]
                det.update({
                    "map_x": float(point[0]), "map_y": float(point[1]), "map_z": float(point[2]),
                    "depth_m": depth_m, "valid_depth_samples": valid_count, "depth_roi_xyxy": roi,
                    "range_m": float(np.linalg.norm(camera_point)),
                    "bearing_rad": math.atan2(-float(camera_point[0]), float(camera_point[2])),
                })
                # Crop the detection region for CLIP encoding
                if clip_model is not None:
                    x1, y1, x2, y2 = (
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                    )
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    crop_bgr = frame[y1:y2, x1:x2]
                    if crop_bgr.size == 0:
                        continue  # An empty crop cannot have a visual embedding.
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    crops.append(PILImage.fromarray(crop_rgb))
                detections.append(det)

        # Batch-encode CLIP embeddings for all crops
        if clip_model is not None and crops:
            batch = torch.stack([clip_preprocess(crop) for crop in crops]).to(clip_device)
            with torch.no_grad():
                embeddings = clip_model.encode_image(batch)
            # Unit vectors make their dot product equal to cosine similarity.
            embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
            embeddings_np = embeddings.cpu().numpy()
            if (embeddings_np.shape != (len(detections), 512)
                    or not np.isfinite(embeddings_np).all()
                    or not np.allclose(np.linalg.norm(embeddings_np, axis=1), 1.0, atol=1e-5, rtol=0)):
                raise ValueError("CLIP must produce finite unit vectors of length 512")
            for det, vec in zip(detections, embeddings_np):
                det["embedding"] = base64.b64encode(vec.astype("<f4").tobytes()).decode("ascii")
                det["embedding_dim"] = 512
                det["embedding_dtype"] = "<f4"
                det["embedding_model"] = clip_tag

        # Build keyframe envelope with pose metadata
        map_yaw = math.atan2(map_from_base[1, 0], map_from_base[0, 0])
        map_from_camera = map_from_base @ T_BASE_FROM_CAMERA @ R_CAMERA_FROM_OPTICAL
        camera_q = camera_quaternion(map_from_base)
        pose_data = {
            "odom_x": pose[0], "odom_y": pose[1], "odom_yaw": pose[2],
            "odom_frame": odom_msg.header.frame_id, "odom_timestamp_ns": odom_stamp,
            "map_x": float(map_from_base[0, 3]), "map_y": float(map_from_base[1, 3]),
            "map_z": float(map_from_base[2, 3]), "map_yaw": map_yaw,
            "camera_x": float(map_from_camera[0, 3]), "camera_y": float(map_from_camera[1, 3]),
            "camera_z": float(map_from_camera[2, 3]),
            **{f"camera_q{axis}": float(value) for axis, value in zip("xyzw", camera_q)},
        }
        next_keyframe_id = keyframe_id + 1

        for idx, det in enumerate(detections):
            det["det_id"] = f"{run_id}_kf{next_keyframe_id}_d{idx}"

        envelope = {
            "schema_version": 2, "frame_id": "map", "clock_epoch": 0,
            "run_id": run_id,
            "keyframe_id": next_keyframe_id,
            "timestamp": stamp / 1_000_000_000,
            "timestamp_ns": stamp, "seen_at_ns": stamp,
            "timestamp_domain": "ROS",
            "image_frame": img_msg.header.frame_id,
            "depth_frame": depth_msg.header.frame_id,
            "depth_timestamp_ns": depth_stamp, "depth_delta_ns": depth_stamp-stamp,
            "odom_delta_ns": odom_stamp-stamp,
            "depth_fraction": CENTRAL_FRACTION, "rejected_depth_count": rejected_depth,
            "range_convention": "3D distance from camera centre, metres",
            "bearing_convention": "0 camera forward; positive left, radians",
            **pose_data,
            "detections": detections,
        }
        if args.save_keyframes:
            image_path = args.save_keyframes / f"{run_id}_kf{next_keyframe_id}.png"
            # YOLO uses BGR; save an RGB PNG from this exact accepted frame.
            PILImage.fromarray(frame[:, :, ::-1]).save(image_path)
            envelope["keyframe_image"] = image_path.name
        payload = json.dumps(envelope, allow_nan=False)
        pub.put(payload.encode())
        # Advance only after a valid frame has been processed and published.
        last_keyframe_pose = pose
        keyframe_id = next_keyframe_id

        classes = [d["class"] for d in detections]
        points = [f"{d['class']}=({d['map_x']:.2f},{d['map_y']:.2f},{d['map_z']:.2f})"
                  for d in detections[:3]]
        print(
            f"KF#{keyframe_id} Detected: {classes} "
            f"@ map ({pose_data['map_x']:.4f}, {pose_data['map_y']:.4f}), "
            f"yaw={math.degrees(map_yaw):.1f}deg; "
            f"sync depth={(depth_stamp-stamp)/1e6:.1f}ms odom={(odom_stamp-stamp)/1e6:.1f}ms; "
            f"wait={processing_started-received_at:.2f}s process={time.monotonic()-processing_started:.2f}s; "
            f"depth rejected={rejected_depth}; " + "; ".join(points), flush=True
        )

    odom_sub = session.declare_subscriber(args.odom_key, odom_callback)
    img_sub = session.declare_subscriber(args.image_key, image_callback)
    depth_sub = session.declare_subscriber(args.depth_key, depth_callback)
    tf_sub = session.declare_subscriber(args.tf_key, tf_callback)
    static_sub = session.declare_subscriber(args.tf_static_key, lambda sample: tf_callback(sample, True))
    # Request cached static TF because robot_state_publisher may have sent it
    # before this detector started. The live static subscriber remains active.
    try:
        for reply in session.get(args.tf_static_key, timeout=2.0):
            if reply.ok is not None:
                tf_callback(reply.ok, True)
    except Exception:
        pass
    print("Detector running. Press Ctrl+C to stop.")

    try:
        while True:
            matches = None
            reason = None
            with sensor_lock:
                if color_frames:
                    stamp, arrived, img_msg = color_frames[0]
                    depths, odoms = list(depth_samples), list(odom_samples)
                    reached = (depths and odoms
                               and max(t for t, _ in depths) >= stamp
                               and max(t for t, _ in odoms) >= stamp)
                    # Wait for both streams to reach the RGB timestamp, then
                    # search all buffered samples. Allow two wall seconds for
                    # late arrivals; sensor timestamps must still be within
                    # 200 ms. A slow simulation need not run in wall time.
                    if reached or time.monotonic()-arrived >= SYNC_WAIT_SECONDS:
                        color_frames.popleft()
                        depth_match = nearest_sample(depths, stamp)
                        odom_match = nearest_sample(odoms, stamp)
                        if stamp <= last_processed_ns:
                            pass  # A late older RGB frame cannot advance this run.
                        elif depth_match is None or odom_match is None:
                            offsets = []
                            for name, samples in (("depth", depths), ("odom", odoms)):
                                if samples:
                                    nearest_ns = min(samples, key=lambda item: abs(item[0]-stamp))[0]
                                    offsets.append(f"{name}={(nearest_ns-stamp)/1e6:+.1f}ms")
                                else:
                                    offsets.append(f"{name}=no samples")
                            reason = ("No depth/odom timestamp match within 200 ms; "
                                      + ", ".join(offsets)
                                      + f"; RGB={stamp/1e9:.3f}s; waited={time.monotonic()-arrived:.1f}s")
                        else:
                            odom_msg = odom_match[1]
                            p, q = odom_msg.pose.pose.position, odom_msg.pose.pose.orientation
                            pose = (p.x, p.y, quaternion_to_yaw(q))
                            if is_keyframe(pose):
                                try:
                                    parent, child = odom_msg.header.frame_id, odom_msg.child_frame_id
                                    if not parent or not child:
                                        raise LookupError("Odometry frame names are empty")
                                    map_from_parent = lookup_transform("map", parent, odom_match[0], dynamic_tf, static_tf)
                                    child_from_base = lookup_transform(child, "base_link", odom_match[0], dynamic_tf, static_tf)
                                    parent_from_child = pose_matrix((p.x, p.y, p.z), (q.x, q.y, q.z, q.w))
                                    map_from_base = map_from_parent @ parent_from_child @ child_from_base
                                    matches = (img_msg, depth_match, odom_match, map_from_base, arrived)
                                except (LookupError, ValueError) as e:
                                    reason = str(e)
                        last_processed_ns = max(last_processed_ns, stamp)
            if reason and time.monotonic()-last_skip_log >= 1.0:
                print(f"Skipped RGB keyframe: {reason}", flush=True)
                last_skip_log = time.monotonic()
            if matches is not None:
                try:
                    process_keyframe(*matches)
                except Exception as e:
                    print(f"Keyframe processing error: {e}", flush=True)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        img_sub.undeclare()
        odom_sub.undeclare()
        depth_sub.undeclare()
        tf_sub.undeclare()
        static_sub.undeclare()
        pub.undeclare()
        session.close()


if __name__ == "__main__":
    main()
