#!/usr/bin/env python3
"""Zenoh YOLO/CLIP detector with synchronized depth and map-frame geometry.

Adapted from the supplied turtlebot-maze reference. The timestamp buffers,
TF handling, depth geometry, worker, and map envelope are separate additions.
No rclpy or local ROS installation is used by this process.
"""

import argparse
import base64
from collections import Counter, deque
from datetime import datetime, timezone
import json
import math
import signal
import socket
import threading
import time

import numpy as np
from PIL import Image as PILImage
import zenoh

from mapping_core import (
    BASE_FROM_OPTICAL, Intrinsics, TimedBuffer, TransformBuffer,
    decode_color, decode_depth, detection_geometry, keyframe_needed,
    matrix_to_quaternion, median_depth, pose_matrix, robot_pose,
)
from ros_cdr import Image, Odometry, TFMessage, frame_name, stamp_ns, xyz, xyzw


class DetectorRuntime:
    def __init__(self, args, publisher):
        self.args, self.publisher = args, publisher
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.depth = TimedBuffer(128)
        self.odom = TimedBuffer(2048)
        self.transforms = TransformBuffer(4096)
        self.pending = deque(maxlen=4)
        self.counters = Counter()
        self.epoch = 0
        self.newest_color_ns = -1
        self.last_published_ns = -1
        self.last_keyframe_pose = None
        self.keyframe_id = 0
        self.last_inference = -math.inf
        self.warning_times = {}
        self.warning_lock = threading.Lock()
        self.intrinsics = Intrinsics(args.width, args.height, args.fx, args.fy, args.cx, args.cy)
        self.max_delta_ns = round(args.max_sync_ms * 1_000_000)
        self.run_id = args.run_id or (
            f"{socket.gethostname()}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        )

    def warn(self, category, message):
        now = time.monotonic()
        with self.warning_lock:
            if now - self.warning_times.get(category, -math.inf) < 5:
                return
            self.warning_times[category] = now
        print(f"[{category}] {message}", flush=True)

    def on_color(self, sample):
        try:
            message = Image.deserialize(bytes(sample.payload))
            timestamp = stamp_ns(message.header)
            if (message.width, message.height) != (self.args.width, self.args.height):
                raise ValueError("color size differs from configured camera intrinsics")
            with self.condition:
                if timestamp < self.newest_color_ns - 1_000_000_000:
                    self.epoch += 1
                    self.depth.clear()
                    self.odom.clear()
                    self.transforms.clear_dynamic()
                    self.pending.clear()
                    self.last_keyframe_pose = None
                    self.last_published_ns = -1
                    self.newest_color_ns = timestamp
                    self.warn("clock-reset", "ROS time reset; cleared dynamic histories and keyframe reference")
                self.newest_color_ns = max(self.newest_color_ns, timestamp)
                self.counters["rgb"] += 1
                if len(self.pending) == self.pending.maxlen:
                    self.counters["queue_dropped"] += 1
                self.pending.append((timestamp, time.monotonic(), message))
                self.condition.notify_all()
        except Exception as error:
            self.warn("color", str(error))

    def on_depth(self, sample):
        try:
            message = Image.deserialize(bytes(sample.payload))
            image = decode_depth(message)
            if image.shape != (self.args.height, self.args.width):
                raise ValueError("depth size differs from configured, registered RGB camera")
            with self.condition:
                self.depth.add(stamp_ns(message.header), (message, image))
                self.counters["depth"] += 1
                self.condition.notify_all()
        except Exception as error:
            self.warn("depth", str(error))

    def on_odom(self, sample):
        try:
            message = Odometry.deserialize(bytes(sample.payload))
            if not message.header.frame_id or not message.child_frame_id:
                raise ValueError("odometry parent and child frames must be specified")
            matrix = pose_matrix(xyz(message.pose.pose.position), xyzw(message.pose.pose.orientation))
            with self.condition:
                self.odom.add(stamp_ns(message.header), (message, matrix))
                self.counters["odom"] += 1
                self.condition.notify_all()
        except Exception as error:
            self.warn("odom", str(error))

    def on_tf(self, sample, is_static=False):
        try:
            message = TFMessage.deserialize(bytes(sample.payload))
            entries = [(
                frame_name(t.header.frame_id), frame_name(t.child_frame_id), stamp_ns(t.header),
                pose_matrix(xyz(t.transform.translation), xyzw(t.transform.rotation)),
            ) for t in message.transforms]
            with self.condition:
                for parent, child, timestamp, matrix in entries:
                    self.transforms.add(parent, child, timestamp, matrix, is_static)
                self.counters["tf_static" if is_static else "tf"] += 1
                self.condition.notify_all()
        except Exception as error:
            self.warn("tf_static" if is_static else "tf", str(error))

    def request_static_history(self, session):
        # ROS transient-local TF can be cached/queryable by the DDS bridge.
        # A live tf_static subscriber is also active; lack of a query reply is OK.
        try:
            for reply in session.get(self.args.tf_static_key, timeout=2.0):
                try:
                    sample = reply.ok
                    if sample is not None:
                        self.on_tf(sample, is_static=True)
                except Exception:
                    continue
        except Exception as error:
            self.warn("static-query", str(error))

    def snapshot(self, timestamp):
        # Called with the condition lock held. Wait for streams to bracket the
        # RGB stamp, then choose the closest timestamp, not the newest arrival.
        if not self.depth.has_reached(timestamp):
            raise LookupError("waiting for depth at/after the RGB timestamp")
        if not self.odom.has_reached(timestamp):
            raise LookupError("waiting for odometry at/after the RGB timestamp")
        depth = self.depth.nearest(timestamp, self.max_delta_ns)
        odom = self.odom.nearest(timestamp, self.max_delta_ns)
        if depth is None or odom is None:
            raise LookupError(f"no depth/odom match within {self.args.max_sync_ms:g} ms")
        depth_stamp, (depth_message, depth_image) = depth
        odom_stamp, (odom_message, odom_from_child) = odom
        parent = frame_name(odom_message.header.frame_id)
        child = frame_name(odom_message.child_frame_id)
        map_from_odom = self.transforms.lookup(self.args.map_frame, parent, timestamp)
        child_from_base = self.transforms.lookup(child, self.args.base_frame, timestamp)
        odom_from_base = odom_from_child @ child_from_base
        map_from_base = map_from_odom @ odom_from_base
        return (depth_stamp, odom_stamp, depth_message, depth_image, odom_message,
                map_from_base, robot_pose(odom_from_base), self.epoch)

    def process(self, image_message, timestamp, snapshot):
        depth_stamp, odom_stamp, depth_message, depth_image, odom_message, map_from_base, _, epoch = snapshot
        frame = decode_color(image_message)
        results = self.model.predict(frame, conf=self.args.confidence,
                                     device=self.device, verbose=False)
        detections, crops = [], []
        rejected_depth = 0
        yolo_count = 0
        for result in results:
            for box in result.boxes:
                yolo_count += 1
                try:
                    estimate = median_depth(depth_image, box.xyxy[0].tolist(),
                                            self.args.depth_fraction, self.args.min_depth_samples)
                except ValueError:
                    rejected_depth += 1
                    continue
                x1, y1, x2, y2 = estimate.bbox
                crop = frame[math.floor(y1):math.ceil(y2), math.floor(x1):math.ceil(x2)]
                if crop.size == 0:
                    rejected_depth += 1
                    continue
                class_id = int(box.cls[0])
                detection = {
                    "class": self.model.names[class_id],
                    "confidence": float(box.conf[0]),
                    "bbox": list(estimate.bbox),
                    **detection_geometry(estimate, map_from_base, self.intrinsics),
                }
                detections.append(detection)
                crops.append(PILImage.fromarray(crop[:, :, ::-1].copy()))

        if self.clip_model is not None and crops:
            batch = self.torch.stack([self.clip_preprocess(crop) for crop in crops]).to(self.device)
            with self.torch.inference_mode():
                vectors = self.clip_model.encode_image(batch)
                vectors = self.torch.nn.functional.normalize(vectors.float(), dim=-1)
            vectors = vectors.cpu().numpy()
            if vectors.shape != (len(detections), 512) or not np.isfinite(vectors).all():
                raise ValueError("CLIP must produce finite 512-dimensional vectors")
            if np.any(np.linalg.norm(vectors, axis=1) < 0.99):
                raise ValueError("CLIP produced an invalid normalized vector")
            for detection, vector in zip(detections, vectors):
                detection.update({
                    "embedding": base64.b64encode(vector.astype("<f4").tobytes()).decode("ascii"),
                    "embedding_dim": 512, "embedding_dtype": "<f4",
                    "embedding_model": f"{self.args.clip_model}/{self.args.clip_pretrained}",
                })

        rx, ry, rz, yaw = robot_pose(map_from_base)
        map_from_camera = map_from_base @ BASE_FROM_OPTICAL
        camera_position, camera_rotation = map_from_camera[:3, 3], matrix_to_quaternion(map_from_camera)
        envelope = {
            "schema_version": 2, "run_id": self.run_id, "clock_epoch": epoch,
            "timestamp_ns": timestamp, "seen_at_ns": timestamp,
            "timestamp": timestamp / 1_000_000_000, "timestamp_domain": "ROS",
            "frame_id": self.args.map_frame, "robot_frame": self.args.base_frame,
            "odom_frame": frame_name(odom_message.header.frame_id),
            "image_frame": image_message.header.frame_id,
            "depth_frame": depth_message.header.frame_id,
            "depth_timestamp_ns": depth_stamp, "odom_timestamp_ns": odom_stamp,
            "depth_delta_ns": depth_stamp-timestamp, "odom_delta_ns": odom_stamp-timestamp,
            "map_x": rx, "map_y": ry, "map_z": rz, "map_yaw": yaw,
            **{f"camera_{axis}": float(value) for axis, value in zip("xyz", camera_position)},
            **{f"camera_q{axis}": float(value) for axis, value in zip("xyzw", camera_rotation)},
            "camera_convention": "optical: x right, y down, z forward",
            "range_convention": "3D distance from camera centre, metres",
            "bearing_convention": "0 camera forward; positive left, radians",
            "depth_fraction": self.args.depth_fraction,
            "yolo_detection_count": yolo_count, "rejected_depth_count": rejected_depth,
            "detections": detections,
        }
        return envelope

    def worker(self):
        while not self.stop.is_set():
            with self.condition:
                if not self.pending:
                    self.condition.wait(timeout=0.25)
                    continue
                timestamp, arrived, image_message = self.pending[0]
                if timestamp <= self.last_published_ns:
                    self.pending.popleft()
                    continue
                remaining = 1/self.args.max_fps - (time.monotonic()-self.last_inference)
                if remaining > 0:
                    self.condition.wait(timeout=min(remaining, 0.1))
                    continue
                try:
                    snapshot = self.snapshot(timestamp)
                except LookupError as error:
                    reason = str(error)
                    if time.monotonic()-arrived < self.args.sync_wait:
                        self.condition.wait(timeout=0.05)
                        continue
                    self.pending.popleft()
                    self.counters["sync_rejected"] += 1
                    self.warn("sync", reason + "; skipped frame. Check tf/tf_static bridge keys if this persists")
                    continue
                self.pending.popleft()
                if not keyframe_needed(self.last_keyframe_pose, snapshot[6],
                                       self.args.keyframe_dist, math.radians(self.args.keyframe_angle)):
                    self.counters["motion_skipped"] += 1
                    continue
                self.last_inference = time.monotonic()
            try:
                envelope = self.process(image_message, timestamp, snapshot)
                with self.condition:
                    if snapshot[7] != self.epoch:
                        continue  # The simulator reset while inference was running.
                    next_id = self.keyframe_id + 1
                    envelope["keyframe_id"] = next_id
                    for index, detection in enumerate(envelope["detections"]):
                        detection["det_id"] = f"{self.run_id}_kf{next_id}_d{index}"
                    self.publisher.put(json.dumps(envelope, allow_nan=False).encode())
                    self.keyframe_id = next_id
                    self.last_keyframe_pose = snapshot[6]
                    self.last_published_ns = timestamp
                    self.counters["keyframes"] += 1
                classes = [d["class"] for d in envelope["detections"]]
                examples = [f"{d['class']}=({d['map_x']:.2f},{d['map_y']:.2f},{d['map_z']:.2f})"
                            for d in envelope["detections"][:3]]
                print(f"KF#{next_id} Mapped: {classes} @ robot-map "
                      f"({envelope['map_x']:.3f}, {envelope['map_y']:.3f}); "
                      f"sync depth={envelope['depth_delta_ns']/1e6:.1f}ms "
                      f"odom={envelope['odom_delta_ns']/1e6:.1f}ms; "
                      f"depth rejected={envelope['rejected_depth_count']} " + "; ".join(examples), flush=True)
            except Exception as error:
                self.warn("inference", f"{type(error).__name__}: {error}; keyframe reference was not advanced")

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--connect", default="")
    parser.add_argument("-m", "--model", default="yolov8n.pt")
    parser.add_argument("-c", "--confidence", type=float, default=0.3)
    parser.add_argument("--image-key", default="camera/color/image_raw")
    parser.add_argument("--depth-key", default="camera/depth/image_rect_raw")
    parser.add_argument("--odom-key", default="odom")
    parser.add_argument("--tf-key", default="tf")
    parser.add_argument("--tf-static-key", default="tf_static")
    parser.add_argument("--detection-key", default="tb/detections")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--max-fps", type=float, default=10.0)
    parser.add_argument("--keyframe-dist", type=float, default=0.5)
    parser.add_argument("--keyframe-angle", type=float, default=15.0)
    parser.add_argument("--enable-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-sync-ms", type=float, default=200.0)
    parser.add_argument("--sync-wait", type=float, default=2.0, help="wall seconds to await matching sensor/TF data")
    parser.add_argument("--depth-fraction", type=float, default=0.5)
    parser.add_argument("--min-depth-samples", type=int, default=9)
    for name, default in (("width", 320), ("height", 240)):
        parser.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("fx", 277.13), ("fy", 277.13), ("cx", 160.0), ("cy", 120.0)):
        parser.add_argument(f"--{name}", type=float, default=default)
    args = parser.parse_args()
    if not 0 < args.max_fps <= 10 or not 0 < args.max_sync_ms <= 200:
        parser.error("max-fps must be in (0,10]; max-sync-ms must be in (0,200]")
    if (not 0 < args.depth_fraction <= 1 or args.min_depth_samples < 1
            or args.sync_wait <= 0 or args.fx <= 0 or args.fy <= 0
            or args.width <= 0 or args.height <= 0 or not 0 <= args.confidence <= 1
            or args.keyframe_dist <= 0 or not 0 < args.keyframe_angle <= 180):
        parser.error("invalid depth, camera, confidence, or keyframe settings")
    args.map_frame, args.base_frame = frame_name(args.map_frame), frame_name(args.base_frame)
    return args


def main():
    args = arguments()
    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", json.dumps([args.connect]))
    session = zenoh.open(conf)
    publisher = session.declare_publisher(args.detection_key)
    runtime = DetectorRuntime(args, publisher)
    subscribers = [
        session.declare_subscriber(args.image_key, runtime.on_color),
        session.declare_subscriber(args.depth_key, runtime.on_depth),
        session.declare_subscriber(args.odom_key, runtime.on_odom),
        session.declare_subscriber(args.tf_key, runtime.on_tf),
        session.declare_subscriber(args.tf_static_key, lambda s: runtime.on_tf(s, True)),
    ]
    thread = None
    signal.signal(signal.SIGTERM, lambda *_: runtime.close())
    try:
        print(f"Run ID: {runtime.run_id}", flush=True)
        print(f"Subscribing RGB={args.image_key} depth={args.depth_key} odom={args.odom_key} "
              f"TF={args.tf_key} static-TF={args.tf_static_key}", flush=True)
        print("Publishing map-frame observations to: " + args.detection_key, flush=True)
        runtime.request_static_history(session)
        import torch
        from ultralytics import YOLO
        runtime.torch = torch
        runtime.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        runtime.model = YOLO(args.model)
        runtime.clip_model = None
        print(f"Loaded model: {args.model}; device={runtime.device}", flush=True)
        if args.enable_embeddings:
            import open_clip
            clip, _, runtime.clip_preprocess = open_clip.create_model_and_transforms(
                args.clip_model, pretrained=args.clip_pretrained)
            if clip.visual.output_dim != 512:
                raise ValueError("this assignment requires 512-dimensional CLIP embeddings")
            runtime.clip_model = clip.to(runtime.device).eval()
            print(f"CLIP enabled: {args.clip_model}/{args.clip_pretrained}, 512-dim on {runtime.device}", flush=True)
        thread = threading.Thread(target=runtime.worker, name="mapping-worker", daemon=True)
        thread.start()
        print("Detector running. Keep Gazebo running; move 0.5m or turn 15 degrees for a keyframe.", flush=True)
        while not runtime.stop.wait(10):
            with runtime.condition:
                counters = dict(runtime.counters)
            print("Status: " + json.dumps(counters), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()
        if thread is not None:
            thread.join(timeout=10)
        for subscriber in subscribers:
            subscriber.undeclare()
        publisher.undeclare()
        session.close()


if __name__ == "__main__":
    main()
