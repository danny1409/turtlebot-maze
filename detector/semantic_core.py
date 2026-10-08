"""Validate detector schema v2; association math independent of PostgreSQL."""
import base64
from dataclasses import dataclass
import math
import numpy as np


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def integer(value, name, minimum=0, maximum=2**63-1):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def text_value(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def unit_vector(values):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (512,) or not np.isfinite(result).all():
        raise ValueError("embedding must contain 512 finite numbers")
    length = float(np.linalg.norm(result))
    if length < 1e-12:
        raise ValueError("embedding has zero length")
    return result / length


def decode_embedding(detection):
    if detection.get("embedding_dim") != 512 or detection.get("embedding_dtype") != "<f4":
        raise ValueError("expected a 512-dimensional little-endian float32 CLIP embedding")
    try:
        raw = base64.b64decode(detection["embedding"], validate=True)
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("invalid base64 embedding") from error
    if len(raw) != 512 * 4:
        raise ValueError("embedding byte length must be 2048")
    vector = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if not np.isfinite(vector).all() or not 0.98 <= np.linalg.norm(vector) <= 1.02:
        raise ValueError("detector embedding must already be finite and L2-normalized")
    return unit_vector(vector)


def vector_literal(vector):
    # This string is bound as a SQL parameter followed by ::vector.
    return "[" + ",".join(format(float(x), ".9g") for x in vector) + "]"


def merge_estimate(position, embedding, count, new_position, new_embedding):
    """Assignment running average, then renormalize the updated embedding."""
    if count < 1:
        raise ValueError("existing observation count must be positive")
    position = (count * np.asarray(position, dtype=float) + new_position) / (count + 1)
    embedding = unit_vector(count * unit_vector(embedding) + unit_vector(new_embedding))
    if not np.isfinite(position).all():
        raise ValueError("nonfinite merged position")
    return position, embedding


@dataclass
class Observation:
    source_det_id: str
    class_name: str
    embedding_model: str
    position: np.ndarray  # OBJECT estimate in map
    embedding: np.ndarray
    seen_at_ns: int
    robot_pose: tuple  # ROBOT map_x, map_y, map_yaw
    camera_pose: tuple  # camera map x/y/z, optical quaternion xyzw
    bearing_rad: float
    range_m: float
    bbox: tuple
    confidence: float
    run_id: str
    clock_epoch: int
    keyframe_id: int


def parse_envelope(message):
    """Reject old reference messages that called odometry coordinates 'map'."""
    if not isinstance(message, dict):
        raise ValueError("message must be a JSON object")
    if (message.get("schema_version") != 2 or message.get("frame_id") != "map"
            or message.get("timestamp_domain") != "ROS"):
        raise ValueError("requires detector schema_version=2, frame_id=map, timestamp_domain=ROS")
    run_id = text_value(message.get("run_id"), "run_id")
    epoch = integer(message.get("clock_epoch"), "clock_epoch", maximum=2**31-1)
    keyframe_id = integer(message.get("keyframe_id"), "keyframe_id", minimum=1)
    timestamp = integer(message.get("seen_at_ns"), "seen_at_ns")
    if integer(message.get("timestamp_ns"), "timestamp_ns") != timestamp:
        raise ValueError("timestamp_ns must equal seen_at_ns")
    robot = tuple(finite(message.get(k), k) for k in ("map_x", "map_y", "map_yaw"))
    camera = tuple(finite(message.get(k), k) for k in (
        "camera_x", "camera_y", "camera_z", "camera_qx", "camera_qy", "camera_qz", "camera_qw"))
    if not 0.99 <= np.linalg.norm(camera[3:]) <= 1.01:
        raise ValueError("camera quaternion must have unit length")
    for key in ("depth_delta_ns", "odom_delta_ns"):
        delta = integer(message.get(key), key, minimum=-200_000_000, maximum=200_000_000)
        source_key = "depth_timestamp_ns" if key == "depth_delta_ns" else "odom_timestamp_ns"
        if integer(message.get(source_key), source_key) - timestamp != delta:
            raise ValueError(f"{key} disagrees with its source timestamp")
    detections = message.get("detections")
    if not isinstance(detections, list):
        raise ValueError("detections must be a list")
    observations = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            raise ValueError("each detection must be an object")
        source_id = text_value(detection.get("det_id"), "det_id")
        if source_id != f"{run_id}_kf{keyframe_id}_d{index}":
            raise ValueError("det_id must agree with run, keyframe, and detection index")
        position = np.array([finite(detection.get(k), k) for k in ("map_x", "map_y", "map_z")])
        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("bbox must contain four xyxy coordinates")
        bbox = [finite(x, "bbox") for x in bbox]
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("invalid xyxy bounding box")
        bbox = (math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3]))
        if max(bbox) >= 2**31:
            raise ValueError("bounding box is outside SQL integer range")
        confidence = finite(detection.get("confidence"), "confidence")
        range_m = finite(detection.get("range_m"), "range_m")
        bearing = finite(detection.get("bearing_rad"), "bearing_rad")
        if not 0 <= confidence <= 1 or range_m <= 0 or not -math.pi <= bearing <= math.pi:
            raise ValueError("invalid confidence, range, or bearing")
        observations.append(Observation(
            source_id, text_value(detection.get("class"), "class"),
            text_value(detection.get("embedding_model"), "embedding_model"), position,
            decode_embedding(detection), timestamp, robot, camera, bearing, range_m,
            bbox, confidence, run_id, epoch, keyframe_id))
    return observations
