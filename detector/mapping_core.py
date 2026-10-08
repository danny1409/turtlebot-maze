"""Timestamp buffers, rigid transforms, and depth geometry, independent of ML/ROS."""

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    width: int = 320
    height: int = 240
    fx: float = 277.13
    fy: float = 277.13
    cx: float = 160.0
    cy: float = 120.0


# Maps camera optical coordinates (right, down, forward) to base_link.
BASE_FROM_OPTICAL = np.array([
    [0.0, 0.0, 1.0, 0.064],
    [-1.0, 0.0, 0.0, -0.065],
    [0.0, -1.0, 0.0, 0.094],
    [0.0, 0.0, 0.0, 1.0],
])


class TimedBuffer:
    """Sorted by ROS timestamp, including samples received out of order."""

    def __init__(self, capacity=256):
        self.capacity = capacity
        self.times = []
        self.values = []

    def add(self, timestamp_ns, value):
        timestamp_ns = int(timestamp_ns)
        index = bisect_left(self.times, timestamp_ns)
        if index < len(self.times) and self.times[index] == timestamp_ns:
            self.values[index] = value
        else:
            self.times.insert(index, timestamp_ns)
            self.values.insert(index, value)
        excess = len(self.times) - self.capacity
        if excess > 0:
            del self.times[:excess]
            del self.values[:excess]

    def clear(self):
        self.times.clear()
        self.values.clear()

    def nearest(self, timestamp_ns, max_delta_ns):
        index = bisect_left(self.times, timestamp_ns)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(self.times)]
        if not candidates:
            return None
        best = min(candidates, key=lambda i: abs(self.times[i] - timestamp_ns))
        if abs(self.times[best] - timestamp_ns) > max_delta_ns:
            return None
        return self.times[best], self.values[best]

    def has_reached(self, timestamp_ns):
        return bool(self.times) and self.times[-1] >= timestamp_ns

    def interpolate_transform(self, timestamp_ns):
        index = bisect_left(self.times, timestamp_ns)
        if index < len(self.times) and self.times[index] == timestamp_ns:
            return self.values[index]
        if index == 0 or index == len(self.times):
            # Do not apply a future transform or extrapolate a stale transform.
            raise LookupError("timestamp is outside the dynamic TF history")
        t0, t1 = self.times[index - 1], self.times[index]
        fraction = (timestamp_ns - t0) / (t1 - t0)
        left, right = self.values[index - 1], self.values[index]
        translation = (1 - fraction) * left[:3, 3] + fraction * right[:3, 3]
        rotation = slerp(matrix_to_quaternion(left), matrix_to_quaternion(right), fraction)
        return pose_matrix(translation, rotation)


def normalized_quaternion(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must contain four finite xyzw components")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    return q / norm


def pose_matrix(translation, quaternion):
    t = np.asarray(translation, dtype=np.float64)
    if t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError("translation must contain three finite components")
    x, y, z, w = normalized_quaternion(quaternion)
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, t
    return result


def matrix_to_quaternion(matrix):
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(r))
    if trace > 0:
        s = math.sqrt(trace + 1) * 2
        q = [(r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s,
             (r[1, 0]-r[0, 1])/s, s/4]
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q = [s/4, (r[0, 1]+r[1, 0])/s, (r[0, 2]+r[2, 0])/s,
                 (r[2, 1]-r[1, 2])/s]
        elif i == 1:
            s = math.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q = [(r[0, 1]+r[1, 0])/s, s/4, (r[1, 2]+r[2, 1])/s,
                 (r[0, 2]-r[2, 0])/s]
        else:
            s = math.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q = [(r[0, 2]+r[2, 0])/s, (r[1, 2]+r[2, 1])/s, s/4,
                 (r[1, 0]-r[0, 1])/s]
    return normalized_quaternion(q)


def slerp(left, right, fraction):
    left, right = normalized_quaternion(left), normalized_quaternion(right)
    dot = float(left @ right)
    if dot < 0:
        right, dot = -right, -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalized_quaternion(left + fraction * (right - left))
    angle = math.acos(dot)
    return (math.sin((1-fraction)*angle)*left + math.sin(fraction*angle)*right) / math.sin(angle)


class TransformBuffer:
    """A TF tree with interpolated dynamic edges and timeless static edges."""

    def __init__(self, capacity=2048):
        self.capacity = capacity
        self.dynamic = {}
        self.static = {}

    def add(self, parent, child, timestamp_ns, matrix, is_static=False):
        parent, child = parent.lstrip("/"), child.lstrip("/")
        if not parent or not child or parent == child:
            return
        edge = (parent, child)
        if is_static:
            self.static[edge] = matrix
        else:
            self.dynamic.setdefault(edge, TimedBuffer(self.capacity)).add(timestamp_ns, matrix)

    def clear_dynamic(self):
        self.dynamic.clear()

    def lookup(self, target, source, timestamp_ns):
        target, source = target.lstrip("/"), source.lstrip("/")
        if target == source:
            return np.eye(4)
        adjacency = {}
        for parent, child in self.static.keys() | self.dynamic.keys():
            edge = (parent, child)
            try:
                transform = (self.dynamic[edge].interpolate_transform(timestamp_ns)
                             if edge in self.dynamic else self.static[edge])
            except LookupError:
                continue
            adjacency.setdefault(child, []).append((parent, transform))
            adjacency.setdefault(parent, []).append((child, np.linalg.inv(transform)))
        queue = deque([(source, np.eye(4))])
        seen = {source}
        while queue:
            current, current_from_source = queue.popleft()
            for neighbor, neighbor_from_current in adjacency.get(current, []):
                if neighbor in seen:
                    continue
                neighbor_from_source = neighbor_from_current @ current_from_source
                if neighbor == target:
                    return neighbor_from_source
                seen.add(neighbor)
                queue.append((neighbor, neighbor_from_source))
        raise LookupError(f"no time-valid TF path {source} -> {target} at ROS ns {timestamp_ns}")


def decode_color(message):
    if message.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported color encoding {message.encoding}")
    height, width, step = int(message.height), int(message.width), int(message.step)
    data = bytes(message.data)
    if height <= 0 or width <= 0 or step < width * 3 or len(data) < height * step:
        raise ValueError("invalid color dimensions, step, or data length")
    frame = np.ndarray((height, width, 3), dtype=np.uint8, buffer=data,
                       strides=(step, 3, 1))
    return (frame[:, :, ::-1] if message.encoding == "rgb8" else frame).copy()


def decode_depth(message):
    if message.encoding != "32FC1":
        raise ValueError(f"expected 32FC1 depth in metres, got {message.encoding}")
    height, width, step = int(message.height), int(message.width), int(message.step)
    data = bytes(message.data)
    if height <= 0 or width <= 0 or step < width * 4 or len(data) < height * step:
        raise ValueError("invalid depth dimensions, step, or data length")
    dtype = ">f4" if message.is_bigendian else "<f4"
    return np.ndarray((height, width), dtype=dtype, buffer=data,
                      strides=(step, 4)).astype(np.float32, copy=True)


@dataclass(frozen=True)
class DepthEstimate:
    bbox: tuple
    center: tuple
    roi: tuple
    depth: float
    count: int


def median_depth(depth, bbox, fraction=0.5, min_samples=9):
    """Median of finite positive depths in the central fraction of each side."""
    if not 0 < fraction <= 1 or min_samples < 1:
        raise ValueError("invalid central fraction or minimum sample count")
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    box = np.asarray(bbox, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("bounding box must contain four finite xyxy coordinates")
    x1, y1, x2, y2 = box
    h, w = depth.shape
    x1, x2 = max(0.0, x1), min(float(w), x2)
    y1, y2 = max(0.0, y1), min(float(h), y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty or reversed bounding box")
    u, v = (x1+x2)/2, (y1+y2)/2
    hx, hy = (x2-x1)*fraction/2, (y2-y1)*fraction/2
    # Integer slicing rounds the continuous interior outwards by <1 pixel.
    a, b = max(0, math.floor(u-hx)), max(0, math.floor(v-hy))
    c, d = min(w, math.ceil(u+hx)), min(h, math.ceil(v+hy))
    samples = depth[b:d, a:c]
    valid = samples[np.isfinite(samples) & (samples > 0)]
    if valid.size < min_samples:
        raise ValueError(f"only {valid.size} valid depth samples; need {min_samples}")
    return DepthEstimate((x1, y1, x2, y2), (u, v), (a, b, c, d),
                         float(np.median(valid)), int(valid.size))


def optical_point(u, v, depth, intrinsics=Intrinsics()):
    if not np.isfinite([u, v, depth]).all() or depth <= 0:
        raise ValueError("pixel and positive depth must be finite")
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        raise ValueError("focal lengths must be positive")
    return np.array([(u-intrinsics.cx)*depth/intrinsics.fx,
                     (v-intrinsics.cy)*depth/intrinsics.fy, depth])


def back_project(u, v, depth, map_from_base, intrinsics=Intrinsics()):
    point = np.append(optical_point(u, v, depth, intrinsics), 1.0)
    return (map_from_base @ BASE_FROM_OPTICAL @ point)[:3]


def robot_pose(matrix):
    return tuple(float(matrix[i, 3]) for i in range(3)) + (
        math.atan2(float(matrix[1, 0]), float(matrix[0, 0])),)


def keyframe_needed(previous, current, distance=0.5, angle=math.radians(15)):
    if previous is None:
        return True
    translation = math.hypot(current[0]-previous[0], current[1]-previous[1])
    delta = current[3]-previous[3]
    rotation = abs(math.atan2(math.sin(delta), math.cos(delta)))
    return translation >= distance or rotation >= angle


def detection_geometry(estimate, map_from_base, intrinsics=Intrinsics()):
    u, v = estimate.center
    point_optical = optical_point(u, v, estimate.depth, intrinsics)
    point_map = back_project(u, v, estimate.depth, map_from_base, intrinsics)
    return {
        "map_x": float(point_map[0]), "map_y": float(point_map[1]),
        "map_z": float(point_map[2]), "depth_m": estimate.depth,
        "range_m": float(np.linalg.norm(point_optical)),
        "bearing_rad": math.atan2(-float(point_optical[0]), float(point_optical[2])),
        "valid_depth_samples": estimate.count,
        "depth_roi_xyxy": list(estimate.roi),
    }
