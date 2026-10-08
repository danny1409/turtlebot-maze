"""ROS 2 wire schemas for pycdr2; this module does not require rclpy."""

from dataclasses import dataclass
from typing import List

from pycdr2 import IdlStruct
from pycdr2.types import array, float64, int32, uint8, uint32


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
    data: List[uint8]


@dataclass
class Point(IdlStruct, typename="geometry_msgs/msg/Point"):
    x: float64
    y: float64
    z: float64


@dataclass
class Vector3(IdlStruct, typename="geometry_msgs/msg/Vector3"):
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
    # ROS uses float64[36], NOT a variable-length sequence with a length prefix.
    covariance: array[float64, 36]


@dataclass
class Twist(IdlStruct, typename="geometry_msgs/msg/Twist"):
    linear: Vector3
    angular: Vector3


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
    translation: Vector3
    rotation: Quaternion


@dataclass
class TransformStamped(IdlStruct, typename="geometry_msgs/msg/TransformStamped"):
    header: Header
    child_frame_id: str
    transform: Transform


@dataclass
class TFMessage(IdlStruct, typename="tf2_msgs/msg/TFMessage"):
    transforms: List[TransformStamped]


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def frame_name(name):
    return name.lstrip("/")


def xyz(vector):
    return (float(vector.x), float(vector.y), float(vector.z))


def xyzw(quaternion):
    return tuple(float(getattr(quaternion, key)) for key in ("x", "y", "z", "w"))
