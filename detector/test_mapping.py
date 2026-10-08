"""Run with: python -m unittest -v test_mapping (numpy; wire tests need pycdr2/Zenoh)."""

from contextlib import redirect_stdout
import io
import json
import math
import struct
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np

from mapping_core import (
    BASE_FROM_OPTICAL, TimedBuffer, TransformBuffer, back_project,
    decode_color, decode_depth, detection_geometry, keyframe_needed,
    matrix_to_quaternion, median_depth, pose_matrix,
)

try:
    from ros_cdr import Odometry, TFMessage
    from object_detector import DetectorRuntime
    WIRE_AVAILABLE = True
except ImportError:
    WIRE_AVAILABLE = False


def yaw_quaternion(angle):
    return (0, 0, math.sin(angle/2), math.cos(angle/2))


class CDRWriter:
    """An independent XCDR1 writer: align relative to the encapsulated data."""
    def __init__(self):
        self.body = bytearray()

    def number(self, format_code, value):
        size = struct.calcsize(format_code)
        self.body.extend(b'\0' * (-len(self.body) % size))
        self.body.extend(struct.pack('<' + format_code, value))

    def string(self, text):
        data = text.encode() + b'\0'
        self.number('I', len(data))
        self.body.extend(data)

    def header(self, sec, nanosec, frame):
        self.number('i', sec)
        self.number('I', nanosec)
        self.string(frame)

    def doubles(self, values):
        for value in values:
            self.number('d', value)

    def finish(self):
        return b'\x00\x01\x00\x00' + bytes(self.body)


class GeometryTests(unittest.TestCase):
    def test_hand_computed_complete_chain_off_center(self):
        # u=160+277.13/4, v=120-277.13/10, depth=2 -> optical=(.5,-.2,2).
        # Optical->base=(2.064,-.565,.294).
        # Robot at (1,2,0), yaw=+90: odom=(1.565,4.064,.294).
        # map<-odom: translation=(10,-3,.5), yaw=-90 -> (14.064,-4.565,.794).
        map_from_odom = pose_matrix((10, -3, .5), yaw_quaternion(-math.pi/2))
        odom_from_base = pose_matrix((1, 2, 0), yaw_quaternion(math.pi/2))
        actual = back_project(229.2825, 92.287, 2, map_from_odom @ odom_from_base)
        np.testing.assert_allclose(actual, [14.064, -4.565, .794], atol=1e-9, rtol=0)

    def test_transform_keeps_nonplanar_rotation(self):
        # A +90 degree roll rotates base (2.064,-.565,.294) to
        # (2.064,-.294,-.565), before adding robot translation (1,2,3).
        matrix = pose_matrix((1, 2, 3), (math.sqrt(.5), 0, 0, math.sqrt(.5)))
        actual = back_project(229.2825, 92.287, 2, matrix)
        np.testing.assert_allclose(actual, [3.064, 1.706, 2.435], atol=1e-9, rtol=0)

    def test_median_ignores_background_border_and_invalid_pixels(self):
        depth = np.full((20, 20), 9, dtype=np.float32)
        depth[5:15, 5:15] = 2
        depth[5, 5:9] = [0, -1, np.nan, np.inf]
        result = median_depth(depth, [0, 0, 20, 20])
        self.assertEqual(result.roi, (5, 5, 15, 15))
        self.assertEqual(result.count, 96)
        self.assertEqual(result.depth, 2)

    def test_rejects_insufficient_valid_depth(self):
        depth = np.zeros((20, 20), dtype=np.float32)
        depth[6, 6] = 2
        with self.assertRaisesRegex(ValueError, "only 1 valid"):
            median_depth(depth, [0, 0, 20, 20])

    def test_camera_distance_and_left_positive_bearing(self):
        estimate = median_depth(np.full((240, 320), 2), [219.2825, 82.287, 239.2825, 102.287])
        result = detection_geometry(estimate, np.eye(4))
        self.assertAlmostEqual(result['range_m'], math.sqrt(4.29))
        self.assertAlmostEqual(result['bearing_rad'], math.atan2(-.5, 2))
        camera_q = matrix_to_quaternion(BASE_FROM_OPTICAL)
        np.testing.assert_allclose(pose_matrix((0, 0, 0), camera_q)[:3, :3],
                                   BASE_FROM_OPTICAL[:3, :3], atol=1e-12)

    def test_padded_big_endian_depth_and_color_rows(self):
        data = struct.pack('>ff', 1, 2) + b'pad!' + struct.pack('>ff', 3, 4) + b'pad!'
        message = SimpleNamespace(encoding='32FC1', height=2, width=2, step=12,
                                  is_bigendian=1, data=data)
        np.testing.assert_array_equal(decode_depth(message), [[1, 2], [3, 4]])
        color = SimpleNamespace(encoding='rgb8', height=2, width=1, step=4,
                                data=bytes([10, 20, 30, 99, 40, 50, 60, 99]))
        np.testing.assert_array_equal(decode_color(color), [[[30, 20, 10]], [[60, 50, 40]]])

    def test_keyframe_yaw_wrap_and_thresholds(self):
        previous = (1, 2, 0, math.radians(179))
        self.assertFalse(keyframe_needed(previous, (1, 2, 0, math.radians(-179))))
        self.assertTrue(keyframe_needed(previous, (1.5, 2, 0, math.radians(-179))))
        self.assertTrue(keyframe_needed(previous, (1, 2, 0, math.radians(-160))))


class TimestampTests(unittest.TestCase):
    def test_nearest_uses_timestamps_after_out_of_order_arrivals(self):
        buffer = TimedBuffer()
        buffer.add(1_100_000_000, 'later')
        buffer.add(900_000_000, 'earlier-arrived-last')
        self.assertEqual(buffer.nearest(1_080_000_000, 200_000_000), (1_100_000_000, 'later'))

    def test_sync_limit_is_inclusive_200ms(self):
        buffer = TimedBuffer()
        buffer.add(1_000_000_000, 'measurement')
        self.assertIsNotNone(buffer.nearest(1_200_000_000, 200_000_000))
        self.assertIsNone(buffer.nearest(1_200_000_001, 200_000_000))

    def test_tf_interpolates_wraparound_and_static_child(self):
        buffer = TransformBuffer()
        buffer.add('map', 'odom', 100, pose_matrix((0, 0, 0), yaw_quaternion(math.radians(179))))
        buffer.add('map', 'odom', 300, pose_matrix((2, 0, 0), yaw_quaternion(math.radians(-179))))
        buffer.add('odom', 'base_link', 0, pose_matrix((1, 0, .1), (0, 0, 0, 1)), True)
        actual = buffer.lookup('map', 'base_link', 200)
        np.testing.assert_allclose(actual[:3, 3], [0, 0, .1], atol=1e-12)
        np.testing.assert_allclose(actual[:3, :3], np.diag([-1, -1, 1]), atol=1e-12)
        inverse = buffer.lookup('base_link', 'map', 200)
        np.testing.assert_allclose(inverse @ actual, np.eye(4), atol=1e-12)

    def test_dynamic_tf_cannot_extrapolate(self):
        buffer = TransformBuffer()
        buffer.add('map', 'odom', 100, np.eye(4))
        buffer.add('map', 'odom', 300, np.eye(4))
        for timestamp in (99, 301):
            with self.assertRaises(LookupError):
                buffer.lookup('map', 'odom', timestamp)


@unittest.skipUnless(WIRE_AVAILABLE, 'run inside detector container for pycdr2/Zenoh checks')
class WireAndRuntimeTests(unittest.TestCase):
    def test_odom_fixed_covariances_preserve_twist_alignment(self):
        writer = CDRWriter()
        writer.header(86, 796_000_000, 'odom')
        writer.string('base_footprint')
        writer.doubles([1, 2, 3, 0, 0, 0, 1])
        writer.doubles([.125 + i for i in range(36)])
        writer.doubles([12.5, -2, 3, 4, 5, 6])
        writer.doubles([50.25 + i for i in range(36)])
        message = Odometry.deserialize(writer.finish())
        self.assertEqual(message.child_frame_id, 'base_footprint')
        self.assertEqual(message.pose.covariance[35], 35.125)
        self.assertEqual(message.twist.twist.linear.x, 12.5)
        self.assertEqual(message.twist.twist.angular.z, 6)
        self.assertEqual(message.twist.covariance[35], 85.25)

    def test_tfmessage_ros_cdr_layout(self):
        writer = CDRWriter()
        writer.number('I', 1)
        writer.header(154, 801_000_000, 'map')
        writer.string('odom')
        writer.doubles([-.001, -.077, 0, 0, 0, -.012, 1])
        message = TFMessage.deserialize(writer.finish())
        self.assertEqual(message.transforms[0].header.frame_id, 'map')
        self.assertEqual(message.transforms[0].child_frame_id, 'odom')
        self.assertAlmostEqual(message.transforms[0].transform.translation.y, -.077)

    def runtime(self, publisher):
        args = SimpleNamespace(width=320, height=240, fx=277.13, fy=277.13, cx=160, cy=120,
            max_sync_ms=200, run_id='test', map_frame='map', base_frame='base_link',
            max_fps=10, sync_wait=.1, keyframe_dist=.5, keyframe_angle=15,
            confidence=.3, depth_fraction=.5, min_depth_samples=9)
        runtime = DetectorRuntime(args, publisher)
        runtime.device, runtime.clip_model = 'cpu', None
        box = SimpleNamespace(xyxy=np.array([[219.2825, 82.287, 239.2825, 102.287]]),
                              cls=np.array([0]), conf=np.array([.9]))
        runtime.model = SimpleNamespace(names={0: 'chair'}, predict=lambda *_a, **_k:
                                       [SimpleNamespace(boxes=[box])])
        runtime.transforms.add('map', 'odom', 0, np.eye(4))
        runtime.transforms.add('map', 'odom', 1000, np.eye(4))
        runtime.transforms.add('base_footprint', 'base_link', 0,
                               pose_matrix((0, 0, .1), (0, 0, 0, 1)), True)
        header = SimpleNamespace(frame_id='odom')
        odom = SimpleNamespace(header=header, child_frame_id='base_footprint')
        image = SimpleNamespace(header=SimpleNamespace(frame_id='camera_optical'),
                                width=320, height=240, step=960, encoding='rgb8',
                                data=bytes(320*240*3))
        for timestamp in (100, 200):
            runtime.depth.add(timestamp, (image, np.full((240, 320), 2, dtype=np.float32)))
            runtime.odom.add(timestamp, (odom, np.eye(4)))
        return runtime, image

    def test_envelope_uses_ros_nanoseconds_and_child_frame_offset(self):
        runtime, image = self.runtime(None)
        envelope = runtime.process(image, 100, runtime.snapshot(100))
        self.assertEqual(envelope['timestamp_ns'], 100)
        self.assertEqual(envelope['frame_id'], 'map')
        self.assertEqual(envelope['odom_frame'], 'odom')
        self.assertEqual(envelope['depth_delta_ns'], 0)
        self.assertAlmostEqual(envelope['camera_z'], .194)
        point = envelope['detections'][0]
        np.testing.assert_allclose([point['map_x'], point['map_y'], point['map_z']],
                                   [2.064, -.565, .394], atol=1e-9, rtol=0)

    def test_publish_failure_does_not_advance_keyframe_reference(self):
        class Publisher:
            def __init__(self):
                self.calls, self.payloads, self.done = 0, [], threading.Event()
            def put(self, payload):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError('simulated publication failure')
                self.payloads.append(json.loads(payload))
                self.done.set()
        publisher = Publisher()
        runtime, image = self.runtime(publisher)
        for stamp in (100, 200):
            runtime.pending.append((stamp, time.monotonic(), image))
        thread = threading.Thread(target=runtime.worker, daemon=True)
        # Keep the deliberately simulated failure out of the user's test log.
        with redirect_stdout(io.StringIO()):
            thread.start()
            try:
                self.assertTrue(publisher.done.wait(3), 'same-pose second frame must retry after failure')
            finally:
                runtime.close()
                thread.join(2)
        self.assertEqual(runtime.keyframe_id, 1)
        self.assertEqual(publisher.payloads[0]['keyframe_id'], 1)
        self.assertEqual(publisher.payloads[0]['timestamp_ns'], 200)


if __name__ == '__main__':
    unittest.main()
