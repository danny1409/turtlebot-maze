"""Task 2 checks. Run in the detector container: python -m unittest -v test_task2

The off-centre hand calculation below uses an absolute tolerance of 1e-9 m.
No simulation, model download, or GPU inference is needed for these tests.
"""

import math
import unittest

import numpy as np

from object_detector import (
    Header, Image, Time, MAX_SYNC_NS, R_CAMERA_FROM_OPTICAL,
    back_project, camera_quaternion, decode_depth, lookup_transform,
    median_box_depth, nearest_sample, optical_point, pose_matrix,
)


def yaw_quaternion(degrees):
    angle = math.radians(degrees) / 2
    return (0, 0, math.sin(angle), math.cos(angle))


class Task2Tests(unittest.TestCase):
    def test_complete_chain_against_off_centre_hand_calculation(self):
        # Bounding-box centre (229.2825, 92.287), depth 2 m:
        # optical = (0.5, -0.2, 2)
        # camera body axes = (2, -0.5, 0.2)
        # base_link, after camera offset = (2.064, -0.565, 0.294)
        # robot in odom: (1, 2, 0), yaw +90 deg
        # object in odom = (1.565, 4.064, 0.294)
        # map <- odom: (10, -3, 0.5), yaw -90 deg
        # object in map = (14.064, -4.565, 0.794)
        bbox = [219.2825, 82.287, 239.2825, 102.287]
        depth, _, _ = median_box_depth(np.full((240, 320), 2.0), bbox)
        u, v = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        odom_from_base = pose_matrix((1, 2, 0), yaw_quaternion(90))
        supplied_tf = pose_matrix((10, -3, 0.5), yaw_quaternion(-90))
        map_from_odom = lookup_transform(
            "map", "odom", 5_000_000_000,
            {("map", "odom"): [(5_000_000_000, supplied_tf)]}, {},
        )
        result = back_project(u, v, depth, map_from_odom @ odom_from_base)
        np.testing.assert_allclose(result, [14.064, -4.565, 0.794], atol=1e-9, rtol=0)

    def test_optical_axis_signs_away_from_image_centre(self):
        point = optical_point(229.2825, 92.287, 2.0)
        np.testing.assert_allclose(point, [.5, -.2, 2, 1], atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            R_CAMERA_FROM_OPTICAL @ point, [2, -.5, .2, 1], atol=1e-12, rtol=0,
        )

    def test_timestamp_selection_ignores_arrival_order(self):
        # Newest arrival is deliberately NOT the nearest timestamp.
        samples = [(1_190_000_000, "late"), (990_000_000, "nearest"),
                   (800_000_000, "last arrival")]
        self.assertEqual(nearest_sample(samples, 1_000_000_000), (990_000_000, "nearest"))

    def test_timestamp_limit_is_inclusive(self):
        for sign in [-1, 1]:
            stamp = 1_000_000_000 + sign * MAX_SYNC_NS
            self.assertIsNotNone(nearest_sample([(stamp, "sample")], 1_000_000_000))
            self.assertIsNone(nearest_sample([(stamp+sign, "sample")], 1_000_000_000))
        self.assertIsNone(nearest_sample([], 1_000_000_000))

    def test_median_uses_only_valid_central_samples(self):
        depth = np.full((240, 320), 20.0)  # Background outside the box interior.
        depth[110:130, 150:170] = np.nan
        depth[110:113, 150:153] = np.array([1, 2, 2, 2, 2, 2, 2, 2, 20]).reshape(3, 3)
        depth[120, 160] = 0  # The single centre pixel would be unusable.
        depth[120, 161] = np.inf
        depth[120, 162] = -1
        value, count, roi = median_box_depth(depth, [140, 100, 180, 140])
        self.assertEqual((value, count, roi), (2.0, 9, [150, 110, 170, 130]))

    def test_too_few_valid_depth_samples_rejects_detection(self):
        depth = np.full((240, 320), np.nan)
        depth[110, 150:158] = 2  # Eight valid pixels; minimum is nine.
        with self.assertRaises(ValueError):
            median_box_depth(depth, [140, 100, 180, 140])

    def test_depth_cdr_bytes_padding_and_endianness(self):
        for bigendian in [0, 1]:
            dtype = ">f4" if bigendian else "<f4"
            values = np.array([[1.5, 2.25], [3.0, 4.5]], dtype=dtype)
            data = b"".join(row.tobytes() + b"PAD!" for row in values)
            message = Image(Header(Time(7, 123), "camera_depth_optical_frame"),
                            2, 2, "32FC1", bigendian, 12, list(data))
            decoded = Image.deserialize(message.serialize())
            np.testing.assert_array_equal(decode_depth(decoded), values)
            self.assertEqual(decoded.header.stamp.nanosec, 123)

    def test_missing_map_to_odom_cannot_be_assumed_identity(self):
        with self.assertRaises(LookupError):
            lookup_transform("map", "odom", 0, {}, {})
        np.testing.assert_array_equal(lookup_transform("map", "map", 0, {}, {}), np.eye(4))

    def test_stale_dynamic_tf_rejects_frame(self):
        with self.assertRaises(LookupError):
            lookup_transform("map", "odom", 0,
                             {("map", "odom"): [(MAX_SYNC_NS+1, np.eye(4))]}, {})

    def test_base_footprint_offset_and_inverse_tf(self):
        footprint_from_base = pose_matrix((0, 0, .01), yaw_quaternion(0))
        edge = {("base_footprint", "base_link"): footprint_from_base}
        forward = lookup_transform("base_footprint", "base_link", 123, {}, edge)
        inverse = lookup_transform("base_link", "base_footprint", 123, {}, edge)
        np.testing.assert_allclose(forward @ inverse, np.eye(4), atol=1e-12, rtol=0)
        result = back_project(160, 120, 2, forward)
        np.testing.assert_allclose(result, [2.064, -.065, .104], atol=1e-12, rtol=0)

    def test_camera_quaternion_describes_optical_orientation(self):
        for yaw in [0, 90, 180, -90]:
            robot = pose_matrix((1, 2, 3), yaw_quaternion(yaw))
            q = camera_quaternion(robot)
            self.assertAlmostEqual(np.linalg.norm(q), 1.0)
            expected = (robot @ R_CAMERA_FROM_OPTICAL)[:3, :3]
            np.testing.assert_allclose(pose_matrix((0, 0, 0), q)[:3, :3],
                                       expected, atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
