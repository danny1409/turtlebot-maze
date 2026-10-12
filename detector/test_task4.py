"""Task 4 checks; PostgreSQL checks run in a temporary schema.

TASK4_TEST_DB=1 python -B -m unittest -v test_task4
"""

import base64
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import psycopg2

import test_task3 as task3
from embedding_ingest import store_detection
from task4_measurements import association_metrics, error_measurements, load_records, make_labels, read_labels, replay_associations


def detection_at(position, vector=None, stamp=10_000_000_000, class_name="chair"):
    message = task3.example_message()
    message["timestamp_ns"] = stamp
    detection = message["detections"][0]
    detection.update(zip(("map_x", "map_y", "map_z"), position))
    detection["class"] = class_name
    if vector is not None:
        values = np.zeros(512, dtype="<f4")
        values[:len(vector)] = vector
        values /= np.linalg.norm(values)
        detection["embedding"] = base64.b64encode(values.tobytes()).decode("ascii")
    return message, detection


def example_record(det_id, position, vector=(1, 0), stamp=10):
    from embedding_ingest import decode_embedding, prepare_detection
    message, detection = detection_at(position, vector, stamp)
    detection["det_id"] = det_id
    detection["depth_m"] = 2.
    return {**prepare_detection(message, detection), "det_id": det_id,
            "embedding": np.asarray(decode_embedding(detection["embedding"])),
            "message": message, "detection": detection}


class Task4MeasurementTests(unittest.TestCase):
    def test_four_chair_trial_exposes_radius_and_tie_break_tradeoff(self):
        # Synthetic algorithm check, not a measurement of the Gazebo house.
        records, labels = [], {}
        for offset in (0., .55):
            for index in range(4):
                vector = np.eye(4)[index]
                key = f"chair_{index}_{offset}"
                records.append(example_record(key, (index+offset, 0, 0), vector))
                labels[key] = {"object_label": f"physical_chair_{index}"}
        small, _ = replay_associations(records, .2)
        clip, counts = replay_associations(records, .6, "clip")
        nearest, _ = replay_associations(records, .6, "nearest")
        large, _ = replay_associations(records, 1.1, "clip")
        self.assertEqual(association_metrics(small, labels)["duplicate_records"], 4)
        self.assertEqual(association_metrics(clip, labels)["duplicate_records"], 0)
        self.assertEqual(association_metrics(clip, labels)["false_merge_records"], 0)
        self.assertGreater(association_metrics(nearest, labels)["false_merge_records"], 0)
        self.assertGreater(association_metrics(large, labels)["false_merge_records"], 0)
        self.assertGreater(counts["multiple"], 0)

    def test_pair_metrics_count_splits_and_false_merges_separately(self):
        labels = {"a1": {"object_label": "a"}, "a2": {"object_label": "a"},
                  "b1": {"object_label": "b"}, "unknown": {"object_label": ""}}
        metrics = association_metrics({"a1": 1, "a2": 2, "b1": 2, "unknown": 1}, labels)
        self.assertEqual(metrics["labeled_detections"], 3)
        self.assertEqual(metrics["duplicate_records"], 1)
        self.assertEqual(metrics["false_merge_records"], 1)
        self.assertEqual(metrics["split_same_object_pairs"], 1)
        self.assertEqual(metrics["merged_different_object_pairs"], 1)
        self.assertEqual(metrics["duplicate_pair_rate"], 1.)
        self.assertEqual(metrics["false_merge_pair_rate"], .5)

    def test_missing_ground_truth_produces_missing_counts_not_zero_error(self):
        record = example_record("a", (0, 0, 0))
        measurements = error_measurements([record], {"a": {"object_label": "chair_1"}})
        self.assertEqual(measurements["planar_transform_point_error_bound_m"], {"count": 0})
        self.assertEqual(measurements["stationary_depth_noise_m"], {"count": 0})

    def test_repeat_bbox_and_opposite_view_measurements_use_verified_identity(self):
        first = example_record("a", (0, 0, 0), stamp=10)
        repeat = example_record("b", (.1, 0, 0), stamp=20)
        opposite = example_record("c", (.2, 0, 0), stamp=30)
        first["camera"] = (0, -1, 0, 0, 0, 0, 1)
        repeat["camera"] = (0, -1, 0, 0, 0, 0, 1)
        opposite["camera"] = (0, 1, 0, 0, 0, 1, 0)
        repeat["detection"]["bbox"] = [value+1 for value in first["detection"]["bbox"]]
        labels = {key: {"object_label": "verified_chair", "truth": (1.25, -2.5, .3)}
                  for key in ("a", "b", "c")}
        measurements = error_measurements([first, repeat, opposite], labels)
        self.assertEqual(measurements["bbox_centre_variation_equivalent_m"]["count"], 1)
        self.assertAlmostEqual(measurements["bbox_centre_variation_equivalent_m"]["median"], 2/277.13*np.sqrt(2))
        self.assertEqual(measurements["opposite_view_position_difference_m"]["count"], 2)
        self.assertEqual(measurements["robot_translation_error_m"]["p95"], 0.)
        self.assertEqual(measurements["same_object_running_mean_residual_m"]["count"], 2)
        self.assertAlmostEqual(measurements["same_object_running_mean_residual_m"]["median"], .125)

    def test_label_template_preserves_odom_time_and_geometry_column_alignment(self):
        import csv
        record = example_record("a", (1, 2, 3))
        record["message"]["odom_timestamp_ns"] = 25
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"labels.csv"
            make_labels([record], path)
            with path.open(newline="") as source:
                row = next(csv.DictReader(source))
            self.assertEqual(row["odom_timestamp_ns"], "25")
            self.assertEqual(float(row["map_x"]), 1.)
            self.assertEqual(float(row["bbox_xmin"]), record["detection"]["bbox"][0])
            self.assertEqual(row["true_robot_yaw"], "")
            self.assertIsNone(read_labels(path, [record])["a"]["truth"])

    def test_repeated_recording_ids_are_rejected_before_double_counting(self):
        record = example_record("same_id", (0, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"recording.jsonl"
            line = json.dumps(record["message"])+"\n"
            path.write_text(line+line)
            with self.assertRaisesRegex(ValueError, "repeated det_id"):
                load_records(path)


@unittest.skipUnless(os.environ.get("TASK4_TEST_DB") == "1", "Set TASK4_TEST_DB=1 for PostgreSQL checks")
class Task4DatabaseTests(unittest.TestCase):
    setUpClass = classmethod(task3.Task3DatabaseTests.setUpClass.__func__)
    tearDownClass = classmethod(task3.Task3DatabaseTests.tearDownClass.__func__)
    setUp = task3.Task3DatabaseTests.setUp
    fetchone = task3.Task3DatabaseTests.fetchone

    def add(self, position, vector=None, stamp=10_000_000_000, class_name="chair", radius=.35):
        message, detection = detection_at(position, vector, stamp, class_name)
        return store_detection(self.connection, message, detection, radius)

    def test_empty_candidates_create_object_and_evidence(self):
        object_id = self.add((1, 2, 3))
        self.assertEqual(self.fetchone("SELECT object_id, observation_count FROM semantic_objects"),
                         (object_id, 1))
        self.assertEqual(self.fetchone("SELECT object_id FROM object_observations"), (object_id,))

    def test_only_same_class_can_merge(self):
        first = self.add((0, 0, 0), class_name="chair")
        second = self.add((0, 0, 0), class_name="couch")
        self.assertNotEqual(first, second)
        self.assertEqual(self.fetchone("SELECT count(*) FROM semantic_objects")[0], 2)

    def test_radius_uses_xyz_and_includes_boundary(self):
        first = self.add((0, 0, 0), radius=.5)
        self.assertEqual(self.add((0, 0, .5), radius=.5), first)
        self.assertNotEqual(self.add((0, 0, .75001), radius=.5), first)

    def test_single_candidate_merges_without_appearance_threshold(self):
        first = self.add((0, 0, 0), vector=[1, 0])
        self.assertEqual(self.add((.1, 0, 0), vector=[0, 1]), first)
        self.assertEqual(self.fetchone("SELECT observation_count FROM semantic_objects")[0], 2)

    def test_multiple_candidates_choose_clip_even_when_another_is_nearer(self):
        left = self.add((0, 0, 0), vector=[1, 0], radius=.6)
        right = self.add((1, 0, 0), vector=[0, 1], radius=.6)
        self.assertNotEqual(left, right)
        chosen = self.add((.45, 0, 0), vector=[0, 1], radius=.6)
        self.assertEqual(chosen, right)
        self.assertEqual(self.fetchone("SELECT observation_count FROM semantic_objects WHERE object_id=%s",
                                      (left,))[0], 1)

    def test_exact_cosine_tie_uses_smallest_object_id(self):
        left = self.add((0, 0, 0), vector=[1, 0], radius=.6)
        self.add((1, 0, 0), vector=[1, 0], radius=.6)
        self.assertEqual(self.add((.5, 0, 0), vector=[0, 1], radius=.6), left)

    def test_positions_use_count_weighted_running_average(self):
        object_id = self.add((0, 0, 0), radius=2)
        self.add((1, .3, .1), radius=2)
        self.add((1.5, .6, .2), radius=2)
        row = self.fetchone("SELECT map_x, map_y, map_z, observation_count FROM semantic_objects WHERE object_id=%s",
                            (object_id,))
        np.testing.assert_allclose(row[:3], [2.5/3, .3, .1], atol=1e-12, rtol=0)
        self.assertEqual(row[3], 3)
        self.assertEqual(self.fetchone("SELECT count(*) FROM object_observations")[0], 3)

    def test_candidates_use_fused_position_after_previous_merge(self):
        first = self.add((0, 0, 0), radius=.4)
        self.assertEqual(self.add((.4, 0, 0), radius=.4), first)
        # Fused x=.2, latest raw x=.4. New x=.7 is .5 from the fused point.
        self.assertNotEqual(self.add((.7, 0, 0), radius=.4), first)

    def test_embedding_is_count_weighted_then_renormalized_after_each_merge(self):
        object_id = self.add((0, 0, 0), vector=[1, 0])
        self.add((0, 0, 0), vector=[0, 1])
        self.add((0, 0, 0), vector=[1, 0])
        first_mean = np.asarray([1., 1.]) / np.sqrt(2)
        expected = 2*first_mean + np.asarray([1., 0.])
        expected /= np.linalg.norm(expected)
        vector = json.loads(self.fetchone("SELECT mean_embedding::text FROM semantic_objects WHERE object_id=%s",
                                         (object_id,))[0])
        np.testing.assert_allclose(vector[:2], expected, atol=1e-6, rtol=0)
        self.assertAlmostEqual(np.linalg.norm(vector), 1., places=6)

    def test_late_observation_preserves_first_and_last_ros_times(self):
        object_id = self.add((0, 0, 0), stamp=20)
        self.add((0, 0, 0), stamp=30)
        self.add((0, 0, 0), stamp=10)
        self.assertEqual(self.fetchone("SELECT first_seen_ns,last_seen_ns,observation_count FROM semantic_objects WHERE object_id=%s",
                                      (object_id,)), (10, 30, 3))
        self.assertEqual(self.fetchone("SELECT seen_at_ns FROM object_observations ORDER BY observation_id DESC LIMIT 1"), (10,))

    def test_observation_failure_rolls_back_merge_and_embedding(self):
        object_id = self.add((0, 0, 0), vector=[1, 0], stamp=10)
        before = self.fetchone("SELECT map_x,map_y,map_z,observation_count,mean_embedding::text,last_seen_ns FROM semantic_objects")
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute("ALTER TABLE object_observations ADD CONSTRAINT test_reject CHECK (confidence < 0.5) NOT VALID")
        try:
            with self.assertRaises(psycopg2.IntegrityError):
                self.add((.2, 0, 0), vector=[0, 1], stamp=20)
            self.assertEqual(self.fetchone("SELECT map_x,map_y,map_z,observation_count,mean_embedding::text,last_seen_ns FROM semantic_objects"), before)
            self.assertEqual(self.fetchone("SELECT count(*) FROM object_observations")[0], 1)
        finally:
            with self.connection:
                with self.connection.cursor() as cursor:
                    cursor.execute("ALTER TABLE object_observations DROP CONSTRAINT test_reject")
        self.assertEqual(self.add((.2, 0, 0), vector=[0, 1], stamp=20), object_id)

    def test_zero_mean_is_rejected_without_partial_merge(self):
        self.add((0, 0, 0), vector=[1, 0])
        with self.assertRaisesRegex(ValueError, "cannot be L2-normalized"):
            self.add((0, 0, 0), vector=[-1, 0])
        self.assertEqual(self.fetchone("SELECT observation_count FROM semantic_objects"), (1,))
        self.assertEqual(self.fetchone("SELECT count(*) FROM object_observations"), (1,))

    def test_invalid_radius_does_not_change_database(self):
        for radius in (0, -.1, float("inf"), float("nan")):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                self.add((0, 0, 0), radius=radius)
        self.assertEqual(self.fetchone("SELECT count(*) FROM object_observations"), (0,))

    def test_offline_clip_replay_matches_postgresql_associations(self):
        records = [example_record("a", (0, 0, 0), [1, 0]),
                   example_record("b", (1, 0, 0), [0, 1]),
                   example_record("c", (.45, 0, 0), [0, 1]),
                   example_record("d", (.1, 0, 0), [1, 0])]
        offline, _ = replay_associations(records, .6)
        actual = {record["det_id"]: store_detection(self.connection, record["message"], record["detection"], .6)
                  for record in records}
        self.assertEqual(actual, offline)


if __name__ == "__main__":
    unittest.main()
