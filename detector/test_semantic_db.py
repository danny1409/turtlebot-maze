"""Pure validation tests, plus opt-in PostgreSQL/pgvector integration tests.

SEMANTIC_TEST_DB=1 enables real integration tests in a new temporary schema.
That schema is removed afterwards; existing assignment tables are untouched.
"""
import argparse
import base64
import math
import os
import unittest
import uuid

import numpy as np
from semantic_core import merge_estimate, parse_envelope, unit_vector
from semantic_db import (ObjectStore, add_database_arguments, connect_database,
                         export_database, initialize_database, run_query)


def basis(index):
    vector = np.zeros(512, dtype=np.float64)
    vector[index] = 1
    return vector


def frame(keyframe=1, class_name="chair", point=(1.0, 2.0, 0.8), embedding=None,
          timestamp=1_000_000_000, epoch=0):
    vector = basis(0) if embedding is None else unit_vector(embedding)
    return {
        "schema_version": 2, "frame_id": "map", "timestamp_domain": "ROS",
        "run_id": "unit-fixture", "clock_epoch": epoch, "keyframe_id": keyframe,
        "timestamp_ns": timestamp, "seen_at_ns": timestamp,
        "depth_timestamp_ns": timestamp, "depth_delta_ns": 0,
        "odom_timestamp_ns": timestamp - 18_000_000, "odom_delta_ns": -18_000_000,
        "map_x": -3.0, "map_y": -0.5, "map_yaw": 0.2,
        "camera_x": -2.94, "camera_y": -0.56, "camera_z": 0.194,
        "camera_qx": 0.0, "camera_qy": 0.0, "camera_qz": 0.0, "camera_qw": 1.0,
        "detections": [{
            "det_id": f"unit-fixture_kf{keyframe}_d0", "class": class_name,
            "map_x": point[0], "map_y": point[1], "map_z": point[2],
            "bbox": [10.2, 20.7, 70.1, 80.3], "confidence": 0.8,
            "range_m": 2.3, "bearing_rad": 0.1,
            "embedding": base64.b64encode(vector.astype("<f4").tobytes()).decode(),
            "embedding_dim": 512, "embedding_dtype": "<f4", "embedding_model": "fixture/model",
        }],
    }


def observation(*args, **kwargs):
    return parse_envelope(frame(*args, **kwargs))[0]


class ValidationTests(unittest.TestCase):
    def test_object_pose_is_distinct_from_robot_pose(self):
        obs = observation()
        np.testing.assert_array_equal(obs.position, [1.0, 2.0, 0.8])
        self.assertEqual(obs.robot_pose, (-3.0, -0.5, 0.2))
        self.assertEqual(obs.seen_at_ns, 1_000_000_000)
        self.assertEqual(obs.bbox, (10, 20, 71, 81))

    def test_little_endian_embedding_decodes(self):
        expected = unit_vector(np.arange(1, 513, dtype=float))
        actual = observation(embedding=expected).embedding
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
        self.assertAlmostEqual(np.linalg.norm(actual), 1.0)

    def test_legacy_frame_walltime_and_wrong_map_rejected(self):
        for field, value in (("schema_version", 1), ("timestamp_domain", "UTC"), ("frame_id", "odom")):
            with self.subTest(field=field):
                message = frame()
                message[field] = value
                with self.assertRaises(ValueError):
                    parse_envelope(message)

    def test_bad_ns_ids_and_sync_rejected(self):
        for field, value in (("seen_at_ns", 1.2), ("seen_at_ns", True),
                             ("seen_at_ns", 2**63), ("timestamp_ns", 1_000_000_000.0),
                             ("clock_epoch", -1), ("keyframe_id", 0),
                             ("odom_delta_ns", 200_000_001), ("depth_timestamp_ns", 0)):
            with self.subTest(field=field, value=value):
                message = frame()
                message[field] = value
                with self.assertRaises(ValueError):
                    parse_envelope(message)

    def test_exact_200_ms_boundary_accepted(self):
        message = frame()
        message["odom_delta_ns"] = 200_000_000
        message["odom_timestamp_ns"] = message["seen_at_ns"] + 200_000_000
        self.assertEqual(len(parse_envelope(message)), 1)

    def test_bad_embedding_shape_endian_and_norm_rejected(self):
        for field, value in (("embedding_dim", 511), ("embedding_dtype", ">f4"),
                             ("embedding", "!invalid"),
                             ("embedding", base64.b64encode(bytes(2048)).decode()),
                             ("embedding", base64.b64encode(basis(0).astype("<f4").tobytes()[:-4]).decode())):
            with self.subTest(field=field):
                message = frame()
                message["detections"][0][field] = value
                with self.assertRaises(ValueError):
                    parse_envelope(message)

    def test_nonfinite_data_and_bad_quaternion_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            message = frame()
            message["detections"][0]["map_x"] = value
            with self.assertRaises(ValueError):
                parse_envelope(message)
        message = frame()
        message["camera_qw"] = 0
        with self.assertRaises(ValueError):
            parse_envelope(message)

    def test_invalid_bbox_range_confidence_and_source_id_rejected(self):
        for field, value in (("bbox", [20, 20, 10, 30]), ("range_m", 0),
                             ("confidence", 1.1), ("det_id", "wrong-run")):
            message = frame()
            message["detections"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_envelope(message)

    def test_empty_keyframe_is_valid(self):
        message = frame()
        message["detections"] = []
        self.assertEqual(parse_envelope(message), [])

    def test_weighted_mean_and_normalized_embedding(self):
        position, embedding = merge_estimate((1, 2, 3), basis(0), 3,
                                              np.array([5, 6, 7]), basis(1))
        np.testing.assert_array_equal(position, [2, 3, 4])
        np.testing.assert_allclose(embedding[:2], np.array([3, 1]) / math.sqrt(10), atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(embedding), 1)

    def test_opposite_embeddings_do_not_make_nan(self):
        with self.assertRaises(ValueError):
            merge_estimate((0, 0, 0), basis(0), 1, np.zeros(3), -basis(0))


class PostgresIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("SEMANTIC_TEST_DB") != "1":
            raise unittest.SkipTest("set SEMANTIC_TEST_DB=1 for real PostgreSQL/pgvector tests")
        from psycopg2 import sql
        parser = argparse.ArgumentParser()
        add_database_arguments(parser)
        cls.connection = connect_database(parser.parse_args([]))
        cls.schema = "semantic_test_" + uuid.uuid4().hex
        cls.connection.autocommit = True
        with cls.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
            cursor.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(cls.schema)))
        cls.connection.autocommit = False
        initialize_database(cls.connection)

    @classmethod
    def tearDownClass(cls):
        from psycopg2 import sql
        cls.connection.rollback()
        cls.connection.autocommit = True
        with cls.connection.cursor() as cursor:
            cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))
        cls.connection.close()

    def setUp(self):
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute("TRUNCATE object_observations, semantic_objects RESTART IDENTITY")
        self.store = ObjectStore(self.connection, 0.35)

    def test_new_object_evidence_and_pose_semantics(self):
        result = self.store.ingest(observation())
        self.assertEqual(result, {"object_id": 1, "action": "created", "candidates": 0})
        data = export_database(self.connection)
        self.assertEqual(len(data["semantic_objects"]), 1)
        self.assertEqual(len(data["object_observations"]), 1)
        obj, obs = data["semantic_objects"][0], data["object_observations"][0]
        self.assertEqual(obj["map_x"], 1)
        self.assertEqual(obs["map_x"], -3)
        self.assertEqual(obs["object_x"], 1)
        self.assertEqual(obs["bbox_xmin"], 10)
        self.assertEqual(obs["association_radius_m"], 0.35)

    def test_same_class_and_full_3d_radius(self):
        self.store.ingest(observation(1, point=(0, 0, 0)))
        self.assertEqual(self.store.ingest(observation(2, point=(0.35, 0, 0)))["action"], "merged")
        self.assertEqual(self.store.ingest(observation(3, point=(0.175, 0, 0.36)))["action"], "created")
        self.assertEqual(self.store.ingest(observation(4, "couch", point=(0.175, 0, 0)))["action"], "created")

    def test_multiple_candidates_choose_embedding_not_nearest(self):
        first = self.store.ingest(observation(1, point=(0, 0, 0), embedding=basis(0)))
        second = self.store.ingest(observation(2, point=(0.6, 0, 0), embedding=basis(1)))
        # 0.28 is closer to the first object, but looks like the second.
        result = self.store.ingest(observation(3, point=(0.28, 0, 0), embedding=basis(1)))
        self.assertEqual(result["candidates"], 2)
        self.assertEqual(result["object_id"], second["object_id"])
        self.assertNotEqual(result["object_id"], first["object_id"])

    def test_running_average_and_timestamp_extrema(self):
        self.store.ingest(observation(1, point=(1, 2, 0.8), embedding=basis(0), timestamp=2_000_000_000))
        self.store.ingest(observation(2, point=(1.2, 2.2, 0.9), embedding=basis(1), timestamp=1_000_000_000))
        obj = export_database(self.connection)["semantic_objects"][0]
        np.testing.assert_allclose([obj["map_x"], obj["map_y"], obj["map_z"]], [1.1, 2.1, 0.85])
        self.assertEqual(obj["observation_count"], 2)
        np.testing.assert_allclose(obj["mean_embedding"][:2], [1 / math.sqrt(2)] * 2, atol=1e-6)
        self.assertEqual((obj["first_seen_ns"], obj["last_seen_ns"]), (1_000_000_000, 2_000_000_000))

    def test_duplicate_delivery_does_not_increment_counts(self):
        obs = observation()
        self.store.ingest(obs)
        self.assertEqual(self.store.ingest(obs)["action"], "duplicate")
        data = export_database(self.connection)
        self.assertEqual(data["semantic_objects"][0]["observation_count"], 1)
        self.assertEqual(len(data["object_observations"]), 1)

    def test_evidence_failure_rolls_back_object_creation(self):
        obs = observation()
        obs.bbox = (10, 20, 5, 25)  # PostgreSQL CHECK fails after INSERT of the object.
        import psycopg2
        with self.assertRaises(psycopg2.IntegrityError):
            self.store.ingest(obs)
        data = export_database(self.connection)
        self.assertEqual(data["semantic_objects"], [])
        self.assertEqual(data["object_observations"], [])

    def test_embedding_model_change_is_explicit(self):
        self.store.ingest(observation(1))
        obs = observation(2)
        obs.embedding_model = "different/model"
        with self.assertRaises(ValueError):
            self.store.ingest(obs)
        self.assertEqual(len(export_database(self.connection)["object_observations"]), 1)

    def test_all_required_queries_and_bound_sql_values(self):
        self.store.ingest(observation(1, point=(1, 2, 0.8)))
        self.store.ingest(observation(2, point=(1.1, 2, 0.8), epoch=1, timestamp=100_000_000))
        self.store.ingest(observation(3, "couch", point=(4, 5, 0.8)))
        nearest = run_query(self.connection, "nearest", (0, 0, "chair"))
        self.assertEqual(nearest[0]["object_id"], 1)
        self.assertAlmostEqual(nearest[0]["distance"], math.hypot(1.05, 2))
        region = run_query(self.connection, "region", (0, 2, 0, 3))
        self.assertEqual([row["class_name"] for row in region], ["chair"])
        self.assertEqual(run_query(self.connection, "histogram"), [
            {"class_name": "chair", "object_count": 1}, {"class_name": "couch", "object_count": 1}])
        poses = run_query(self.connection, "poses", (1,))
        self.assertEqual([p["clock_epoch"] for p in poses], [0, 1])
        self.assertEqual([p["map_x"] for p in poses], [-3, -3])
        self.assertEqual(run_query(self.connection, "nearest", (0, 0, "chair' OR 1=1 --")), [])
        self.assertEqual(len(export_database(self.connection)["semantic_objects"]), 2)


if __name__ == "__main__":
    unittest.main()
