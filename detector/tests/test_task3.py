"""Task 3 input and PostgreSQL tests.

Run the database checks in the detector container after initialization:
  TASK3_TEST_DB=1 python -B -m unittest -v test_task3
Database checks use a temporary schema, not the assignment's public tables.
"""

import base64
import json
import os
import unittest
import uuid

import numpy as np
import psycopg2
from psycopg2 import sql

from embedding_ingest import decode_embedding, initialize_database, prepare_detection, store_detection


def example_message():
    vector = np.zeros(512, dtype="<f4")
    vector[:2] = [.6, .8]
    return {
        "frame_id": "map", "timestamp_domain": "ROS", "timestamp_ns": 5_987_654_321,
        "keyframe_id": 1, "map_x": 1.25, "map_y": -2.5, "map_yaw": .3,
        "camera_x": 1.33, "camera_y": -2.47, "camera_z": .094,
        "camera_qx": -.5, "camera_qy": .5, "camera_qz": -.5, "camera_qw": .5,
        "detections": [{
            "class": "chair", "confidence": .8, "bbox": [10.2, 20.8, 50.1, 60.6],
            "map_x": 10.0, "map_y": 20.0, "map_z": .7,
            "range_m": 2.3, "bearing_rad": -.2,
            "embedding": base64.b64encode(vector.tobytes()).decode("ascii"),
            "embedding_dim": 512, "embedding_dtype": "<f4",
        }],
    }


class Task3InputTests(unittest.TestCase):
    def test_object_and_robot_coordinates_are_distinct(self):
        message = example_message()
        data = prepare_detection(message, message["detections"][0])
        self.assertEqual(data["position"], (10.0, 20.0, .7))
        self.assertEqual(data["robot"], (1.25, -2.5, .3))
        self.assertEqual(data["camera"], (1.33, -2.47, .094, -.5, .5, -.5, .5))
        self.assertEqual(data["stamp"], 5_987_654_321)

    def test_float_box_is_enclosed_by_integer_pixels(self):
        message = example_message()
        self.assertEqual(prepare_detection(message, message["detections"][0])["bbox"], (10, 20, 51, 61))

    def test_embedding_preserves_normalized_512_vector(self):
        vector = decode_embedding(example_message()["detections"][0]["embedding"])
        self.assertEqual(len(vector), 512)
        np.testing.assert_allclose(vector[:2], [.6, .8], atol=1e-7, rtol=0)
        self.assertAlmostEqual(np.linalg.norm(vector), 1.0, places=6)

    def test_embedding_rejects_wrong_dimension_and_invalid_base64(self):
        for encoded in ("not valid base64!", base64.b64encode(np.ones(511, dtype="<f4")).decode()):
            with self.subTest(encoded_length=len(encoded)), self.assertRaises(ValueError):
                decode_embedding(encoded)

    def test_embedding_rejects_zero_nonunit_and_nonfinite_vectors(self):
        for first in (0, 2, np.nan, np.inf):
            vector = np.zeros(512, dtype="<f4")
            vector[0] = first
            with self.subTest(first=first), self.assertRaises(ValueError):
                decode_embedding(base64.b64encode(vector.tobytes()).decode())

    def test_timestamp_must_be_integer_ros_nanoseconds(self):
        for timestamp in (5.987654321, True, -1, 2**63):
            message = example_message()
            message["timestamp_ns"] = timestamp
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                prepare_detection(message, message["detections"][0])

    def test_odom_frame_and_wall_clock_are_rejected(self):
        for field, value in (("frame_id", "odom"), ("timestamp_domain", "UNIX")):
            message = example_message()
            message[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare_detection(message, message["detections"][0])

    def test_invalid_geometry_confidence_and_camera_quaternion_are_rejected(self):
        for field, value in (("map_x", np.nan), ("range_m", 0), ("confidence", 1.2),
                             ("bbox", [10, 20, 10, 30])):
            message = example_message()
            message["detections"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare_detection(message, message["detections"][0])
        message = example_message()
        message["camera_qw"] = 2
        with self.assertRaises(ValueError):
            prepare_detection(message, message["detections"][0])


@unittest.skipUnless(os.environ.get("TASK3_TEST_DB") == "1", "Set TASK3_TEST_DB=1 for PostgreSQL checks")
class Task3DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connection = psycopg2.connect(
            host=os.environ.get("PGHOST", "127.0.0.1"), port=os.environ.get("PGPORT", "5436"),
            dbname=os.environ.get("PGDATABASE", "semantic_assignment"),
            user=os.environ.get("PGUSER", "postgres"), password=os.environ.get("PGPASSWORD", "postgres"),
            connect_timeout=5,
        )
        cls.schema = "task3_test_" + uuid.uuid4().hex
        try:
            with cls.connection:
                with cls.connection.cursor() as cursor:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
                    cursor.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(cls.schema)))
            initialize_database(cls.connection)
        except Exception:
            cls.connection.rollback()
            with cls.connection:
                with cls.connection.cursor() as cursor:
                    cursor.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(cls.schema)))
            cls.connection.close()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.connection.rollback()
        try:
            with cls.connection:
                with cls.connection.cursor() as cursor:
                    cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))
        finally:
            cls.connection.close()

    def setUp(self):
        self.connection.rollback()
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(sql.SQL("TRUNCATE {}.object_observations, {}.semantic_objects RESTART IDENTITY").format(
                    sql.Identifier(self.schema), sql.Identifier(self.schema)))

    def fetchone(self, statement, parameters=()):
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute(statement, parameters)
                return cursor.fetchone()

    def test_object_and_observation_are_linked_and_preserve_ros_pose_fields(self):
        message = example_message()
        object_id = store_detection(self.connection, message, message["detections"][0])
        obj = self.fetchone("SELECT object_id, class_name, map_x, map_y, map_z, observation_count, "
                            "first_seen_ns, last_seen_ns, vector_dims(mean_embedding) FROM semantic_objects")
        self.assertEqual(obj, (object_id, "chair", 10.0, 20.0, .7, 1, 5_987_654_321, 5_987_654_321, 512))
        observation = self.fetchone("SELECT object_id, seen_at_ns, map_x, map_y, map_yaw, "
                                    "camera_x, camera_y, camera_z, camera_qx, camera_qy, camera_qz, camera_qw, "
                                    "bearing_rad, range_m, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, "
                                    "confidence, embedding::text FROM object_observations")
        expected = (object_id, 5_987_654_321, 1.25, -2.5, .3, 1.33, -2.47, .094,
                    -.5, .5, -.5, .5, -.2, 2.3, 10, 20, 51, 61)
        self.assertEqual(observation[:18], expected)
        self.assertAlmostEqual(observation[18], .8, places=6)
        self.assertAlmostEqual(np.linalg.norm(json.loads(observation[19])), 1.0, places=6)

    def test_observation_failure_rolls_back_object_insert(self):
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute("ALTER TABLE object_observations ADD CONSTRAINT test_reject CHECK (confidence < 0.5)")
        message = example_message()
        try:
            with self.assertRaises(psycopg2.IntegrityError):
                store_detection(self.connection, message, message["detections"][0])
            self.assertEqual(self.fetchone("SELECT count(*) FROM semantic_objects")[0], 0)
            self.assertEqual(self.fetchone("SELECT count(*) FROM object_observations")[0], 0)
        finally:
            with self.connection:
                with self.connection.cursor() as cursor:
                    cursor.execute("ALTER TABLE object_observations DROP CONSTRAINT test_reject")
        store_detection(self.connection, message, message["detections"][0])
        self.assertEqual(self.fetchone("SELECT count(*) FROM object_observations")[0], 1)

    def test_cascade_delete_removes_linked_observation(self):
        message = example_message()
        object_id = store_detection(self.connection, message, message["detections"][0])
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM semantic_objects WHERE object_id=%s", (object_id,))
        self.assertEqual(self.fetchone("SELECT count(*) FROM object_observations")[0], 0)

    def test_database_enforces_512_dimensions(self):
        short_vector = json.dumps([1] + [0]*510)
        with self.assertRaises(psycopg2.DataError):
            with self.connection:
                with self.connection.cursor() as cursor:
                    cursor.execute("INSERT INTO semantic_objects "
                                   "(class_name,map_x,map_y,map_z,observation_count,mean_embedding,first_seen_ns,last_seen_ns) "
                                   "VALUES (%s,0,0,0,1,%s::vector,0,0)", ("chair", short_vector))

    def test_repeat_initialization_preserves_data_and_required_indexes(self):
        message = example_message()
        store_detection(self.connection, message, message["detections"][0])
        initialize_database(self.connection)
        self.assertEqual(self.fetchone("SELECT count(*) FROM object_observations")[0], 1)
        with self.connection:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT indexname FROM pg_indexes WHERE schemaname=%s", (self.schema,))
                names = {row[0] for row in cursor.fetchall()}
        self.assertIn("semantic_objects_class_name_idx", names)
        self.assertIn("object_observations_object_id_idx", names)


if __name__ == "__main__":
    unittest.main()
