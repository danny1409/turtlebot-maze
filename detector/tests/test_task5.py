"""Task 5 integration checks in a private PostgreSQL schema.

TASK5_TEST_DB=1 python -B -m unittest -v test_task5
Uses the existing test_task3 setup; production tables are never changed.
"""

import json
import math
import os
import unittest

import test_task3 as task3
from embedding_ingest import store_detection
from embedding_ingest import export_database, run_query


@unittest.skipUnless(os.environ.get("TASK5_TEST_DB") == "1", "Set TASK5_TEST_DB=1 for PostgreSQL checks")
class Task5DatabaseTests(unittest.TestCase):
    setUpClass = classmethod(task3.Task3DatabaseTests.setUpClass.__func__)
    tearDownClass = classmethod(task3.Task3DatabaseTests.tearDownClass.__func__)
    setUp = task3.Task3DatabaseTests.setUp

    def add(self, class_name, position, stamp=10_000_000_000, robot=(1.25, -2.5, .3)):
        message = task3.example_message()
        message["timestamp_ns"] = stamp
        message.update(zip(("map_x", "map_y", "map_yaw"), robot))
        detection = message["detections"][0]
        detection["class"] = class_name
        detection.update(zip(("map_x", "map_y", "map_z"), position))
        return store_detection(self.connection, message, detection, .35)

    def test_nearest_filters_class_and_uses_horizontal_distance(self):
        self.add("bed", (0, 0, 0))
        near = self.add("chair", (1, 0, 100))
        self.add("chair", (2, 0, 0))
        rows = run_query(self.connection, "nearest", (0, 0, "chair"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["object_id"], near)
        self.assertEqual(rows[0]["distance"], 1.)

    def test_nearest_tie_is_deterministic_and_missing_class_is_empty(self):
        first = self.add("chair", (-1, 0, 0))
        self.add("chair", (1, 0, 0))
        self.assertEqual(run_query(self.connection, "nearest", (0, 0, "chair"))[0]["object_id"], first)
        self.assertEqual(run_query(self.connection, "nearest", (0, 0, "vase")), [])

    def test_class_value_with_quote_and_sql_text_is_bound_as_data(self):
        literal = "chair' OR TRUE --"
        self.add("chair", (0, 0, 0))
        selected = self.add(literal, (4, 5, 0))
        rows = run_query(self.connection, "nearest", (0, 0, literal))
        self.assertEqual([row["object_id"] for row in rows], [selected])
        self.assertEqual(rows[0]["class_name"], literal)

    def test_rectangle_includes_boundaries_and_orders_by_class_then_id(self):
        right = self.add("chair", (2, 3, 0))
        left = self.add("bed", (-2, -3, 0))
        self.add("chair", (2.001, 3, 1))
        rows = run_query(self.connection, "region", (-2, 2, -3, 3))
        self.assertEqual([row["object_id"] for row in rows], [left, right])

    def test_histogram_counts_objects_instead_of_observations(self):
        self.add("chair", (0, 0, 0))
        self.add("chair", (.1, 0, 0))
        self.add("chair", (2, 0, 0))
        self.add("bed", (0, 0, 0))
        self.add("couch", (0, 0, 0))
        rows = run_query(self.connection, "histogram")
        self.assertEqual(rows, [{"class_name": "chair", "object_count": 2},
                                {"class_name": "bed", "object_count": 1},
                                {"class_name": "couch", "object_count": 1}])

    def test_pose_history_preserves_every_observation_and_sorts_ros_time(self):
        obj = self.add("chair", (0, 0, 0), stamp=30, robot=(1, 2, .1))
        self.add("chair", (.1, 0, 0), stamp=10, robot=(3, 4, .2))
        self.add("chair", (.2, 0, 0), stamp=20, robot=(5, 6, .3))
        self.add("bed", (0, 0, 0), stamp=15)
        rows = run_query(self.connection, "poses", (obj,))
        self.assertEqual([row["seen_at_ns"] for row in rows], [10, 20, 30])
        self.assertEqual([(row["map_x"], row["map_y"], row["map_yaw"]) for row in rows],
                         [(3., 4., .2), (5., 6., .3), (1., 2., .1)])
        self.assertEqual(run_query(self.connection, "poses", (9999,)), [])

    def test_export_preserves_all_columns_vectors_and_evidence(self):
        obj = self.add("chair", (0, 0, .7), stamp=10)
        self.add("chair", (.1, 0, .7), stamp=20)
        data = export_database(self.connection)
        self.assertEqual((data["frame_id"], data["timestamp_domain"]), ("map", "ROS"))
        self.assertEqual(len(data["semantic_objects"]), 1)
        self.assertEqual(len(data["object_observations"]), 2)
        stored = data["semantic_objects"][0]
        self.assertEqual(set(stored), {"object_id", "class_name", "map_x", "map_y", "map_z",
                                      "observation_count", "mean_embedding", "first_seen_ns", "last_seen_ns"})
        self.assertEqual((stored["observation_count"], stored["first_seen_ns"], stored["last_seen_ns"]), (2, 10, 20))
        self.assertEqual(len(stored["mean_embedding"]), 512)
        self.assertAlmostEqual(math.sqrt(sum(x*x for x in stored["mean_embedding"])), 1., places=6)
        for evidence in data["object_observations"]:
            self.assertEqual(evidence["object_id"], obj)
            self.assertEqual(len(evidence), 21)
            self.assertEqual(len(evidence["embedding"]), 512)
            self.assertEqual(evidence["camera_qx"], -.5)
        json.dumps(data, allow_nan=False)

    def test_empty_database_exports_two_empty_lists(self):
        data = export_database(self.connection)
        self.assertEqual(data["semantic_objects"], [])
        self.assertEqual(data["object_observations"], [])


if __name__ == "__main__":
    unittest.main()
