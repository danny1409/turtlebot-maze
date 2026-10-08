"""Transactional pgvector persistence and the four required bound queries."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

from semantic_core import merge_estimate, vector_literal


def add_database_arguments(parser):
    parser.add_argument("--db-host", default=os.environ.get("PGHOST", "127.0.0.1"))
    parser.add_argument("--db-port", type=int, default=int(os.environ.get("PGPORT", "5436")))
    parser.add_argument("--db-name", default=os.environ.get("PGDATABASE", "vectordb"))
    parser.add_argument("--db-user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--db-password", default=os.environ.get("PGPASSWORD", "postgres"))


def connect_database(args):
    import psycopg2
    return psycopg2.connect(host=args.db_host, port=args.db_port, dbname=args.db_name,
                           user=args.db_user, password=args.db_password, connect_timeout=5,
                           application_name="semantic-map-assignment")


def initialize_database(connection):
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(Path(__file__).with_name("semantic_schema.sql").read_text())
            # Existing incompatible tables are left untouched; fail with a clear error.
            cursor.execute("SELECT embedding_model FROM semantic_objects LIMIT 0")
            cursor.execute("SELECT source_det_id, object_x, object_y, object_z, "
                           "run_id, clock_epoch, keyframe_id, association_radius_m "
                           "FROM object_observations LIMIT 0")


def class_lock_key(class_name):
    raw = hashlib.sha256(("semantic-map/class/" + class_name).encode()).digest()
    return int.from_bytes(raw[:8], "big", signed=True)


class ObjectStore:
    def __init__(self, connection, radius):
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("association radius must be positive and finite")
        self.connection, self.radius = connection, float(radius)

    def ingest(self, observation):
        """One detection = one atomic object update plus one evidence row.

        Same class, 3D radius; multiple candidates: smallest pgvector cosine
        distance, then object_id for a deterministic exact tie. No threshold.
        """
        o = observation
        embedding = vector_literal(o.embedding)
        with self.connection:
            with self.connection.cursor() as cursor:
                # Serialize creation and fusion for a class even with two ingestors.
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", (class_lock_key(o.class_name),))
                cursor.execute("SELECT object_id FROM object_observations WHERE source_det_id = %s",
                               (o.source_det_id,))
                duplicate = cursor.fetchone()
                if duplicate:
                    return {"object_id": duplicate[0], "action": "duplicate"}
                cursor.execute("SELECT DISTINCT embedding_model FROM semantic_objects WHERE class_name = %s",
                               (o.class_name,))
                if any(row[0] != o.embedding_model for row in cursor.fetchall()):
                    raise ValueError("embedding model changed for this class; use a separate trial database")
                x, y, z = (float(v) for v in o.position)
                cursor.execute("""
                    SELECT object_id, map_x, map_y, map_z, observation_count,
                           mean_embedding::text, mean_embedding <=> %s::vector AS cosine_distance
                    FROM semantic_objects
                    WHERE class_name = %s
                      AND power(map_x - %s, 2) + power(map_y - %s, 2) + power(map_z - %s, 2) <= %s
                    ORDER BY cosine_distance, object_id
                    FOR UPDATE
                """, (embedding, o.class_name, x, y, z, self.radius**2))
                candidates = cursor.fetchall()
                if candidates:
                    row = candidates[0]
                    object_id, count = row[0], row[4]
                    position, mean = merge_estimate(row[1:4], json.loads(row[5]), count, o.position, o.embedding)
                    cursor.execute("""
                        UPDATE semantic_objects SET map_x=%s, map_y=%s, map_z=%s,
                            observation_count=%s, mean_embedding=%s::vector,
                            first_seen_ns=LEAST(first_seen_ns, %s), last_seen_ns=GREATEST(last_seen_ns, %s)
                        WHERE object_id=%s
                    """, (*[float(v) for v in position], count + 1, vector_literal(mean),
                          o.seen_at_ns, o.seen_at_ns, object_id))
                    action = "merged"
                else:
                    cursor.execute("""
                        INSERT INTO semantic_objects
                            (class_name, map_x, map_y, map_z, observation_count, mean_embedding,
                             first_seen_ns, last_seen_ns, embedding_model)
                        VALUES (%s, %s, %s, %s, 1, %s::vector, %s, %s, %s) RETURNING object_id
                    """, (o.class_name, x, y, z, embedding, o.seen_at_ns, o.seen_at_ns, o.embedding_model))
                    object_id = cursor.fetchone()[0]
                    action = "created"
                cursor.execute("""
                    INSERT INTO object_observations
                        (object_id, seen_at_ns, map_x, map_y, map_yaw,
                         camera_x, camera_y, camera_z, camera_qx, camera_qy, camera_qz, camera_qw,
                         bearing_rad, range_m, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
                         confidence, embedding, source_det_id, run_id, clock_epoch, keyframe_id,
                         object_x, object_y, object_z, association_radius_m)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s,
                            %s, %s, %s, %s)
                """, (object_id, o.seen_at_ns, *o.robot_pose, *o.camera_pose, o.bearing_rad,
                      o.range_m, *o.bbox, o.confidence, embedding, o.source_det_id, o.run_id,
                      o.clock_epoch, o.keyframe_id, x, y, z, self.radius))
                return {"object_id": object_id, "action": action, "candidates": len(candidates)}


QUERIES = {
    "nearest": """
        SELECT object_id, class_name, map_x, map_y, map_z,
               sqrt(power(map_x - %s, 2) + power(map_y - %s, 2)) AS distance
        FROM semantic_objects WHERE class_name = %s ORDER BY distance, object_id LIMIT 1
    """,
    "region": """
        SELECT object_id, class_name, map_x, map_y, map_z, observation_count
        FROM semantic_objects WHERE map_x BETWEEN %s AND %s AND map_y BETWEEN %s AND %s
        ORDER BY class_name, object_id
    """,
    "histogram": """
        SELECT class_name, count(*) AS object_count FROM semantic_objects
        GROUP BY class_name ORDER BY object_count DESC, class_name
    """,
    "poses": """
        SELECT observation_id, seen_at_ns, map_x, map_y, map_yaw,
               camera_x, camera_y, camera_z, bearing_rad, range_m, confidence,
               run_id, clock_epoch, keyframe_id
        FROM object_observations WHERE object_id = %s
        ORDER BY run_id, clock_epoch, seen_at_ns, observation_id
    """,
}


def fetch_rows(cursor):
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def run_query(connection, name, parameters=()):
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(QUERIES[name], parameters)
            return fetch_rows(cursor)


def export_database(connection):
    # Objects and observations must come from the same consistent snapshot.
    with connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            cursor.execute("SELECT object_id, class_name, map_x, map_y, map_z, observation_count, "
                           "mean_embedding::text, first_seen_ns, last_seen_ns, embedding_model "
                           "FROM semantic_objects ORDER BY object_id")
            objects = fetch_rows(cursor)
            for item in objects:
                item["mean_embedding"] = json.loads(item["mean_embedding"])
            cursor.execute("SELECT observation_id, object_id, seen_at_ns, map_x, map_y, map_yaw, "
                           "camera_x, camera_y, camera_z, camera_qx, camera_qy, camera_qz, camera_qw, "
                           "bearing_rad, range_m, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, "
                           "confidence, embedding::text, source_det_id, run_id, clock_epoch, keyframe_id, "
                           "object_x, object_y, object_z, association_radius_m "
                           "FROM object_observations ORDER BY observation_id")
            observations = fetch_rows(cursor)
            for item in observations:
                item["embedding"] = json.loads(item["embedding"])
    return {"schema_version": 2, "frame_id": "map", "timestamp_domain": "ROS",
            "range_convention": "3D distance from camera centre, metres",
            "bearing_convention": "0 camera forward; positive left, radians",
            "association_rule": "same class, 3D radius; highest CLIP cosine similarity for multiple candidates",
            "semantic_objects": objects, "object_observations": observations}
