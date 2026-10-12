#!/usr/bin/env python3
"""Tasks 3–5 semantic storage, association, measurements, queries, and map.

Keep the reference main() and nested Zenoh callback. Store the Task 2
detector's object estimate and observation together in PostgreSQL/pgvector.
Associate same-class detections by 3D map radius, then use CLIP to break ties.
The SQL schema and reference main()/nested-callback structure are unchanged.
Task 4 tools and Task 5 commands are ordinary functions in this same file.
"""

import argparse
import base64
import csv
import itertools
import json
import math
import sys
from collections import deque
from pathlib import Path
from threading import Event, Lock
import time

import numpy as np
import psycopg2
import zenoh


def decode_embedding(b64_str: str) -> list[float]:
    """Decode the detector's normalized, little-endian float32 CLIP vector."""
    raw = base64.b64decode(b64_str, validate=True)
    if len(raw) != 512 * 4:
        raise ValueError("Expected a 512-dimensional float32 CLIP embedding")
    embedding = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains a non-finite value")
    if not np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-4, rtol=0):
        raise ValueError("CLIP embedding must already be L2-normalized")
    return embedding.tolist()


def prepare_detection(message, detection):
    """Keep object coordinates separate from the robot and camera map poses."""
    if message.get("frame_id") != "map" or message.get("timestamp_domain") != "ROS":
        raise ValueError("Expected the Task 2 detector's map-frame, ROS-time envelope")
    stamp = message["timestamp_ns"]
    if type(stamp) is not int or not 0 <= stamp <= 2**63 - 1:
        raise ValueError("timestamp_ns must be a non-negative integer ROS timestamp")

    robot = tuple(float(message[name]) for name in ("map_x", "map_y", "map_yaw"))
    camera = tuple(float(message[name]) for name in (
        "camera_x", "camera_y", "camera_z", "camera_qx", "camera_qy", "camera_qz", "camera_qw",
    ))
    position = tuple(float(detection[name]) for name in ("map_x", "map_y", "map_z"))
    bearing, distance = float(detection["bearing_rad"]), float(detection["range_m"])
    confidence = float(detection["confidence"])
    if not all(math.isfinite(value) for value in (*robot, *camera, *position, bearing, distance, confidence)):
        raise ValueError("Geometry and confidence must be finite")
    if distance <= 0 or not 0 <= confidence <= 1:
        raise ValueError("Range must be positive and confidence must be between 0 and 1")
    if not math.isclose(math.sqrt(sum(value * value for value in camera[3:])), 1.0, abs_tol=1e-4):
        raise ValueError("Camera orientation must be a unit xyzw quaternion")

    bbox = detection["bbox"]
    if len(bbox) != 4:
        raise ValueError("bbox must contain xmin, ymin, xmax, ymax")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid detection bounding box")
    # SQL uses INTEGER pixels. Enclose the floating-point YOLO box rather than
    # shrinking it; Task 2 already used its full-precision centre for geometry.
    pixels = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
    if not isinstance(detection["class"], str) or not detection["class"].strip():
        raise ValueError("class must be a non-empty COCO label")
    if detection.get("embedding_dim", 512) != 512 or detection.get("embedding_dtype", "<f4") != "<f4":
        raise ValueError("Expected a 512-dimensional little-endian float32 embedding")
    embedding = decode_embedding(detection["embedding"])
    return {
        "class_name": detection["class"], "stamp": stamp,
        "position": position, "robot": robot, "camera": camera,
        "bearing": bearing, "range": distance, "bbox": pixels, "confidence": confidence,
        "embedding": json.dumps(embedding, allow_nan=False, separators=(",", ":")),
    }


def initialize_database(connection):
    schema = Path(__file__).with_name("semantic_schema.sql").read_text(encoding="utf-8")
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(schema)


def store_detection(connection, message, detection, radius_m=0.50):
    """Atomically associate/update an object and insert this observation.

    The radius is Euclidean distance in map XYZ, including the boundary.
    The selected calibration default is 0.50 m, as documented in REPORT.md.
    An explicit --radius overrides it for comparison trials.
    """
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Association radius must be finite and positive")
    data = prepare_detection(message, detection)
    with connection:
        with connection.cursor() as cursor:
            # Also serialize same-class writers from separate worker processes.
            # This covers the empty-candidate case where no object row can be locked.
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))",
                           ("semantic_objects:" + data["class_name"],))
            cursor.execute(
                """SELECT object_id, map_x, map_y, map_z, observation_count,
                          mean_embedding::text
                   FROM semantic_objects
                   WHERE class_name = %s
                     AND (map_x - %s)^2 + (map_y - %s)^2 + (map_z - %s)^2 <= %s
                   ORDER BY mean_embedding <=> %s::vector, object_id
                   FOR UPDATE""",
                (data["class_name"], *data["position"], radius_m * radius_m, data["embedding"]),
            )
            candidates = cursor.fetchall()
            if not candidates:
                cursor.execute(
                    """INSERT INTO semantic_objects
                       (class_name, map_x, map_y, map_z, observation_count,
                        mean_embedding, first_seen_ns, last_seen_ns)
                       VALUES (%s, %s, %s, %s, 1, %s::vector, %s, %s)
                       RETURNING object_id""",
                    (data["class_name"], *data["position"], data["embedding"], data["stamp"], data["stamp"]),
                )
                object_id = cursor.fetchone()[0]
            else:
                # One candidate merges unconditionally. For multiple candidates,
                # ORDER BY <=> selects highest cosine similarity; ID breaks exact ties.
                object_id, x, y, z, count, old_embedding = candidates[0]
                position = [(count * old + new) / (count + 1)
                            for old, new in zip((x, y, z), data["position"])]
                embedding = (count * np.asarray(json.loads(old_embedding), dtype=float)
                             + np.asarray(json.loads(data["embedding"]), dtype=float)) / (count + 1)
                norm = np.linalg.norm(embedding)
                if not math.isfinite(norm) or norm < 1e-12:
                    raise ValueError("Mean embedding cannot be L2-normalized")
                mean_embedding = json.dumps((embedding / norm).tolist(), allow_nan=False,
                                            separators=(",", ":"))
                cursor.execute(
                    """UPDATE semantic_objects
                       SET map_x=%s, map_y=%s, map_z=%s, mean_embedding=%s::vector,
                           observation_count=observation_count + 1,
                           first_seen_ns=LEAST(first_seen_ns, %s),
                           last_seen_ns=GREATEST(last_seen_ns, %s)
                       WHERE object_id=%s""",
                    (*position, mean_embedding, data["stamp"], data["stamp"], object_id),
                )
            cursor.execute(
                """INSERT INTO object_observations
                   (object_id, seen_at_ns, map_x, map_y, map_yaw,
                    camera_x, camera_y, camera_z, camera_qx, camera_qy, camera_qz, camera_qw,
                    bearing_rad, range_m, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax,
                    confidence, embedding)
                   VALUES (%s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s, %s, %s::vector)""",
                (object_id, data["stamp"], *data["robot"], *data["camera"],
                 data["bearing"], data["range"], *data["bbox"], data["confidence"], data["embedding"]),
            )
    return object_id


def main():
    # Keep the original ingest arguments; optional commands share this file.
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode in ("capture-views", "depth-noise", "labels", "evaluate", "replay"):
        measurement_main()
        return
    if mode == "query":
        query_main()
        return
    if mode == "map":
        map_main()
        return
    parser = argparse.ArgumentParser(
        description="Zenoh → Tasks 3–4 semantic object database",
        epilog="Other commands: capture-views, depth-noise, labels, evaluate, replay, query, map. "
               "Use COMMAND --help for its options.",
    )
    parser.add_argument("-e", "--connect", default="tcp/localhost:7447", help="Zenoh endpoint")
    parser.add_argument("--key", default="tb/detections", help="Zenoh subscription key")
    parser.add_argument("--pg-host", default="localhost", help="PostgreSQL host")
    parser.add_argument("--pg-port", type=int, default=5436, help="PostgreSQL port")
    parser.add_argument("--pg-db", default="vectordb", help="PostgreSQL database; use a fresh Task 4 database for calibration")
    parser.add_argument("--pg-user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--pg-password", default="postgres", help="PostgreSQL password")
    parser.add_argument("--init-only", action="store_true", help="Create the extension/tables, then exit")
    parser.add_argument("--radius", type=float, default=0.50,
                        help="3D association radius in metres (default: %(default)s)")
    parser.add_argument("--record", type=Path, help="Append raw keyframes to JSONL for radius measurements/replay")
    args = parser.parse_args()
    if not args.init_only and (args.radius is None or not math.isfinite(args.radius) or args.radius <= 0):
        parser.error("--radius must be finite and positive")

    connection = psycopg2.connect(
        host=args.pg_host, port=args.pg_port, dbname=args.pg_db,
        user=args.pg_user, password=args.pg_password, connect_timeout=5,
    )
    session = subscriber = None
    insert_count = 0
    db_lock = Lock()
    recording = None
    try:
        initialize_database(connection)
        print(f"Database ready: {args.pg_host}:{args.pg_port}/{args.pg_db}", flush=True)
        if args.init_only:
            print("semantic_objects and object_observations are ready.", flush=True)
            return
        if args.record:
            args.record.parent.mkdir(parents=True, exist_ok=True)
            recording = args.record.open("a", encoding="utf-8")
            print(f"Recording keyframes: {args.record}", flush=True)

        def detection_callback(sample):
            nonlocal insert_count
            try:
                message = json.loads(sample.payload.to_bytes())
                if not isinstance(message, dict) or not isinstance(message.get("detections"), list):
                    raise ValueError("Expected a Task 2 keyframe envelope")
            except (ValueError, TypeError) as error:
                print(f"Rejected message: {error}", flush=True)
                return
            # Serialize the complete callback so JSONL replay retains ingest order.
            with db_lock:
                if recording is not None:
                    recording.write(json.dumps(message, allow_nan=False, separators=(",", ":")) + "\n")
                    recording.flush()
                stored = 0
                for detection in message["detections"]:
                    try:
                        object_id = store_detection(connection, message, detection, args.radius)
                        insert_count += 1
                        stored += 1
                        print(f"KF#{message.get('keyframe_id')} {detection['class']}: "
                              f"object #{object_id} + observation stored", flush=True)
                    except (ValueError, TypeError, KeyError, psycopg2.Error) as error:
                        print(f"Skipped detection: {error}", flush=True)
                print(f"KF#{message.get('keyframe_id')}: stored={stored}, total={insert_count}", flush=True)

        config = zenoh.Config()
        if args.connect:
            config.insert_json5("connect/endpoints", json.dumps([args.connect]))
        session = zenoh.open(config)
        subscriber = session.declare_subscriber(args.key, detection_callback)
        print(f"Subscribed to: {args.key}", flush=True)
        print(f"Task 4 association running; 3D radius={args.radius:g} m.", flush=True)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
    finally:
        if subscriber is not None:
            subscriber.undeclare()
        if session is not None:
            session.close()
        if recording is not None:
            recording.close()
        connection.close()
        print(f"Shutdown. Total observations stored: {insert_count}", flush=True)


def load_records(path):
    records = []
    seen_ids = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        message = json.loads(line)
        for detection in message["detections"]:
            data = prepare_detection(message, detection)
            det_id = detection["det_id"]
            if not isinstance(det_id, str) or not det_id or det_id in seen_ids:
                raise ValueError(f"Missing or repeated det_id on line {line_number}; use one recording per run")
            seen_ids.add(det_id)
            records.append({
                **data, "det_id": det_id, "message": message, "detection": detection,
                "embedding": np.asarray(decode_embedding(detection["embedding"]), dtype=float),
            })
    return records


def make_labels(records, path):
    fields = ("det_id", "object_label", "class_name", "timestamp_ns", "odom_timestamp_ns", "map_x", "map_y", "map_z",
              "robot_x", "robot_y", "robot_yaw", "camera_x", "camera_y", "camera_z",
              "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "depth_m",
              "true_robot_x", "true_robot_y", "true_robot_yaw")
    with Path(path).open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in records:
            box = record["detection"]["bbox"]
            writer.writerow(dict(zip(fields, (
                record["det_id"], "", record["class_name"], record["stamp"],
                record["message"].get("odom_timestamp_ns", record["stamp"]), *record["position"],
                *record["robot"], *record["camera"][:3], *box,
                record["detection"].get("depth_m", ""), "", "", "",
            ))))
    print(f"Created {path}; fill object_label with verified physical IDs, such as dining_chair_1.")


def read_labels(path, records):
    known = {record["det_id"] for record in records}
    labels = {}
    with Path(path).open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            key = row["det_id"]
            if key not in known or key in labels:
                raise ValueError(f"Unknown or repeated det_id in labels: {key}")
            row["object_label"] = row["object_label"].strip()
            truth_fields = ("true_robot_x", "true_robot_y", "true_robot_yaw")
            values = [row.get(field, "").strip() for field in truth_fields]
            if any(values) and not all(values):
                raise ValueError("Supply all three ground-truth robot pose fields, or leave all blank")
            row["truth"] = tuple(float(value) for value in values) if all(values) else None
            if row["truth"] is not None and not all(math.isfinite(value) for value in row["truth"]):
                raise ValueError("Ground-truth robot pose must be finite, in map, yaw in radians")
            labels[key] = row
    return labels


def replay_associations(records, radius, rule="clip"):
    """Offline version of the exact running-average rule; nearest is an ablation.

    Round fused embeddings to float32 after updates, matching pgvector storage.
    This function does not write to the assignment database.
    """
    if not math.isfinite(radius) or radius <= 0 or rule not in ("clip", "nearest"):
        raise ValueError("Invalid radius or tie-break rule")
    objects, assignments, candidates_seen = [], {}, {"zero": 0, "one": 0, "multiple": 0}
    for record in records:
        position = np.asarray(record["position"], dtype=float)
        candidates = [index for index, obj in enumerate(objects)
                      if obj["class_name"] == record["class_name"]
                      and np.sum((obj["position"] - position)**2) <= radius**2]
        candidates_seen["zero" if not candidates else "one" if len(candidates) == 1 else "multiple"] += 1
        if not candidates:
            index = len(objects)
            objects.append({"class_name": record["class_name"], "position": position,
                            "embedding": record["embedding"].copy(), "count": 1})
        else:
            if rule == "clip":
                def score(index):
                    embedding = objects[index]["embedding"]
                    return (-float(np.dot(embedding, record["embedding"]) /
                                   (np.linalg.norm(embedding) * np.linalg.norm(record["embedding"]))), index)
            else:
                def score(index):
                    return (float(np.linalg.norm(objects[index]["position"] - position)), index)
            index = min(candidates, key=score)
            obj = objects[index]
            count = obj["count"]
            mean = (count*obj["embedding"] + record["embedding"]) / (count + 1)
            norm = np.linalg.norm(mean)
            if not math.isfinite(norm) or norm < 1e-12:
                raise ValueError("Mean embedding cannot be L2-normalized in this replay")
            obj["position"] = (count*obj["position"] + position) / (count + 1)
            obj["embedding"] = (mean / norm).astype(np.float32).astype(float)
            obj["count"] += 1
        assignments[record["det_id"]] = index + 1
    return assignments, candidates_seen


def association_metrics(assignments, labels):
    labeled = [(key, row["object_label"]) for key, row in labels.items()
               if key in assignments and row["object_label"]]
    by_identity, by_object = {}, {}
    for key, identity in labeled:
        by_identity.setdefault(identity, set()).add(assignments[key])
        by_object.setdefault(assignments[key], set()).add(identity)
    same_pairs = different_pairs = split_pairs = merged_pairs = 0
    for (key_a, identity_a), (key_b, identity_b) in itertools.combinations(labeled, 2):
        if identity_a == identity_b:
            same_pairs += 1
            split_pairs += assignments[key_a] != assignments[key_b]
        else:
            different_pairs += 1
            merged_pairs += assignments[key_a] == assignments[key_b]
    return {
        "labeled_detections": len(labeled), "physical_objects": len(by_identity),
        "duplicate_records": sum(max(len(ids)-1, 0) for ids in by_identity.values()),
        "false_merge_records": sum(len(ids) > 1 for ids in by_object.values()),
        "same_object_pairs": same_pairs, "split_same_object_pairs": split_pairs,
        "different_object_pairs": different_pairs, "merged_different_object_pairs": merged_pairs,
        "duplicate_pair_rate": split_pairs/same_pairs if same_pairs else None,
        "false_merge_pair_rate": merged_pairs/different_pairs if different_pairs else None,
    }


def stats(values):
    return {"count": len(values), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "maximum": float(np.max(values))} if values else {"count": 0}


def error_measurements(records, labels, depth_csv=None):
    near_depth, box_variation, opposite_spread, all_spread, running_residual = [], [], [], [], []
    pose_xy, pose_yaw, transform_bound = [], [], []
    groups = {}
    for record in records:
        row = labels.get(record["det_id"], {})
        if row.get("object_label"):
            groups.setdefault((record["class_name"], row["object_label"]), []).append(record)
        if row.get("truth") is not None:
            truth = row["truth"]
            xy_error = math.hypot(record["robot"][0]-truth[0], record["robot"][1]-truth[1])
            yaw_error = abs(math.atan2(math.sin(record["robot"][2]-truth[2]), math.cos(record["robot"][2]-truth[2])))
            lever = math.hypot(record["position"][0]-record["robot"][0], record["position"][1]-record["robot"][1])
            pose_xy.append(xy_error)
            pose_yaw.append(math.degrees(yaw_error))
            transform_bound.append(xy_error + 2*lever*math.sin(yaw_error/2))
    for group in groups.values():
        anchor = np.median([record["position"] for record in group], axis=0)
        total = np.zeros(3)
        for count, record in enumerate(group):
            point = np.asarray(record["position"])
            if count:
                running_residual.append(float(np.linalg.norm(point-total/count)))
            total += point
        for a, b in itertools.combinations(group, 2):
            if a["stamp"] == b["stamp"]:
                continue
            spread = float(np.linalg.norm(np.asarray(a["position"])-b["position"]))
            all_spread.append(spread)
            ca, cb = np.asarray(a["camera"][:3]), np.asarray(b["camera"][:3])
            dot = abs(float(np.dot(a["camera"][3:], b["camera"][3:])))
            rotation = 2*math.acos(min(1., dot))
            da, db = a["detection"].get("depth_m"), b["detection"].get("depth_m")
            if np.linalg.norm(ca-cb) <= .05 and rotation <= math.radians(3) and da and db:
                near_depth.append(abs(da-db))
                u_a, v_a = np.asarray(a["detection"]["bbox"]).reshape(2, 2).mean(axis=0)
                u_b, v_b = np.asarray(b["detection"]["bbox"]).reshape(2, 2).mean(axis=0)
                box_variation.append((da+db)/2/277.13 * math.hypot(u_a-u_b, v_a-v_b))
            va, vb = ca[:2]-anchor[:2], cb[:2]-anchor[:2]
            if np.linalg.norm(va) > 1e-9 and np.linalg.norm(vb) > 1e-9:
                angle = math.acos(float(np.clip(np.dot(va, vb)/(np.linalg.norm(va)*np.linalg.norm(vb)), -1, 1)))
                if angle >= math.pi/2:
                    opposite_spread.append(spread)
    depth_noise = {"count": 0}
    if depth_csv:
        with Path(depth_csv).open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        depths = np.asarray([float(row["depth_m"]) for row in rows])
        if len(depths) < 2 or not np.isfinite(depths).all() or np.any(depths <= 0):
            raise ValueError("Depth-noise CSV needs at least two finite positive stationary samples")
        factors = [math.sqrt(1 + ((float(row["u"])-160)/277.13)**2 + ((float(row["v"])-120)/277.13)**2) for row in rows]
        errors = np.abs(depths-np.median(depths))
        depth_noise = {**stats(errors.tolist()), "std_m": float(np.std(depths, ddof=1)),
                       "map_point_error_p95_m": float(np.percentile(errors*np.asarray(factors), 95))}
    return {
        "stationary_depth_noise_m": depth_noise,
        "repeat_view_depth_difference_m": stats(near_depth),
        "bbox_centre_variation_equivalent_m": stats(box_variation),
        "robot_translation_error_m": stats(pose_xy), "robot_yaw_error_deg": stats(pose_yaw),
        "planar_transform_point_error_bound_m": stats(transform_bound),
        "opposite_view_position_difference_m": stats(opposite_spread),
        "same_object_position_difference_m": stats(all_spread),
        "same_object_running_mean_residual_m": stats(running_residual),
    }


def probe_depth(args):
    import zenoh
    from object_detector import Image, decode_depth, median_box_depth, timestamp_ns
    rows, seen, done, lock = [], set(), Event(), Lock()
    counts = {"received": 0, "rejected": 0, "duplicates": 0}
    last_error = ""

    def callback(sample):
        nonlocal last_error
        with lock:
            counts["received"] += 1
        try:
            message = Image.deserialize(bytes(sample.payload))
            stamp = timestamp_ns(message.header)
            depth, valid, _ = median_box_depth(decode_depth(message), args.bbox)
            with lock:
                if stamp in seen:
                    counts["duplicates"] += 1
                    return
                if stamp not in seen and len(rows) < args.samples:
                    seen.add(stamp)
                    rows.append((stamp, depth, valid, (args.bbox[0]+args.bbox[2])/2, (args.bbox[1]+args.bbox[3])/2))
                    if len(rows) == 1 or len(rows) % 15 == 0 or len(rows) == args.samples:
                        print(f"Depth samples: {len(rows)}/{args.samples}; median={depth:.4f} m; valid pixels={valid}", flush=True)
                    if len(rows) == args.samples:
                        done.set()
        except Exception as error:
            with lock:
                counts["rejected"] += 1
                last_error = f"{type(error).__name__}: {error}"
                if counts["rejected"] <= 3:
                    print(f"Depth sample rejected: {last_error}", flush=True)

    config = zenoh.Config()
    if args.connect:
        config.insert_json5("connect/endpoints", json.dumps([args.connect]))
    session = zenoh.open(config)
    subscriber = session.declare_subscriber("camera/depth/image_rect_raw", callback)
    print(f"Depth source: camera/depth/image_rect_raw; connection={args.connect or 'automatic discovery'}", flush=True)
    print("Keep the robot stationary; probing a fixed central box ROI, without detector inference.", flush=True)
    try:
        deadline = time.monotonic()+45
        next_status = time.monotonic()+5
        while not done.wait(1):
            now = time.monotonic()
            with lock:
                status = (f"received={counts['received']}, accepted={len(rows)}/{args.samples}, "
                          f"rejected={counts['rejected']}, repeated timestamps={counts['duplicates']}")
                error = last_error
            if now >= next_status:
                print(f"Depth probe: {status}", flush=True)
                next_status = now+5
            if now >= deadline:
                if counts["received"] == 0:
                    reason = "No depth messages arrived; check the simulation, bridge, and Zenoh connection."
                elif error:
                    reason = f"Last rejection: {error}"
                else:
                    reason = "Too few distinct valid ROS-stamped depth images; check whether simulation time is advancing."
                raise TimeoutError(f"Depth probe timed out after 45 seconds: {status}. {reason}")
    finally:
        subscriber.undeclare()
        session.close()
    with Path(args.out).open("x", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("timestamp_ns", "depth_m", "valid_depth_samples", "u", "v"))
        writer.writerows(rows)
    print(f"Saved {len(rows)} fixed-ROI depth measurements to {args.out}")


def capture_views(args):
    """Save exact RGB keyframes for Task 4 labels; the detector is read-only."""
    import zenoh
    from PIL import Image as PILImage
    from object_detector import Image, timestamp_ns

    frames, pending, saved, lock = deque(maxlen=300), deque(), set(), Lock()
    counts = {"rgb": 0, "invalid_rgb": 0, "saved": 0, "missed": 0}

    def on_rgb(sample):
        try:
            message = Image.deserialize(bytes(sample.payload))
            if message.encoding.lower() not in ("rgb8", "bgr8"):
                raise ValueError(f"Unsupported color encoding: {message.encoding}")
            if message.width <= 0 or message.height <= 0 or message.step < message.width*3:
                raise ValueError("Invalid color dimensions or row stride")
            data = np.frombuffer(bytes(message.data), dtype=np.uint8)
            if data.size < message.height*message.step:
                raise ValueError("Truncated color image")
            frame = data[:message.height*message.step].reshape(message.height, message.step)
            frame = frame[:, :message.width*3].reshape(message.height, message.width, 3)
            if message.encoding.lower() == "bgr8":
                frame = frame[:, :, ::-1]
            with lock:
                frames.append((timestamp_ns(message.header), frame.copy()))
                counts["rgb"] += 1
        except Exception as error:
            with lock:
                counts["invalid_rgb"] += 1
                if counts["invalid_rgb"] <= 3:
                    print(f"RGB sample rejected: {error}", flush=True)

    def on_keyframe(sample):
        try:
            message = json.loads(bytes(sample.payload))
            run_id, keyframe_id, stamp = message["run_id"], message["keyframe_id"], message["timestamp_ns"]
            if (not isinstance(run_id, str) or not run_id
                    or not all(c.isalnum() or c in "._-" for c in run_id)
                    or type(keyframe_id) is not int or keyframe_id < 1
                    or type(stamp) is not int or stamp < 0
                    or not isinstance(message.get("detections"), list)):
                raise ValueError("Invalid run, keyframe ID, or ROS timestamp")
            with lock:
                pending.append((message, time.monotonic()))
        except Exception as error:
            print(f"Keyframe rejected: {error}", flush=True)

    config = zenoh.Config()
    if args.connect:
        config.insert_json5("connect/endpoints", json.dumps([args.connect]))
    with (args.out / "frame-index.jsonl").open("x", encoding="utf-8") as index:
        session = zenoh.open(config)
        subscribers = []
        try:
            subscribers.append(session.declare_subscriber("camera/color/image_raw", on_rgb))
            subscribers.append(session.declare_subscriber("tb/detections", on_keyframe))
            print(f"Task 4 camera capture ready: {args.out}", flush=True)
            print("Start the ingest recorder before moving. Images require an exact RGB timestamp match.", flush=True)
            next_status = time.monotonic()+5
            while True:
                with lock:
                    batch = list(pending)
                    pending.clear()
                    available = dict(frames)
                for message, arrived in batch:
                    key = (message["run_id"], message["keyframe_id"])
                    if key in saved:
                        continue
                    frame = available.get(message["timestamp_ns"])
                    if frame is None:
                        # Arrival grace only: never substitute a different RGB timestamp.
                        if time.monotonic()-arrived < 5:
                            with lock:
                                pending.append((message, arrived))
                        else:
                            counts["missed"] += 1
                            print(f"KF#{key[1]} image unavailable; start capture before the route.", flush=True)
                        continue
                    name = f"{key[0]}_kf{key[1]}.png"
                    path = args.out / name
                    if path.exists():
                        raise ValueError(f"Image already exists: {path}; use a fresh study directory")
                    PILImage.fromarray(frame).save(path)
                    index.write(json.dumps({"run_id": key[0], "keyframe_id": key[1],
                                            "timestamp_ns": message["timestamp_ns"], "image": name,
                                            "detections": len(message["detections"])})+"\n")
                    index.flush()
                    saved.add(key)
                    counts["saved"] += 1
                    print(f"KF#{key[1]} saved: {name}; detections={len(message['detections'])}", flush=True)
                if time.monotonic() >= next_status:
                    with lock:
                        print(f"Camera capture: RGB={counts['rgb']}, saved={counts['saved']}, "
                              f"missing={counts['missed']}, invalid RGB={counts['invalid_rgb']}", flush=True)
                    next_status = time.monotonic()+5
                time.sleep(.1)
        except KeyboardInterrupt:
            print(f"Capture stopped: saved={counts['saved']}, missing={counts['missed']}", flush=True)
        finally:
            for subscriber in subscribers:
                subscriber.undeclare()
            session.close()


def measurement_main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    labels_parser = sub.add_parser("labels", help="Create an annotation template; do not overwrite existing labels")
    labels_parser.add_argument("recording", type=Path)
    labels_parser.add_argument("--out", type=Path, required=True)
    depth_parser = sub.add_parser("depth-noise", help="Measure temporal median-depth noise while the robot is stopped")
    depth_parser.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    depth_parser.add_argument("--samples", type=int, default=60)
    depth_parser.add_argument("--connect", default="",
                              help="Zenoh endpoint; empty uses automatic discovery, as in the reference detector")
    depth_parser.add_argument("--out", type=Path, required=True)
    capture_parser = sub.add_parser("capture-views", help="Save matching keyframe PNGs without editing the detector")
    capture_parser.add_argument("--connect", default="", help="Zenoh endpoint; empty uses automatic discovery")
    capture_parser.add_argument("--out", type=Path, required=True, help="Fresh directory for PNGs and frame-index.jsonl")
    evaluation = sub.add_parser("evaluate", help="Replay labeled detections at multiple radii with CLIP and nearest tie-breaks")
    evaluation.add_argument("recording", type=Path)
    evaluation.add_argument("--labels", type=Path, required=True)
    evaluation.add_argument("--depth-csv", type=Path)
    evaluation.add_argument("--radii", type=float, nargs="+", default=[.15, .25, .35, .5, .75])
    evaluation.add_argument("--class-name", action="append", help="Optional study subset, e.g. chair; other classes do not interact")
    evaluation.add_argument("--out-json", type=Path, required=True)
    evaluation.add_argument("--out-csv", type=Path, required=True)
    replay = sub.add_parser("replay", help="Store a recording in an EMPTY database using the actual ingest implementation")
    replay.add_argument("recording", type=Path)
    replay.add_argument("--radius", type=float, required=True)
    replay.add_argument("--pg-db", required=True)
    replay.add_argument("--pg-host", default="localhost")
    replay.add_argument("--pg-port", type=int, default=5436)
    replay.add_argument("--pg-user", default="postgres")
    replay.add_argument("--pg-password", default="postgres")
    args = parser.parse_args()
    if hasattr(args, "out"):
        args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "capture-views":
        args.out.mkdir(parents=True, exist_ok=True)
        capture_views(args)
        return
    if args.mode == "depth-noise":
        if args.samples < 2:
            parser.error("--samples must be at least two")
        probe_depth(args)
        return
    records = load_records(args.recording)
    if args.mode == "labels":
        make_labels(records, args.out)
    elif args.mode == "evaluate":
        if not args.radii:
            parser.error("Supply at least one trial radius")
        labels = read_labels(args.labels, records)
        if args.class_name:
            records = [record for record in records if record["class_name"] in args.class_name]
        if not any(labels.get(record["det_id"], {}).get("object_label") for record in records):
            parser.error("Physical object labels are needed; blank labels cannot establish duplicates or false merges")
        measurements = error_measurements(records, labels, args.depth_csv)
        trials = []
        for radius in sorted(set(args.radii)):
            for rule in ("clip", "nearest"):
                assignments, candidates = replay_associations(records, radius, rule)
                trials.append({"radius_m": radius, "tie_break": rule,
                               **association_metrics(assignments, labels),
                               **{f"candidate_{key}": value for key, value in candidates.items()}})
        required = ("stationary_depth_noise_m", "bbox_centre_variation_equivalent_m",
                    "planar_transform_point_error_bound_m", "opposite_view_position_difference_m")
        missing = [name for name in required if not measurements[name]["count"]]
        report = {"recording": str(args.recording), "detections_replayed": len(records),
                  "classes": args.class_name or "all", "measurements": measurements,
                  "missing_measurements": missing, "radius_status": "provisional; select and justify from measured errors and trial table",
                  "trials": trials}
        for path in (args.out_json, args.out_csv):
            path.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        with args.out_csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(trials[0]))
            writer.writeheader()
            writer.writerows(trials)
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        import psycopg2
        connection = psycopg2.connect(host=args.pg_host, port=args.pg_port, dbname=args.pg_db,
                                      user=args.pg_user, password=args.pg_password, connect_timeout=5)
        try:
            initialize_database(connection)
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT count(*) FROM semantic_objects")
                    if cursor.fetchone()[0]:
                        raise ValueError("Replay requires empty assignment tables; create a fresh database")
            for record in records:
                store_detection(connection, record["message"], record["detection"], args.radius)
            print(f"Replayed {len(records)} observations at radius {args.radius:g} m into {args.pg_db}")
        finally:
            connection.close()


QUERIES = {
    "nearest": """
        SELECT object_id, class_name, map_x, map_y, map_z,
               sqrt(power(map_x - %s, 2) + power(map_y - %s, 2)) AS distance
        FROM semantic_objects
        WHERE class_name = %s
        ORDER BY distance, object_id
        LIMIT 1
    """,
    "region": """
        SELECT object_id, class_name, map_x, map_y, map_z, observation_count
        FROM semantic_objects
        WHERE map_x BETWEEN %s AND %s AND map_y BETWEEN %s AND %s
        ORDER BY class_name, object_id
    """,
    "histogram": """
        SELECT class_name, count(*) AS object_count
        FROM semantic_objects
        GROUP BY class_name
        ORDER BY object_count DESC, class_name
    """,
    "poses": """
        SELECT observation_id, seen_at_ns, map_x, map_y, map_yaw,
               camera_x, camera_y, camera_z, bearing_rad, range_m, confidence
        FROM object_observations
        WHERE object_id = %s
        ORDER BY seen_at_ns, observation_id
    """,
}


def fetch_rows(cursor):
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def run_query(connection, name, parameters=()):
    if name not in QUERIES:
        raise ValueError("Unknown query")
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(QUERIES[name], parameters)
            return fetch_rows(cursor)


def export_database(connection):
    """Preserve every required column, with embeddings as numeric JSON arrays."""
    # A live ingest worker may still be writing. Read both tables from one
    # snapshot so that the object count agrees with its exported evidence rows.
    with connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            cursor.execute(
                """SELECT object_id, class_name, map_x, map_y, map_z,
                          observation_count, mean_embedding::text,
                          first_seen_ns, last_seen_ns
                   FROM semantic_objects ORDER BY object_id"""
            )
            objects = fetch_rows(cursor)
            for obj in objects:
                obj["mean_embedding"] = json.loads(obj["mean_embedding"])
            cursor.execute(
                """SELECT observation_id, object_id, seen_at_ns, map_x, map_y, map_yaw,
                          camera_x, camera_y, camera_z,
                          camera_qx, camera_qy, camera_qz, camera_qw,
                          bearing_rad, range_m, bbox_xmin, bbox_ymin,
                          bbox_xmax, bbox_ymax, confidence, embedding::text
                   FROM object_observations ORDER BY observation_id"""
            )
            observations = fetch_rows(cursor)
            for observation in observations:
                observation["embedding"] = json.loads(observation["embedding"])
    return {
        "frame_id": "map", "timestamp_domain": "ROS",
        "range_convention": "3D distance from camera centre, metres",
        "bearing_convention": "0 camera forward; positive left, radians",
        "semantic_objects": objects, "object_observations": observations,
    }


def query_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=5436)
    parser.add_argument("--pg-db", default="semantic_task4_calibration")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--pg-password", default="postgres")
    sub = parser.add_subparsers(dest="query", required=True)
    nearest = sub.add_parser("nearest", help="Nearest class instance by horizontal map distance")
    nearest.add_argument("class_name")
    nearest.add_argument("x", type=float)
    nearest.add_argument("y", type=float)
    region = sub.add_parser("region", help="Inclusive map rectangle")
    for name in ("xmin", "xmax", "ymin", "ymax"):
        region.add_argument(name, type=float)
    sub.add_parser("histogram", help="Count semantic object records, grouped by class")
    poses = sub.add_parser("poses", help="Every stored observation pose for one object")
    poses.add_argument("object_id", type=int)
    export = sub.add_parser("export", help="Export both tables, including full vectors and camera poses")
    export.add_argument("--output", type=Path, default=Path("/tmp/semantic-map-export.json"))
    args = parser.parse_args(sys.argv[2:])

    parameters = ()
    if args.query == "nearest":
        if not args.class_name.strip() or not all(math.isfinite(v) for v in (args.x, args.y)):
            parser.error("Supply a class name and finite map coordinates")
        parameters = (args.x, args.y, args.class_name)
    elif args.query == "region":
        parameters = (args.xmin, args.xmax, args.ymin, args.ymax)
        if (not all(math.isfinite(v) for v in parameters)
                or args.xmin > args.xmax or args.ymin > args.ymax):
            parser.error("Rectangle bounds must be finite and ordered")
    elif args.query == "poses":
        if args.object_id <= 0:
            parser.error("object_id must be positive")
        parameters = (args.object_id,)

    connection = psycopg2.connect(
        host=args.pg_host, port=args.pg_port, dbname=args.pg_db,
        user=args.pg_user, password=args.pg_password, connect_timeout=5,
    )
    try:
        if args.query == "export":
            result = export_database(connection)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"Exported {len(result['semantic_objects'])} objects and "
                  f"{len(result['object_observations'])} observations to {args.output}")
        else:
            print(json.dumps(run_query(connection, args.query, parameters), indent=2, allow_nan=False))
    finally:
        connection.close()


def map_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--keyframes", type=Path, help="Recorded Task 2 envelopes, including empty keyframes")
    parser.add_argument("--output", type=Path, default=Path("/tmp/semantic-map.png"))
    parser.add_argument("--title", default="Semantic map")
    args = parser.parse_args(sys.argv[2:])
    data = json.loads(args.export.read_text(encoding="utf-8"))
    if data.get("frame_id") != "map":
        parser.error("Export must use the map frame")
    objects, observations = data["semantic_objects"], data["object_observations"]

    # The required observation table contains robot poses but no run/keyframe
    # columns. Raw JSONL supplies empty frames and the complete accepted route.
    routes, route = [], []
    if args.keyframes:
        previous_run = previous_stamp = None
        with args.keyframes.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                frame = json.loads(line)
                if frame.get("frame_id") != "map" or frame.get("timestamp_domain") != "ROS":
                    parser.error("Keyframes must use map coordinates and ROS time")
                run = (frame.get("run_id"), frame.get("clock_epoch", 0))
                stamp = frame["timestamp_ns"]
                if route and (run != previous_run or stamp < previous_stamp):
                    routes.append(route)
                    route = []
                pose = (float(frame["map_x"]), float(frame["map_y"]))
                if not all(math.isfinite(value) for value in pose):
                    parser.error("Robot map coordinates must be finite")
                route.append(pose)
                previous_run, previous_stamp = run, stamp
        if route:
            routes.append(route)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    for index, route in enumerate(routes):
        ax.plot([p[0] for p in route], [p[1] for p in route], ".-", linewidth=1,
                color="0.45", markersize=4, label="Accepted robot poses" if index == 0 else None)
    if not args.keyframes and observations:
        poses = sorted({(o["map_x"], o["map_y"]) for o in observations})
        ax.scatter([p[0] for p in poses], [p[1] for p in poses], s=16,
                   color="0.45", label="Robot poses with stored detections")

    labels = []
    classes = sorted({obj["class_name"] for obj in objects})
    colours = plt.get_cmap("tab20")
    for index, class_name in enumerate(classes):
        subset = [obj for obj in objects if obj["class_name"] == class_name]
        ax.scatter([o["map_x"] for o in subset], [o["map_y"] for o in subset], s=55,
                   color=colours(index % 20), edgecolors="black", linewidths=0.4, label=class_name)
        for obj in subset:
            labels.append((f"{class_name} #{obj['object_id']}", (obj["map_x"], obj["map_y"])))
    ax.set(xlabel="Map x (m)", ylabel="Map y (m)",
           title=f"{args.title}: {len(objects)} objects, {len(observations)} observations")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.15, y=0.1)
    if objects or observations or routes:
        ax.legend(fontsize=8)
    if not args.keyframes:
        fig.text(0.02, 0.01, "Coverage contains only stored observations; supply --keyframes to include empty frames.", fontsize=8)
    # Nearly identical locations can be predictions of different classes.
    # Move overlapping text upward, keeping the measured points unchanged.
    annotations = []
    for label, point in labels:
        annotation = ax.annotate(label, point, xytext=(4, 5), textcoords="offset points", fontsize=8)
        for attempt in range(30):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            box = annotation.get_window_extent(renderer).expanded(1.03, 1.12)
            if not any(box.overlaps(old.get_window_extent(renderer)) for old in annotations):
                break
            annotation.set_position((4, 5 + (attempt + 1) * 14))
        if annotation.get_position() != (4, 5):
            ax.annotate("", point, xytext=annotation.get_position(), textcoords="offset points",
                        arrowprops={"arrowstyle": "-", "color": "0.5", "lw": .6})
        annotations.append(annotation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
