#!/usr/bin/env python3
"""Task 4 measurements, annotated radius trials, and final JSONL replay.

This is an offline evaluation aid, not a replacement detector or database schema.
It never infers physical identity from proximity: object labels are supplied by a human.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from threading import Event, Lock
import time

import numpy as np

from embedding_ingest import decode_embedding, prepare_detection


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


def main():
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
        from embedding_ingest import initialize_database, store_detection
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


if __name__ == "__main__":
    main()
