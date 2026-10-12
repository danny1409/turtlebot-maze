#!/usr/bin/env python3
"""Reproduce the image-labeled Task 4 study without ROS, GPU inference, or DB writes.

Run inside the detector container or with Python and numpy. The existing worker
and measurement helper supply the already-tested pure validation/replay functions.
Only their named function definitions are loaded, not their main/import sections.
"""

import argparse
import ast
import base64
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np


def study_functions(worker, helper):
    namespace = dict(base64=base64, csv=csv, itertools=itertools, json=json,
                     math=math, np=np, Path=Path)
    for path, names in ((worker, {"decode_embedding", "prepare_detection"}),
                        (helper, {"load_records", "read_labels", "replay_associations",
                                  "association_metrics", "stats", "error_measurements"})):
        tree = ast.parse(path.read_text())
        selected = [node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in selected} != names:
            raise ValueError(f"Required pure study functions missing from {path}")
        module = ast.Module(body=selected, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def rotate(quaternion, vector):
    qx, qy, qz, qw = quaternion
    length = math.sqrt(sum(value * value for value in quaternion))
    qx, qy, qz, qw = (value / length for value in quaternion)
    rotation = np.asarray([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)]])
    return rotation @ vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).resolve().parent
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=directory.parents[2] / "detector/embedding_ingest.py")
    parser.add_argument("--helper", type=Path, default=directory.parents[2] / "detector/embedding_ingest.py")
    parser.add_argument("--labels", type=Path, default=directory / "chair-labels.csv")
    parser.add_argument("--depth-csv", type=Path, default=directory / "chair-depth-noise.csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fn = study_functions(args.worker, args.helper)
    all_records = fn["load_records"](args.recording)
    chairs = [record for record in all_records if record["class_name"] == "chair"]
    labels = fn["read_labels"](args.labels, chairs)
    primary = [record for record in chairs if labels.get(record["det_id"], {}).get("object_label")]
    if len(primary) != 8 or len(chairs) != 9:
        raise ValueError("This study expects the unchanged nine-chair-detection recording and eight reviewed labels")
    radii = (.15, .25, .35, .45, .50, .60, .65, .70, .75, .80, .85, .90,
             .95, 1., 1.01, 1.10, 1.25, 1.40, 1.50)
    trials = {"strict_eight_detections": [], "nine_detection_identity_sensitivity": []}
    hypothesis = {key: dict(value) for key, value in labels.items()}
    ambiguous = [record for record in chairs if not labels[record["det_id"]]["object_label"]]
    if len(ambiguous) != 1 or ambiguous[0]["message"]["keyframe_id"] != 104:
        raise ValueError("Expected the excluded KF104 chair box")
    hypothesis[ambiguous[0]["det_id"]]["object_label"] = "chair_entrance_fridge_side"
    for scope, records, annotation in (("strict_eight_detections", primary, labels),
                                       ("nine_detection_identity_sensitivity", chairs, hypothesis)):
        for radius, rule in itertools.product(radii, ("clip", "nearest")):
            assignments, candidates = fn["replay_associations"](records, radius, rule)
            trials[scope].append({"radius_m": radius, "tie_break": rule,
                                  "records": len(set(assignments.values())), **candidates,
                                  **fn["association_metrics"](assignments, annotation)})
    trials["annotation_note"] = {
        "primary": "Eight image-reviewed detections of three physical chairs; KF104 excluded.",
        "sensitivity": "Conditional only: KF104 is assigned the entrance/fridge-side identity. This identity is not verified."}
    measurements = fn["error_measurements"](primary, labels, args.depth_csv)
    by_frame = {record["message"]["keyframe_id"]: record for record in primary}
    repeat_pairs = []
    for first, second in ((3, 4), (46, 47)):
        a, b = by_frame[first], by_frame[second]
        shared_range = (a["range"] + b["range"]) / 2
        projected = []
        for record in (a, b):
            u, v = np.asarray(record["detection"]["bbox"]).reshape(2, 2).mean(axis=0)
            ray = np.asarray([(u - 160) / 277.13, (v - 120) / 277.13, 1.])
            ray /= np.linalg.norm(ray)
            projected.append(np.asarray(record["camera"][:3]) + shared_range *
                             rotate(record["camera"][3:], ray))
        repeat_pairs.append({
            "keyframes": [first, second],
            "camera_displacement_m": float(np.linalg.norm(np.asarray(a["camera"][:3]) - b["camera"][:3])),
            "shared_range_m": shared_range,
            "box_centre_spread_after_rotation_compensation_m": float(np.linalg.norm(projected[0] - projected[1])),
            "method": "camera position + shared camera/object range * map-frame unit ray through box centre",
            "limitation": "near-position repeat with camera rotation; residual localization/viewpoint change is included; this is not isolated stationary detector jitter"})
    measurements["rotation_compensated_bbox_repeats"] = repeat_pairs
    measurements["rotation_compensated_bbox_repeat_spread_m"] = fn["stats"]([
        pair["box_centre_spread_after_rotation_compensation_m"] for pair in repeat_pairs])
    a, b = by_frame[5], by_frame[24]
    anchor = np.median([a["position"], b["position"]], axis=0)
    rays = [np.asarray(record["camera"][:2]) - anchor[:2] for record in (a, b)]
    measurements["opposite_view_pair"] = {
        "keyframes": [5, 24], "angle_deg": math.degrees(math.acos(float(np.clip(
            np.dot(*rays) / (np.linalg.norm(rays[0]) * np.linalg.norm(rays[1])), -1, 1))))}
    discrepancies = []
    for record in all_records:
        message, detection = record["message"], record["detection"]
        u, v = np.asarray(detection["bbox"]).reshape(2, 2).mean(axis=0)
        depth = detection["depth_m"]
        xo, yo = (u - 160) * depth / 277.13, (v - 120) * depth / 277.13
        bx, by, bz = depth + .064, -xo - .065, -yo + .094
        c, s = math.cos(message["map_yaw"]), math.sin(message["map_yaw"])
        calculated = [message["map_x"] + c*bx - s*by,
                      message["map_y"] + s*bx + c*by, message["map_z"] + bz]
        discrepancies.append(float(np.linalg.norm(np.asarray(calculated) - record["position"])))
    measurements["numeric_transform_consistency_m"] = {
        **fn["stats"](discrepancies), "description": "scalar chain vs published coordinates; not localization accuracy"}
    registration = json.loads((directory / "static-registration-and-error.json").read_text())
    measurements["independent_localization_error"] = {
        "status": "measured in a separate stationary native Gazebo/TF capture",
        "source": "static-registration-and-error.json",
        "matched_samples": len(registration["matched_robot_samples"]),
        "xy_error_m": registration["robot_xy_error_m"],
        "yaw_error_deg": registration["robot_yaw_error_deg"],
        "max_sync_ms": max(row["sync_ms"] for row in registration["matched_robot_samples"]),
        "scope": "Independent checkpoint, not simultaneous truth for the chair or final whole-house route."}
    measurements["candidate_radius_m"] = .5
    measurements["radius_status"] = "selected 0.50 m from the reviewed subset and separate measured localization checkpoint; final full-route trials are reported separately"
    measurements["annotation_scope"] = {
        "reviewed_frames": 140, "accepted_chair_labels": len(primary),
        "distinct_labeled_chairs": len({labels[record["det_id"]]["object_label"] for record in primary}),
        "excluded_chair_keyframe": 104,
        "scope": "image-reviewed relative physical table-side identities; current unique SDF model names are not asserted"}
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in (("measurements.json", measurements), ("radius-metrics.json", trials)):
        (args.out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print("Reproduced: 8 reviewed detections, 3 chairs, 19 radii, CLIP/nearest trials.")
    print("Opposite-view angle:", round(measurements["opposite_view_pair"]["angle_deg"], 3), "degrees")
    print("Written:", args.out)


if __name__ == "__main__":
    main()
