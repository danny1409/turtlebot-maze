#!/usr/bin/env python3
"""Plot real exported object estimates and the recorded accepted robot poses."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--keyframes", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/semantic-map.png"))
    args = parser.parse_args()
    data = json.loads(args.export.read_text())
    if data.get("frame_id") != "map":
        parser.error("export must be in the map frame")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    objects, observations = data["semantic_objects"], data["object_observations"]
    poses = {}
    if args.keyframes:
        with args.keyframes.open() as frames:
            for line in frames:
                if not line.strip():
                    continue
                frame = json.loads(line)
                key = (frame["run_id"], frame["clock_epoch"], frame["keyframe_id"])
                poses[key] = (frame["map_x"], frame["map_y"])
    else:
        for observation in observations:
            key = (observation["run_id"], observation["clock_epoch"], observation["keyframe_id"])
            poses[key] = (observation["map_x"], observation["map_y"])
    routes = defaultdict(list)
    for (run, epoch, keyframe), pose in sorted(poses.items()):
        routes[(run, epoch)].append(pose)
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    for index, ((run, epoch), route) in enumerate(routes.items()):
        ax.plot([p[0] for p in route], [p[1] for p in route], ".-", linewidth=1,
                color="0.45", markersize=3, label="Accepted robot poses" if index == 0 else None)
    if observations:
        ax.scatter([o["object_x"] for o in observations], [o["object_y"] for o in observations],
                   s=8, color="0.7", alpha=0.4, label="Raw object measurements")
    classes = sorted({o["class_name"] for o in objects})
    colours = plt.get_cmap("tab20")
    for index, class_name in enumerate(classes):
        subset = [o for o in objects if o["class_name"] == class_name]
        ax.scatter([o["map_x"] for o in subset], [o["map_y"] for o in subset],
                   s=55, color=colours(index % 20), edgecolors="black", linewidths=0.4, label=class_name)
        for o in subset:
            ax.annotate(f"{class_name} #{o['object_id']}", (o["map_x"], o["map_y"]),
                        xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Map x (m)", ylabel="Map y (m)",
           title=f"Semantic map: {len(objects)} objects, {len(observations)} observations")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.15, y=0.1)
    if objects or observations or poses:
        ax.legend(fontsize=8)
    if not args.keyframes:
        fig.text(0.02, 0.01, "Coverage contains only keyframes with stored detections; supply --keyframes for empty frames.",
                 fontsize=8)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
