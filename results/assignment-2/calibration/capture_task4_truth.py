#!/usr/bin/env python3
"""Read-only, simultaneous Gazebo/ROS pose capture. Run from the WSL repo root."""

import argparse
import base64
import concurrent.futures
import datetime
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


def brace_block(text, start):
    opening = text.index("{", start)
    depth = 0
    quoted = escaped = False
    for end in range(opening, len(text)):
        char = text[end]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1:end]
    raise ValueError("Truncated message block")


def number(text, name, default=0.0):
    match = re.search(r"(?m)^\s*" + re.escape(name) + r"\s*:\s*([-+0-9.eE]+)", text)
    return float(match.group(1)) if match else default


def yaw(quaternion):
    x, y, z, w = quaternion
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid quaternion")
    x, y, z, w = (value / norm for value in quaternion)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def gazebo_poses(text):
    """Read complete Pose_V headers and the named root model, never child poses."""
    starts = [match.start() for match in re.finditer(r"(?m)^header\s*\{", text)]
    result = []
    for index, start in enumerate(starts):
        message = text[start:starts[index + 1] if index + 1 < len(starts) else len(text)]
        try:
            header = brace_block(message, 0)
            match = re.search(r"\bstamp\s*\{", header)
            if not match:
                continue
            stamp = brace_block(header, match.start())
            ns = int(number(stamp, "sec")) * 1_000_000_000 + int(number(stamp, "nsec"))
            for match in re.finditer(r"(?m)^pose\s*\{", message):
                pose = brace_block(message, match.start())
                if not re.search(r'(?m)^\s*name\s*:\s*"turtlebot"\s*$', pose):
                    continue
                position_match = re.search(r"\bposition\s*\{", pose)
                orientation_match = re.search(r"\borientation\s*\{", pose)
                position = brace_block(pose, position_match.start()) if position_match else ""
                orientation = brace_block(pose, orientation_match.start()) if orientation_match else ""
                quaternion = [number(orientation, key) for key in ("x", "y", "z", "w")]
                result.append({"timestamp_ns": ns, "frame": "Gazebo world", "model": "turtlebot",
                               "x": number(position, "x"), "y": number(position, "y"),
                               "z": number(position, "z"), "quaternion_xyzw": quaternion,
                               "yaw_rad": yaw(quaternion)})
                break
        except (ValueError, AttributeError):
            continue
    return result


def tf_poses(text):
    # Jazzy prints integer sec.nanosec fields, without zero-padding nanosec.
    result = []
    blocks = re.split(r"At time\s+(\d+)\.(\d+)\s*", text)
    for index in range(1, len(blocks) - 2, 3):
        sec, fraction, body = blocks[index:index + 3]
        translation = re.search(r"Translation:\s*\[([^]]+)\]", body)
        rotation = re.search(r"in Quaternion(?:\s*\(xyzw\))?\s*\[([^]]+)\]", body)
        if not translation or not rotation:
            continue
        position = [float(value) for value in translation.group(1).split(",")]
        quaternion = [float(value) for value in rotation.group(1).split(",")]
        if len(position) != 3 or len(quaternion) != 4:
            continue
        rpy = re.search(r"in RPY\s*\(radian\)\s*\[([^]]+)\]", body)
        yaw_value = float(rpy.group(1).split(",")[2]) if rpy else yaw(quaternion)
        if int(fraction) >= 1_000_000_000:
            raise ValueError("Invalid tf2_echo nanosecond field")
        result.append({"timestamp_ns": int(sec) * 1_000_000_000 + int(fraction),
                       "frame": "map", "child_frame": "base_link", "x": position[0],
                       "y": position[1], "z": position[2], "quaternion_xyzw": quaternion,
                       "yaw_rad": yaw_value})
    return result


def timestamp_matches(truth, estimates):
    if not truth:
        return []
    unique = {row["timestamp_ns"]: row for row in estimates}
    matches = []
    for estimate in sorted(unique.values(), key=lambda row: row["timestamp_ns"]):
        nearest = min(truth, key=lambda row: abs(row["timestamp_ns"] - estimate["timestamp_ns"]))
        difference = nearest["timestamp_ns"] - estimate["timestamp_ns"]
        if abs(difference) <= 200_000_000:
            matches.append({"estimate": estimate, "gazebo_root_model": nearest,
                            "truth_minus_estimate_ns": difference})
    return matches


def capture_file(directory, name, command, duration):
    with (directory / name).open("wb") as output:
        try:
            process = subprocess.run(["docker", "compose", "exec", "-T", "demo-world-house",
                                      "bash", "-lc", "source /opt/ros/jazzy/setup.bash && " + command],
                                     stdout=output, stderr=subprocess.STDOUT, timeout=duration + 12)
            return {"file": name, "exit_code": process.returncode}
        except subprocess.TimeoutExpired:
            return {"file": name, "exit_code": "host timeout"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=20)
    args = parser.parse_args()
    if not 10 <= args.seconds <= 45:
        parser.error("Choose 10–45 seconds")
    root = Path.cwd()
    if not (root / "tb_worlds").is_dir():
        parser.error("Run from ~/robotics/turtlebot-maze, where tb_worlds exists")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = root / "results" / ("task4-transform-check-" + stamp + ".json")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        parser.error("A capture with this timestamp already exists; run again")
    directory = Path(tempfile.mkdtemp(prefix="task4-truth-"))
    containers = subprocess.run(["docker", "ps", "--format",
        '{{.Names}}\t{{.Label "com.docker.compose.service"}}\t{{.Status}}'],
        capture_output=True, text=True, check=True).stdout
    (directory / "running-containers.txt").write_text(containers)
    resolved = subprocess.run(["docker", "compose", "config", "--format", "json"],
                              capture_output=True, text=True, check=True)
    house = json.loads(resolved.stdout)["services"].get("demo-world-house", {})
    # Only world launch settings are needed; omit credentials and unrelated services.
    launch = {key: house[key] for key in ("image", "command", "entrypoint", "working_dir", "volumes")
              if key in house}
    environment = house.get("environment", {})
    if isinstance(environment, dict):
        launch["environment"] = {key: value for key, value in environment.items()
                                 if key in ("ROS_DOMAIN_ID", "WORLD", "WORLD_NAME", "MAP_FILE",
                                            "ROBOT_MODEL", "TURTLEBOT3_MODEL")}
    (directory / "world-launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print("Keep the robot stationary. Capturing Gazebo truth and map/base_link TF together.", flush=True)
    world_rows = [line for line in containers.splitlines()
                  if "demo-world" in line or "\tsim\t" in line]
    if len(world_rows) != 1:
        print("World check: expected one active world; found", len(world_rows), flush=True)
        print("The archive will include this setup issue; no containers are stopped.", flush=True)
    for relative in ("tb_worlds/maps/house_world_map.yaml", "tb_worlds/maps/house_world_map.pgm",
                     "tb_worlds/worlds/house_world.sdf.xacro"):
        path = root / relative
        if path.is_file():
            destination = directory / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    # Preserve map/world generation and launch sources so alignment can be checked independently.
    package = root / "tb_worlds"
    if package.is_dir():
        for path in package.rglob("*"):
            if not path.is_file() or not (path.suffix == ".py" or
                ("house" in path.name.lower() and path.suffix in (".xacro", ".sdf", ".yaml"))):
                continue
            destination = directory / "source" / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    commands = [
        ("gazebo-pose.txt", f"timeout -s INT {args.seconds}s gz topic -e -t /world/default/dynamic_pose/info", args.seconds),
        ("map-base-link-tf.txt", f"timeout -s INT {args.seconds}s ros2 run tf2_ros tf2_echo map base_link", args.seconds),
        ("robot-description.txt", "timeout -s INT 8s ros2 param get /robot_state_publisher robot_description", 8),
        ("map-server-file.txt", "timeout -s INT 8s ros2 param get /map_server yaml_filename", 8),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(capture_file, directory, *command) for command in commands]
        while not all(future.done() for future in futures):
            print("Pose capture is running…", flush=True)
            time.sleep(3)
        exits = [future.result() for future in futures]
    active_map = (directory / "map-server-file.txt").read_text(errors="replace")
    match = re.search(r"(?m)^String value is:\s*(.+\.ya?ml)\s*$", active_map)
    if match:
        active_path = match.group(1).strip()
        with (directory / "active-map.yaml").open("wb") as output:
            subprocess.run(["docker", "compose", "exec", "-T", "demo-world-house", "cat", active_path],
                           stdout=output, stderr=subprocess.STDOUT, timeout=8)
    truth = gazebo_poses((directory / "gazebo-pose.txt").read_text(errors="replace"))
    estimates = tf_poses((directory / "map-base-link-tf.txt").read_text(errors="replace"))
    matches = timestamp_matches(truth, estimates)
    summary = {"gazebo_root_poses": len(truth), "tf_pose_samples": len(estimates),
               "matched_ros_timestamps": len(matches), "max_sync_ns": 200_000_000,
               "world_to_map_alignment": "pending source verification; no identity transform assumed",
               "root_model_to_base_link": "pending robot-description verification",
               "commands": exits, "timestamp_matches": matches}
    captured_files = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if path.suffix.lower() in (".pgm", ".png"):
            captured_files[str(path.relative_to(directory))] = {
                "encoding": "base64", "data": base64.b64encode(raw).decode("ascii")}
        else:
            captured_files[str(path.relative_to(directory))] = {
                "encoding": "utf-8", "data": raw.decode("utf-8", errors="replace")}
    result = {"format": "task4-independent-pose-capture-v1", "pose_summary": summary,
              "captured_files": captured_files}
    with archive.open("x", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False, separators=(",", ":"))
        output.write("\n")
    shutil.rmtree(directory)
    downloads = Path("/mnt/c/Users/mp528/Downloads")
    if downloads.is_dir():
        destination = downloads / archive.name
        if not destination.exists():
            shutil.copy2(archive, destination)
            print("Windows Downloads:", destination.name, flush=True)
    print("Gazebo poses:", len(truth), "| TF poses:", len(estimates), "| synchronized pairs:", len(matches))
    print("Upload:", archive)
    if not matches:
        print("No synchronized pairs were available; the archive preserves the diagnostic output.")


if __name__ == "__main__":
    main()
