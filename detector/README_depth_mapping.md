# Object detection: synchronized depth and map geometry

Prepared on 2026-10-08 from the detector supplied in this session. This package
adds depth-to-map geometry to the working YOLO/CLIP pipeline. The database,
deduplication, queries, and final evaluation are the next assignment components.

## Install into the existing WSL repository

Save `detector-depth-mapping.zip` in your Windows Downloads folder. Keep the
house simulation, Zenoh router, and bridge running. In a second WSL terminal:

```bash
cd ~/robotics/turtlebot-maze
cp detector/object_detector.py detector/object_detector.before_depth.py
python3 -m zipfile -e /mnt/c/Users/mp528/Downloads/detector-depth-mapping.zip detector/
docker compose exec -T detector python -m unittest -v test_mapping
```

The archive contains five files directly at its root. They belong together in
`detector/`: `object_detector.py`, `mapping_core.py`, `ros_cdr.py`,
`test_mapping.py`, and this README. Extraction keeps the downloaded
`detector/yolov8n.pt` model. All 15 tests should report `OK` with no skipped tests
inside the detector container.

If the ZIP was saved elsewhere, adjust its path in the extraction command.
The first backup command is for the first installation; preserve that backup
when installing subsequent revisions.

Restart only the detector and inspect its recent logs:

```bash
docker compose restart detector
docker compose logs -f --since 1m detector
```

The current Compose command and existing Python dependencies are sufficient.
The `./detector:/app` bind mount exposes the new files without an image rebuild.
Wait for `Detector running`, then move the robot using a Nav2 goal in RViz.
New output starts with `KF#... Mapped:` and includes object map coordinates and
RGB-to-depth/odometry timestamp differences. Empty mapped lists are also logged.

`Ctrl+C` in a `docker compose logs -f` terminal closes the log viewer. Keep the
original `docker compose up demo-world-house zenoh-router zenoh-bridge` terminal
open so the simulator continues supplying data.

## If it waits for TF

The detector subscribes to `tf` and `tf_static` and requests cached static TF
when available. Static ROS transforms may have been forwarded before this
detector subscribed. If a persistent `[sync] no time-valid TF path ...` message
appears, or the ten-second `Status:` output shows no `tf_static` messages,
restart only the bridge in another WSL terminal while the detector is running:

```bash
cd ~/robotics/turtlebot-maze
docker compose restart zenoh-bridge
```

This lets the bridge receive and forward the existing transient-local static
transforms again. Confirm that `Status:` counts for `rgb`, `depth`, `odom`, and
`tf` grow. If the warning remains, save that status and the complete warning;
the actual bridge routing/cache behavior must be checked on the laptop.

The ROS-side `map -> odom` transform was verified in this session. Its observed
translation `(-0.001, -0.077, 0)` and yaw approximately `-1.388 degrees` are a
diagnostic snapshot. The code obtains timestamped transforms during execution.

## Data flow and geometry

1. PyCDR2 decodes RGB, 32FC1 depth, odometry, and TF payloads. ROS covariance
   fields are fixed arrays of 36 doubles. Image decoding respects byte order
   and row padding.
2. Callbacks fill bounded timestamp-sorted buffers. A separate worker performs
   GPU inference while incoming messages continue to be received. The worker
   waits for depth and odometry to reach the RGB timestamp, chooses their
   nearest samples, and rejects a frame if either difference exceeds 200 ms.
   The default two-second arrival wait is measured in wall time; matching uses
   ROS time exclusively. Queues are bounded, so frames can be dropped under load.
3. Dynamic TF edges are interpolated at the RGB timestamp, using linear
   translation and quaternion SLERP. Dynamic TF is not extrapolated; missing
   time-valid paths cause the frame to be skipped. Static transforms are valid
   at all timestamps.
4. The odometry child frame is honored. In particular, `base_footprint` is
   converted to `base_link` through TF instead of silently treating them as
   the same frame:

   `T_map_base = T_map_odom @ T_odom_child @ T_child_base_link`.

   The synchronized odometry pose supplies `T_odom_child`. The surrounding TF
   transforms are evaluated at the RGB stamp; the selected odometry sample
   may differ by up to the documented 200 ms tolerance.
5. Movement gating uses the synchronized robot pose in its odometry frame.
   A frame is accepted after 0.5 m of planar translation OR 15 degrees of
   wrapped yaw change. Its reference advances only after successful processing
   and publication. Inference is limited to ten keyframes per wall second.
6. YOLOv8 nano uses confidence 0.3. Each box uses the median finite, positive
   depth in the central 50% of its width and height (25% of box area). Integer
   slicing expands the continuous region by less than one pixel on each side.
   Fewer than nine valid pixels rejects the detection. This minimum is a
   documented initial choice that should be evaluated on small objects.
7. The supplied 320x240 camera intrinsics are `fx=fy=277.13`, `cx=160`, `cy=120`.
   RGB and depth are assumed registered, with matching size and those same
   intrinsics, as used by the assignment. A different camera setup requires
   calibrated intrinsics and appropriate image registration.

   ```text
   p_optical = [(u-160)*d/277.13, (v-120)*d/277.13, d]
   p_base = [z_optical, -x_optical, -y_optical] + [0.064, -0.065, 0.094]
   p_map = T_map_base @ [p_base.x, p_base.y, p_base.z, 1]
   ```

8. Accepted crops are encoded as one CLIP ViT-B-32/laion2b_s34b_b79k batch.
   Embeddings are normalized and stored as base64-encoded, little-endian
   float32 vectors of length 512. The default enables embeddings.

The fixed camera extrinsics come from the assignment. The resulting camera
pose uses the optical convention: x right, y down, z forward. `range_m` is the
three-dimensional distance from the camera centre, not just axial depth.
`bearing_rad = atan2(-x_optical, z_optical)` has zero straight ahead and positive
angles to the camera's left, consistent with base-forward/left directions.

## Published JSON, schema version 2

The key remains `tb/detections`. The envelope includes:

- `timestamp_ns` and `seen_at_ns`: RGB ROS timestamp, integer nanoseconds.
- `timestamp`: the same ROS time in floating seconds for compatibility;
  `timestamp_domain` explicitly says `ROS`. It is not a Unix wall-clock date.
- `depth_timestamp_ns`, `odom_timestamp_ns`, and signed matching deltas.
- `frame_id: map`, the original `odom_frame`, and camera/image frame metadata.
- Top-level `map_x/map_y/map_z/map_yaw`: the ROBOT pose in the map frame.
- `camera_x/y/z`, `camera_qx/qy/qz/qw`: the CAMERA optical pose in the map frame.
- Per-detection `map_x/map_y/map_z`: the OBJECT estimate in the map frame.
- Class, confidence, pixel box, depth median, valid-depth count, range, bearing,
  and the normalized embedding.

The original reference mislabeled raw odometry as map pose. Schema version 2
changes that meaning to an actual map-frame pose. Database ingestion should use
the integer nanosecond fields and distinguish robot fields at the envelope
level from object fields inside each detection. Updating the database ingest
and implementing its two-table semantic schema are the next stage.

If RGB time jumps backward by more than one second, the detector starts a new
`clock_epoch`, clears dynamic histories, and resets the motion reference. This
is a restart heuristic for the local simulator; very delayed RGB messages can
also trigger it. Static robot transforms are retained. Keyframe identifiers
continue to increase, preventing duplicate detection IDs within the run.

## Hand-computed transform test

The main test uses an off-centre pixel `(229.2825, 92.287)` and depth `2 m`.

| Stage | Expected point |
|---|---|
| Optical back-projection | `(0.5, -0.2, 2)` |
| Optical axes converted and camera offset added | `(2.064, -0.565, 0.294)` |
| Robot in odom at `(1,2,0)`, yaw `+90 degrees` | `(1.565, 4.064, 0.294)` |
| Odom in map at `(10,-3,0.5)`, yaw `-90 degrees` | `(14.064, -4.565, 0.794)` |

The numerical tolerance is `1e-9 m`, with relative tolerance zero. The test
exercises both off-centre pixel axes, camera translation, rotation, and the
map/odom transform order. A second independent calculation tests a 90-degree
roll to verify that the transform keeps nonplanar rotation.

## Verification completed and remaining

All 15 automated tests passed locally, with PyCDR2 1.0.0 and Zenoh 1.10.1.
They check geometry, timestamp limits and ordering, interpolation, image byte
order and row stride, independently constructed ROS XCDR1 wire fixtures,
correct odometry covariance layout, and retry behavior after failed publication.
A separate local Zenoh query check confirmed cached TF replies are decoded.
Python compilation and CLI argument parsing also passed.

Full GPU inference and the new depth/TF subscriptions still require a live run
in the user's existing containers. The working reference's GPU inference was
verified earlier in this session; the new implementation has not yet been
validated against live Gazebo messages. Preserve both checks in the report.
The tests use a fake YOLO result when checking envelope/worker behavior.

After `Mapped:` output is verified, implement the assignment's semantic object
and observation tables, measured-radius deduplication, the four parameterized
queries, database export, route/map plot, and per-class evaluation. Furniture
geometry and rendered appearance can cause incorrect classes or background
depth estimates; validate object positions and labels against Gazebo ground
truth before treating detections as physical instances.

## References and provenance

- [Object detection assignment](https://aegean.ai/aiml-common/assignments/topics/ROS/object-detection)
- [Original turtlebot-maze detector](https://github.com/pantelis/turtlebot-maze/blob/main/detector/object_detector.py)
- [PyCDR2](https://pypi.org/project/pycdr2/)
- [Zenoh Python](https://github.com/eclipse-zenoh/zenoh-python)

The supplied reference is the starting point for YOLO and CLIP inference.
The new buffering, synchronization, TF tree, geometry, worker, wire-schema
corrections, and tests are documented in this package. The source credits that
reference so the report can explain the final data flow and the added work.
