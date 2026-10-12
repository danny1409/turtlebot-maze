# Object detection and semantic localization in the house world

Author: Manish

This report documents the verified Task 2 implementation, the user's verified
Task 3 database run, and the implemented Task 4 association rule. Task 4 code
and evaluation tools have passed local checks. A measured radius, real chair
duplicate/false-merge results, the complete route, per-class evaluation, and
final map still require simulator data; no synthetic test result is presented
as a house-world measurement.

## Task 2: Synchronization and depth

The detector retains the reference `main()` and nested callback structure. It
runs as standard Python without `rclpy`. It receives RGB, depth, odometry, and
TF through Zenoh and decodes the raw DDS CDR messages using `pycdr2` dataclasses.
The image byte field uses `sequence[uint8]`, matching the assignment definition.

RGB frames are limited to 10 Hz. Bounded histories store RGB, depth, odometry,
and dynamic transforms. For a candidate RGB keyframe, the detector chooses the
depth and odometry samples with the smallest absolute difference from the RGB
message's ROS header timestamp. It accepts a match only if both differences
are at most **200,000,000 ns (200 ms)**. Matching is based on timestamps rather
than arrival order. The 2-second wall-time arrival allowance gives delayed
messages time to arrive; it does not increase the 200 ms timestamp limit or
specify an inference deadline.

The synchronized odometry pose is checked against the last accepted keyframe
pose. A translation of at least 0.5 m or a wrapped yaw change of at least
15 degrees accepts another keyframe. The reference pose is updated only after
the valid keyframe has been processed and published.

Depth images use `32FC1`, meaning 32-bit floating-point metres. Decoding respects
the message's byte order and padded row size. RGB and depth must be registered
at 320 × 240 pixels for the supplied calibration.

For each clipped detection box, depth is estimated from the central **50% of
the width and 50% of the height**, approximately 25% of its area. The ROI's
lower bounds are rounded down and upper bounds up. Only finite, strictly
positive samples are retained. A detection is rejected if fewer than **9**
valid samples remain; otherwise their median is used. This avoids relying on
a single pixel between chair slats and reduces sensitivity to background near
box boundaries. It cannot guarantee that a large gap or background surface
will never dominate the region.

## Task 2: Transform chain and frame sources

For the full-precision bounding-box centre `(u, v)` and median depth `d`, the
camera optical point is:

$$
x_o = (u-160)d/277.13,\qquad
y_o = (v-120)d/277.13,\qquad z_o=d.
$$

The optical axes are x-right, y-down, and z-forward. The fixed optical-to-body
conversion is `(x_o, y_o, z_o) → (z_o, -x_o, -y_o)`. The camera's body-aligned
mount then adds `(0.064, -0.065, 0.094)` metres in `base_link`.

All transformations are 4 × 4 homogeneous matrices, applied in this order:

$$
p_{map}=T_{map\leftarrow base\_link}
T_{base\_link\leftarrow camera}
R_{camera\leftarrow optical}\,p_{optical}.
$$

The odometry message's `header.frame_id` and `child_frame_id` are checked. When
the parent is `odom`, its pose is not treated as a map pose. The detector
obtains `map ← odom` from the Nav2 localization transform broadcast on ROS
`/tf`, forwarded by the ROS–Zenoh bridge under the `tf` key. If the odometry
child is `base_footprint`, the required `base_footprint ← base_link` edge is
obtained from `tf_static`. An inverse edge can be inverted when necessary.
Dynamic transform samples are selected nearest to the synchronized odometry
timestamp, within 200 ms; static transforms are timeless. Missing or stale
transforms reject the frame rather than assuming an identity transform.

The camera's map position and optical orientation use the same composed chain:
`T_map_camera_optical = T_map_base_link @ T_base_camera @ R_camera_optical`.
Its translation supplies camera x/y/z, and its rotation supplies a normalized
xyzw quaternion. This includes the optical-axis rotation as well as the
robot's localization orientation.

The estimate is a visible surface point inferred from the box centre and
interior median depth. It is not assumed to equal the furniture model's origin.

## Task 2: Hand-computed transform test

The test uses the box `[219.2825, 82.287, 239.2825, 102.287]`, giving the
off-centre pixel `(229.2825, 92.287)`. The depth image contains 2 m throughout.
The robot pose in odom is `(1, 2, 0)` with yaw +90 degrees. The supplied
`map ← odom` transform has translation `(10, -3, 0.5)` and yaw −90 degrees.

| Stage | Expected coordinate in metres |
|---|---|
| Back-projected optical point | `(0.5, -0.2, 2.0)` |
| Optical-to-body rotation | `(2.0, -0.5, 0.2)` |
| Camera mount translation | `(2.064, -0.565, 0.294)` |
| Robot pose applied: point in odom | `(1.565, 4.064, 0.294)` |
| Map-to-odom transform applied: point in map | **`(14.064, -4.565, 0.794)`** |

`test_complete_chain_against_off_centre_hand_calculation` passes the box centre,
depth, robot pose, and map-to-odom transform through the implementation and
compares the output with this expected map point. The absolute tolerance is
**1 × 10⁻⁹ m**, with relative tolerance zero. This checks numerical transform
correctness; it does not claim nanometre accuracy for simulated sensor data.
The off-centre pixel exercises both transverse optical axes as well as depth.

## Task 2: Test and live-run evidence

Run from the repository root:

```bash
docker compose exec -T detector python -B -m unittest -v test_task2
```

The user's container ran all **11 tests successfully in 0.034 seconds** before
the live run `MSI-20261010T052957`. Tests cover the complete hand calculation,
axis signs, out-of-order timestamp selection, the inclusive 200 ms boundary,
median depth and insufficient samples, CDR decoding with row padding and
endianness, missing/stale TF rejection, the base-footprint edge and its inverse,
and optical camera orientation.

For the ten keyframes shown from that run, RGB-to-depth differences were
**0 ms**, and absolute RGB-to-odometry differences were **3–15 ms**. Waiting
before processing was **0–0.01 seconds**. The first keyframe took **1.51 seconds**
to process; later keyframes took **0.01–0.07 seconds**. Two initial frames with
missing `map ← odom` were rejected before valid keyframes were published.
These results verify live synchronization for the displayed run. Physical
object accuracy and domain-related class errors require the later ground-truth
evaluation.

## Task 3: Semantic database storage

The initial Task 3 implementation adapts the reference `embedding_ingest.py`
while retaining its command-line arguments, Zenoh session, and nested detection
callback. It subscribes to `tb/detections`, decodes each normalized 512-element
little-endian float32 CLIP embedding, validates finite geometry, and writes
parameterized SQL statements.

`semantic_schema.sql` uses the website's two tables and indexes with the same
columns and types. Only `IF NOT EXISTS` is added for repeatable initialization.
The `vector` extension supplies both `vector(512)` columns. Object creation and
its observation insert run in one transaction, so a failed observation insert
does not leave an object row without evidence.

| Stored data | Source and meaning |
|---|---|
| `semantic_objects.map_x/map_y/map_z` | Initially the detection's object point in map; then the count-weighted fused point after Task 4 merges |
| `object_observations.map_x/map_y/map_yaw` | The synchronized robot pose, transformed to map |
| Camera position and xyzw quaternion | The camera optical pose in map from the Task 2 transform chain |
| `seen_at_ns`, `first_seen_ns`, `last_seen_ns` | Integer ROS nanoseconds from the RGB keyframe's `timestamp_ns` |
| `range_m` | **Three-dimensional** distance from the camera centre to the back-projected point, `sqrt(x_o² + y_o² + z_o²)`, in metres |
| `bearing_rad` | `atan2(-x_o, z_o)`: zero along camera forward, **positive to the camera's left**, in radians |
| Bounding-box columns | Integer pixel bounds enclosing the clipped floating-point YOLO box: floor minima, ceil maxima |
| Observation embedding | The normalized CLIP vector for this detection |

The initial insert sets `observation_count = 1`, stores that detection as the
initial position/mean embedding, and gives `first_seen_ns` and `last_seen_ns`
the same ROS timestamp. During the original Task 3 run, each detection created a new estimate.
Those historical counts are not final physical-instance counts. The current
Task 4 worker adds same-class spatial association and running-average updates
to this same function, as described below.

The exact schema is used in a fresh database named `semantic_assignment` on
the existing `vector` PostgreSQL service (host port 5436). The earlier database
contains additional required columns from the previous implementation, so it
is preserved rather than altered during this step.

The submitted `test_task3.py` ran all **13 checks successfully** against a
temporary PGlite PostgreSQL build with pgvector, accessed through the actual
`psycopg2` driver. Eight checks validate input data. Five database checks verify
the linked records and ROS/pose fields, vector dimension enforcement, required
indexes and repeatable initialization, cascade deletion, and atomic rollback
when an observation insert fails. The user then created `semantic_assignment`
on the Compose PostgreSQL service, initialized the exact schema, and ran the
same **13 tests successfully in 0.231 seconds**, including all five database
checks. This confirms the storage checks on the actual Docker database as well
as the temporary local test database.

A separate handoff check generated a Task 2 envelope from real CDR messages
and passed it to Task 3. The ROS timestamp, normalized embedding, distinct
object/robot poses, and hand-computed map point `(14.064, -4.565, 0.794)` were
preserved. YOLO and CLIP inference were mocked for this local handoff check.

The reference worker decoded JSON and embeddings and inserted them into
`detection_embeddings`. The adaptation instead uses the required two tables,
keeps object and robot positions separate, checks that ROS timestamps remain
integers, and replaces independent autocommit inserts with one transaction.
It retains the reference's straightforward callback flow and dependencies.

## Task 3: Live verification

The user ran the adapted worker against `semantic_assignment` while the Task 2
detector published to `tb/detections`. Keyframes 11–24 produced **9 stored
observations linked to 9 initial object estimates**, without reported insert
errors. Keyframes without accepted detections produced zero observation rows.

A join of the five latest object and observation records showed separate
object and robot map positions, integer ROS timestamps, and **512-dimensional**
observation embeddings. For example, object #9 had estimated x/y
`(-1.219151, 3.746396)`, while its observing robot was at approximately
`(-2.214474, -0.464607)` in map. Its `seen_at_ns` was `1726200000000`, or
1726.2 seconds of ROS simulation time. The SQL result therefore preserves the
object/robot distinction and the sensor's ROS timestamp domain.

These results verify live storage, not final physical-instance accuracy.
The two `person` estimates #7 and #9 are about 0.030 m apart; their physical
identity cannot be established from coordinates alone. The current Task 4 worker implements same-class spatial association and
embedding-based candidate selection. Class correctness and duplicate/false-merge
rates still need verified object identities and observation evidence.

## Task 4: Association and updates

The implementation extends the existing `embedding_ingest.py`; it retains
`main()`, its nested Zenoh callback, the Task 2 detector, and the exact Task 3
tables. It does not add runtime classes, a new database model, class remapping,
or an appearance threshold. The class-name index supports the candidate filter.

The selected radius is a **3D Euclidean radius in map metres**, including its
boundary. For detection position `p`, candidate objects satisfy:

$$
class_{stored}=class_{detected},\qquad
\|\bar p_{stored}-p\|_2\le r.
$$

The comparison uses the object's current fused position. It does not compare
the detection to the robot position or merely to the object's latest raw point.

| Candidate set | Required behavior implemented |
|---|---|
| Empty | Insert an object with count 1, then its first observation |
| One object | Merge with that object, without an appearance threshold |
| Multiple objects | Select the smallest `mean_embedding <=> new_embedding`, meaning highest CLIP cosine similarity |

An exact cosine tie uses the smallest object ID, making replay deterministic.
Spatial distance determines eligibility; among multiple candidates it does
not override the CLIP ranking. The official pgvector documentation identifies
`<=>` as cosine distance: https://github.com/pgvector/pgvector.

For an existing object with observation count `n`, the update is:

$$
\bar p_{n+1}=\frac{n\bar p_n+p_{n+1}}{n+1},\qquad
e'=\frac{n\bar e_n+e_{n+1}}{n+1},\qquad
\bar e_{n+1}=\frac{e'}{\|e'\|_2}.
$$

The implementation uses the stored, normalized mean embedding in this rule
and normalizes again after every merge. This follows the assignment's recursive
rule; it does not claim to retain an unnormalized sum of all historical vectors.
An exactly cancelling mean cannot be normalized and rejects that detection
without committing a partial update.

Each accepted detection inserts its own observation row, including repeated
sightings. The object count increments by one. `last_seen_ns` is the maximum
ROS timestamp seen; `first_seen_ns` is the minimum. A late-arriving observation
therefore preserves the actual time interval while keeping its own timestamp.

Candidate lookup, creation or update, and observation insertion form one
transaction. Candidate rows are locked, and a transaction-scoped advisory lock
serializes same-class writers even when no object yet exists. A failed evidence
insert rolls back the fused position, embedding, count, and timestamps as well
as any new object. Locking is described in the PostgreSQL documentation:
https://www.postgresql.org/docs/17/explicit-locking.html.

## Task 4: Radius status and measurement definitions

**A final radius has not yet been selected from house-world measurements.**
The example **0.35 m** below is an explicit calibration trial, not a validated
result. The live CLI requires `--radius`, so an unspecified radius cannot
silently become the reported choice. The function's 0.35 m trial default keeps
the already installed Task 3 tests compatible.

`--record` stores the original keyframe envelopes in JSONL alongside live
database ingestion. This preserves the unfused object points, depth medians,
full-precision boxes, embeddings, camera poses, robot poses, and ROS timestamps
needed for evaluation. The two required tables and their indexed observation
retrieval remain unchanged.

`task4_measurements.py` provides a separate measurement workflow:

| Contribution | Measurement and meaning |
|---|---|
| Depth noise | With the robot stationary, sample 60 raw depth images using one fixed bounding box and the same central 50% median policy. Report sample standard deviation and the 95th percentile of absolute deviations from the temporal median. Holding the ROI fixed separates this measurement from detector-box movement. Propagate axial depth deviations by `sqrt(1+((u-160)/277.13)^2+((v-120)/277.13)^2)` to obtain a 3D point-error estimate. |
| Transform error | Independently obtain the robot's ground-truth map x/y/yaw at the synchronized odometry timestamp. Compare with the published map pose. Report translation and yaw errors, and a conservative planar point-error bound `translation_error + 2 * horizontal_robot_to_point_distance * sin(abs(yaw_error)/2)`. This evaluates planar pose error; it does not measure unprovided roll, pitch, or vertical ground truth. |
| Bounding-box variation | For verified sightings of the same object from camera positions within 5 cm and orientations within 3 degrees, report the box-centre shift expressed in metres: `mean_depth/277.13 * sqrt(delta_u^2 + delta_v^2)`. This is a controlled repeat-view estimate, with small residual viewpoint differences. |
| Viewpoint change | Compare mapped points for verified sightings of the same physical object from viewing directions at least 90 degrees apart. Directions are measured around the median estimated object point. Report median, 95th percentile, and maximum 3D displacement. Opposite surfaces can differ even with a perfect transform. |
| Combined association variation | Report the distance of each verified same-class sighting from the running mean of that physical object's earlier sightings, in recorded arrival order. Also report pairwise same-object position displacement. These are direct observations of the spatial gate's required tolerance. |

The `labels` command creates a CSV with detection IDs and measurement fields.
Fill `object_label` with a verified physical identity such as
`dining_chair_1` through `dining_chair_4`, based on the world models and visual
evidence. Do not assign labels by the radius being evaluated or by the fused
database's IDs: that would make the evaluation circular. Unlabeled detections
still affect replay association but are excluded from identity scoring.

For transform measurements, fill `true_robot_x`, `true_robot_y`, and
`true_robot_yaw` from independent simulation truth, not AMCL, `/odom`, or the
same `/tf` chain under test. Use metres and radians in **map**, at the exported
`odom_timestamp_ns`. Gazebo world coordinates first need a documented
world-to-map alignment. If that alignment or synchronized truth is unavailable,
leave these columns blank; the tool reports missing measurements instead of
inventing zero transform error.

Repeat the fixed-ROI depth probe on representative chair views and other
important objects. The command below accepts one depth CSV at a time; keep
separate files and document their results. A noiseless simulated depth probe
may legitimately measure zero temporal noise, while boxes and viewpoints
still cause nonzero mapped-position variation.

Use the combined running-mean residual and the component measurements to
choose a reasonable radius range. Then select and justify the final value
from the annotated trial table, documenting the observed duplicate/false-merge
tradeoff. Component errors are not assumed statistically independent, and
their percentiles are not blindly added into an artificial total.

## Task 4: Chair experiment and metrics

Record at least four verified dining chairs, initially seen separately,
then revisited from both sides of the table. Include a return to the same
camera pose for the box-variation measurements. Keep the detector's 0.5 m /
15-degree keyframe policy and 200 ms synchronization rule.

Replay the same recording from an empty set at several radii, for example
0.15, 0.25, 0.35, 0.50, and 0.75 m. These are trial values, not measured
recommendations. If the measured variation extends beyond this range, include
additional radii. Each trial uses fused running positions and normalized mean
embeddings. Compare the required CLIP rule with a **nearest-position tie-break
ablation**. The ablation affects only multiple-candidate selection and is
available in the evaluation tool; the live worker always uses CLIP.

| Output metric | Definition |
|---|---|
| `duplicate_records` | For each verified physical object, number of associated database IDs minus one, bounded below by zero; summed over objects |
| `false_merge_records` | Number of database IDs containing evidence from more than one verified physical object |
| `duplicate_pair_rate` | Fraction of same-physical-object observation pairs assigned to different database IDs |
| `false_merge_pair_rate` | Fraction of different-physical-object observation pairs assigned to the same database ID |
| `candidate_zero/one/multiple` | Number of detections experiencing each candidate-set case |

The pair denominators and raw error counts are exported as well as rates.
Missing denominators produce no rate, not a misleading zero. A merged record
can also participate in duplicate counts; the metrics describe separate errors.
The four-chair subset should have labels for all four physical chairs and
repeated sightings of each. Report how many detections were labeled and which
classes were studied.

The offline CLIP implementation is checked against real PostgreSQL association
results. Float32 rounding of mean embeddings matches pgvector storage; exact
floating-point ties near a decision boundary still deserve inspection.

A larger radius can cause a false merge when there is exactly one candidate:
the assignment requires an unconditional merge in that case. CLIP therefore
cannot resolve every nearby-chair ambiguity. A small radius may create a new
record before the correct existing object enters the candidate set. These
limitations must be measured rather than hidden by changing the required rule.

## Task 4: Verification

The updated code passed **33 automated checks**: the existing 13 Task 3 checks,
14 new PostgreSQL deduplication checks, and six measurement checks. PostgreSQL
checks used the actual `psycopg2` driver with a temporary PGlite PostgreSQL build
and pgvector. They cover zero/one/multiple candidates, class separation, XYZ
radius and its inclusive boundary, fused-position eligibility, highest-CLIP
selection over a nearer candidate, deterministic ties, count-weighted positions
and normalized embeddings, late ROS timestamps, every-observation insertion,
failed-observation rollback, and invalid-radius/zero-mean rejection.

Measurement checks verify pair metrics, missing-data reporting, template
column alignment and ROS odometry time, repeat-view and opposite-view
calculations, repeated-ID rejection, and a synthetic four-chair radius/tie-break
case. The synthetic case validates code behavior; it is **not** evidence about
the real Gazebo chairs or a justification for any particular radius.

A CLI workflow check created a label template, generated radius-trial CSV and
measurement JSON, and replayed a synthetic eight-observation recording into
real PostgreSQL through the updated worker. It produced four objects with
eight observations and refused replay into the resulting nonempty database.
The live raw-depth probe and the Task 4 worker still need to run on the user's
simulator. No Task 4 simulator measurements are claimed yet.

## Task 4: Install and start calibration

Stop the manually started Task 3 ingest worker with Ctrl+C before copying the
updated source. Download these four current files to Windows Downloads:

- `embedding_ingest_task4.py`: replacement for `detector/embedding_ingest.py`.
- `test_task4.py`: new tests; keep the already installed `test_task3.py`.
- `task4_measurements.py`: separate measurement and replay tool.
- `Assignment2_Report.md`: current report.

Keep the Task 2 detector, its tests, and the Task 3 SQL schema as installed.

```bash
cd ~/robotics/turtlebot-maze &&
task4_backup_dir=$(mktemp -d "$HOME/robotics/task4-backup-XXXXXX") &&
cp detector/embedding_ingest.py "$task4_backup_dir/" &&
cp /mnt/c/Users/mp528/Downloads/embedding_ingest_task4.py detector/embedding_ingest.py &&
cp /mnt/c/Users/mp528/Downloads/test_task4.py detector/test_task4.py &&
cp /mnt/c/Users/mp528/Downloads/task4_measurements.py detector/task4_measurements.py &&
mkdir -p assignments/assignment-2 &&
cp /mnt/c/Users/mp528/Downloads/Assignment2_Report.md assignments/assignment-2/REPORT.md &&
docker compose exec -T -e TASK3_TEST_DB=1 -e TASK4_TEST_DB=1 detector \
  python -B -m unittest -v test_task3 test_task4
```

The tests use a private schema inside `semantic_assignment`, leaving its
existing nine Task 3 records untouched. The detector process does not require
a restart for these ingest-worker changes.

Create a separate calibration database on the existing `vector` service:

```bash
docker compose exec -T vector psql -U postgres -d postgres -v ON_ERROR_STOP=1 <<'SQL'
SELECT 'CREATE DATABASE semantic_task4_calibration'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'semantic_task4_calibration')
\gexec
SQL
```

Start the updated worker in its own WSL terminal before moving the robot:

```bash
cd ~/robotics/turtlebot-maze
docker compose exec detector python -u -B embedding_ingest.py \
  --pg-db semantic_task4_calibration --radius 0.35 \
  --record /tmp/task4/calibration.jsonl
```

The 0.35 m setting is only a starting trial. JSONL preserves the original
measurements so evaluation can retry other radii without repeating inference.
Stop the worker with Ctrl+C after the calibration route, ending with a chair
visible for the stationary depth probe. `/tmp` avoids the detector working
folder's previously observed write-permission problem. Copy the raw recording
to the repository immediately after stopping, before recreating the container:

```bash
mkdir -p results/task4 &&
docker compose cp detector:/tmp/task4/calibration.jsonl results/task4/calibration.jsonl
```

Inspect reuse of object IDs and preservation of all evidence rows:

```bash
docker compose exec -T vector psql -U postgres -d semantic_task4_calibration -c "
SELECT s.object_id, s.class_name, s.map_x, s.map_y, s.map_z,
       s.observation_count, count(o.observation_id) AS evidence_rows
FROM semantic_objects s
LEFT JOIN object_observations o USING (object_id)
GROUP BY s.object_id
ORDER BY s.object_id;"
```

Repeated accepted sightings assigned to one object should increase both counts.
This consistency check does not establish physical identity on its own.

## Task 4: Run the measurements

Create the annotation template after recording:

```bash
docker compose exec -T detector python -B task4_measurements.py labels \
  /tmp/task4/calibration.jsonl --out /tmp/task4/labels.csv
docker compose cp detector:/tmp/task4/labels.csv results/task4/labels.csv
```

Edit `results/task4/labels.csv` on the host. Fill physical identities
and independently measured robot truth where available; keep the exported
coordinates and detection IDs as reference evidence. The command refuses to
overwrite an existing label file.

Copy the edited labels back before evaluation:

```bash
docker compose cp results/task4/labels.csv detector:/tmp/task4/labels.csv
```

For a depth probe, stop the robot while the selected chair is visible. Obtain
its **current** bounding box from the latest recorded detection while remaining
at that camera pose; verify the object in the camera image in RViz. Use its
actual four pixel coordinates, not a box from an earlier viewpoint. In a WSL
terminal, set these four variables to
those numbers, then run:

```bash
docker compose exec -T detector python -B task4_measurements.py depth-noise \
  --bbox "$task4_xmin" "$task4_ymin" "$task4_xmax" "$task4_ymax" \
  --samples 60 --out /tmp/task4/chair-depth-noise.csv
```

The four `task4_*` variables are intentionally not assigned invented pixel
values here. The probe samples raw depth without altering keyframe gating or
running YOLO on stationary frames. It uses a fixed central ROI and times out
after 45 seconds if it cannot obtain 60 valid images.

Evaluate annotated chairs with both tie-break policies:

```bash
docker compose exec -T detector python -B task4_measurements.py evaluate \
  /tmp/task4/calibration.jsonl --labels /tmp/task4/labels.csv \
  --depth-csv /tmp/task4/chair-depth-noise.csv \
  --class-name chair --radii 0.15 0.25 0.35 0.50 0.75 \
  --out-json /tmp/task4/measurements.json \
  --out-csv /tmp/task4/radius-trials.csv
```

Omit `--class-name chair` to study all captured classes as well. A class filter
affects only the evaluation subset; the live detector/worker retain all COCO
labels. The JSON identifies missing measurements, and the CSV contains the
radius, tie-break, coverage counts, duplicate counts, and false-merge counts.

Copy completed evidence into the repository's result directory:

```bash
mkdir -p results/task4
docker compose cp detector:/tmp/task4/. results/task4/
```

Add a report table containing the actual component measurements, trial results,
selected radius, and reasons for accepting its tradeoff. No placeholder value
should be presented as a finished empirical result.

## Task 4: Store the selected result

Once the measured radius has been selected, create a fresh database so the
initial trial radius and earlier Task 3 duplicates do not bias the final map:

```bash
docker compose exec -T vector psql -U postgres -d postgres -v ON_ERROR_STOP=1 <<'SQL'
SELECT 'CREATE DATABASE semantic_task4_final'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'semantic_task4_final')
\gexec
SQL
```

Set `task4_selected_radius` to the justified value in metres, then replay all
recorded detections through the actual database implementation:

```bash
docker compose exec -T detector python -B task4_measurements.py replay \
  /tmp/task4/calibration.jsonl --pg-db semantic_task4_final \
  --radius "$task4_selected_radius"
```

Replay requires empty assignment tables and does not delete existing records.
For subsequent live collection into the final database, pass this same selected
radius to `embedding_ingest.py` and record the new observations separately.
Do not ingest the same recorded route twice into the final tables.
