# Semantic object database

This adds Tasks 3–5 to the synchronized detector installed in `detector/`.
It uses the existing detector container's `psycopg2` and Zenoh packages. No
Docker rebuild is needed. The required two tables retain the assignment's
columns, plus provenance and raw object measurements for replay and evaluation.

## Install and check

Download `semantic-database.zip` into Windows Downloads, then run in WSL:

```bash
cd ~/robotics/turtlebot-maze
python3 -m zipfile -e /mnt/c/Users/mp528/Downloads/semantic-database.zip detector/
docker compose up -d vector
docker compose exec -T vector pg_isready -U postgres -d vectordb
docker compose exec -T detector python semantic_ingest.py --init-only
docker compose exec -T -e SEMANTIC_TEST_DB=1 detector python -m unittest -v test_semantic_db
```

Wait for `accepting connections` before initialization. If PostgreSQL remains
unavailable, inspect `docker compose logs --tail=60 vector`.

The default connection is `127.0.0.1:5436`, database `vectordb`, user/password
`postgres`, matching the course compose service and detector host networking.
All database programs accept `--db-host`, `--db-port`, `--db-name`, `--db-user`,
and `--db-password` before their subcommand. Standard `PGHOST`, `PGPORT`,
`PGDATABASE`, `PGUSER`, and `PGPASSWORD` environment variables also override
defaults. These defaults are for the local course environment.

The SQL creates missing tables and indexes. It never drops existing assignment
data. Existing incompatible tables with these same names produce an error;
inspect their definitions before choosing a separate trial database. The
integration tests create their own randomly named schema, use only its two
tables, then remove that schema. They do not truncate the assignment tables.

## Collect a traversal

Keep Gazebo, localization, the Zenoh router/bridge, and the detector running.
In another terminal, start ingestion **before** moving the robot:

```bash
cd ~/robotics/turtlebot-maze
docker compose exec detector python -u semantic_ingest.py \
  --radius 0.35 --record /tmp/semantic-keyframes.jsonl
```

Move at least 0.5 m or turn at least 15 degrees to produce another keyframe.
Visit known furniture from different positions and directions. Check that
the first observation creates an object and later observations can merge
into it. Empty keyframes are still recorded to preserve accepted-pose coverage.
Press Ctrl+C in this terminal to stop ingestion. This does not stop Gazebo or
the detector.

The JSONL file is inside the running detector container. Copy it out before
recreating or deleting that container. Database records persist in the existing
PostgreSQL volume.

## Association and evidence

For each detection, the program:

1. Validates schema version 2, the map frame, integer ROS nanoseconds, matching
   synchronization deltas within 200 ms, finite geometry, and a normalized
   little-endian float32 embedding with 512 elements.
2. Acquires a transaction advisory lock for the detection class and checks the
   unique source detection identifier. Re-delivery/replay does not increment
   the count or create another observation.
3. Finds candidates with the same class and a **three-dimensional Euclidean
   distance** from the fused map position no greater than the radius.
4. Creates an object if no candidate exists; merges with the sole candidate;
   or, for multiple candidates, chooses the smallest pgvector `<=>` cosine
   distance. Exact ties use the lowest object identifier. There is no extra
   similarity threshold or whitelist.
5. Updates position using `(n * old_position + new_position) / (n + 1)`.
   Updates the mean embedding by the same rule, then L2-normalizes it. This
   follows the assignment's recursive normalized-mean rule.
6. Inserts one evidence row and commits the fused estimate and observation
   atomically. A failed evidence insert rolls back the object change.

`semantic_objects.map_x/map_y/map_z` describe the **object**. Required
`object_observations.map_x/map_y/map_yaw` describe the **robot**. Camera
columns contain the map pose of the optical camera. Extra `object_x/object_y/
object_z` columns preserve each raw object estimate so position spread can be
examined without substituting the robot pose. The source ID, run, epoch,
keyframe, and association radius are recorded with each observation.

`range_m` is the 3D distance from the camera centre. `bearing_rad` is zero
along the camera's forward direction and positive to its left. All `_ns`
timestamps are ROS time. Queries group histories by run and clock epoch before
ordering by ROS time. Numerical `first_seen_ns` and `last_seen_ns` are timestamp
extrema; they are not elapsed wall time across simulation resets. For a clean
evaluation, collect one uninterrupted run in a trial database and maintain a
consistent map frame.

An embedding model change within a class is rejected, rather than comparing
incompatible feature spaces. Input messages from the original unsynchronized
reference detector are rejected explicitly.

## Radius measurements still required

**0.35 m is a provisional starting value, not a measured result.**

Label repeated detections of the same known physical object from multiple
views. Measure their raw map-position deviations and variation in depth,
bounding-box centre, and localization/TF; compare those deviations with the
separations of nearby objects of the same class. Object surface estimates may
move with viewpoint even when the physical object is stationary.

Try smaller and larger radii against the same recorded traversal in separate
trial databases. Use manually established physical identities and SDF ground
truth to count duplicate records and false merges. Spread within an already
associated record alone cannot justify the radius: that record could contain
multiple physical objects. Document the measured justification and selected
radius in the final report.

## The four parameterized queries

```bash
docker compose exec -T detector python semantic_queries.py nearest chair 0 0
docker compose exec -T detector python semantic_queries.py region -10 10 -10 10
docker compose exec -T detector python semantic_queries.py histogram
docker compose exec -T detector python semantic_queries.py poses 1
```

Replace `1` with an object identifier returned by `nearest` or `region`.
The nearest query uses horizontal distance from the requested map point.
Rectangle bounds are inclusive. Queries print JSON. SQL values are passed
through psycopg2 `%s` binding; no class, coordinate, or identifier is interpolated
into SQL text. Executable SQL is in `semantic_db.py`.

## Export and plot actual collected data

```bash
docker compose exec -T detector python semantic_queries.py export \
  --output /tmp/semantic-map-export.json
docker compose exec -T detector python semantic_map.py /tmp/semantic-map-export.json \
  --keyframes /tmp/semantic-keyframes.jsonl --output /tmp/semantic-map.png

mkdir -p results
docker compose cp detector:/tmp/semantic-map-export.json results/semantic-map-export.json
docker compose cp detector:/tmp/semantic-keyframes.jsonl results/semantic-keyframes.jsonl
docker compose cp detector:/tmp/semantic-map.png results/semantic-map.png
```

The export includes both tables and full embeddings from one repeatable-read
snapshot. The plot labels fused positions by class and object ID, shows raw
object measurements, and overlays the accepted robot poses, including empty
keyframes from the recording. It plots estimates in map metres; it is not a
Gazebo floor-plan overlay and is not ground-truth validation.

## Recover or replay

```bash
docker compose exec -T detector python semantic_ingest.py \
  --replay /tmp/semantic-keyframes.jsonl --record /tmp/semantic-replay.jsonl --radius 0.35
```

Replaying into the same database leaves previously inserted source IDs
unchanged. To compare association radii, replay into a separate database with
the same schema, using the database arguments before `--replay`. A radius change
alone does not re-associate observations already stored in the same database.

Live delivery buffers received messages in memory. Transient database
disconnects retain the current observation and retry. Each validated keyframe
is appended to JSONL before persistence. A restart can recover from this
recording; messages published before ingestion began may be unavailable unless
the router has storage. The recording is append-only and may include repeated
deliveries; plotting deduplicates its run/epoch/keyframe keys.

## Tests and remaining deliverables

There are 11 local validation/mathematical tests and 8 optional real
PostgreSQL/pgvector integration tests. The integration tests cover object and
robot pose semantics, 3D radius boundaries, cosine-based selection when it
differs from nearest distance, running averages, duplicate delivery,
transaction rollback, model compatibility, all four queries, SQL parameter
binding, and export. Enable them with `SEMANTIC_TEST_DB=1` as shown above.

Local validation tests have passed. Real database tests must be run against
the user's available PostgreSQL service; they were not run in the assistant's
workspace. This package provides code and commands, not an evaluation dataset
or fabricated metrics. The SDF inventory, measured radius, route evaluation,
per-class precision/recall/detection rate, duplicate rate, false merges, and
final report still need data from the user's traversal.

## Commit these files

```bash
git add detector/semantic_schema.sql detector/semantic_core.py \
  detector/semantic_db.py detector/semantic_ingest.py \
  detector/semantic_queries.py detector/semantic_map.py \
  detector/test_semantic_db.py detector/README_semantic_database.md
git commit -m "Add semantic object database, association, queries and export"
git push origin main
```

These files implement the course specification independently around the
earlier synchronized detector. The existing `turtlebot-maze` compose services
provide the simulation, Zenoh connection, and PostgreSQL deployment.
