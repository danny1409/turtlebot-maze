# Assignment 2 — Object detection and semantic localization

**Author:** Manish  Chandra Tadishetty

Three runtime files: object_detector.py, embedding_ingest.py, semantic_schema.sql.
The four task tests are under detector/tests/. No notebook is required.
The evaluated run, report, complete JSON/SQL exports, queries, world inventory,
matched PNGs/JSONL, annotations and coverage map are under results/assignment-2.

## Start Compose

From the course repository, with its GPU/Docker prerequisites installed:

~~~bash
cd ~/robotics/turtlebot-maze
docker compose up -d --build demo-world-house zenoh-router zenoh-bridge vector detector
~~~

The first build takes several minutes. Wait about 30 seconds(it took me 9-10 minutes for the whole build), then check that all five services are `Up`:

```bash
docker compose ps
```

If `zenoh-bridge` has exited or the detector logs `Missing time-valid TF`, restart them in this order:

```bash
docker compose restart zenoh-bridge
sleep 15
docker compose restart detector
```

Only one world service should run. After a restart, preserve existing commands
and mounts with docker compose start using those same five service names.
The detector uses yolov8n.pt, confidence 0.3, batched ViT-B-32/laion2b_s34b_b79k,
all COCO classes, 10 Hz RGB, 0.5 m/15-degree keyframes and a strict 200 ms
depth/odometry timestamp limit. Model weights may download on first startup.
The semantic database service is vector on port 5436.

## Create a fresh demo database and run tests

~~~bash
docker compose exec -T vector psql -U postgres -d postgres -v ON_ERROR_STOP=1 <<'SQL'
SELECT 'CREATE DATABASE semantic_assignment_demo'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname='semantic_assignment_demo')
\gexec
SQL

docker compose exec -T detector python -B embedding_ingest.py   --pg-db semantic_assignment_demo --init-only

docker compose exec -T -e PGDATABASE=semantic_assignment_demo   -e TASK3_TEST_DB=1 -e TASK4_TEST_DB=1 -e TASK5_TEST_DB=1 detector   python -B -m unittest discover -s tests -p 'test_task[2345].py' -v
~~~

Tests use private temporary schemas. The worked off-center transform expects
(14.064,-4.565,0.794) m at absolute tolerance 1e-9 m; REPORT.md gives the steps.

## Capture and ingest a new route

Choose a fresh database and directory for each restarted simulation clock.
In terminal 1:

~~~bash
docker compose exec detector python -u -B embedding_ingest.py capture-views   --connect tcp/localhost:7447 --out /tmp/assignment2-demo/views
~~~

Wait for camera capture ready, then in terminal 2:

~~~bash
docker compose exec detector python -u -B embedding_ingest.py   --connect tcp/localhost:7447 --pg-db semantic_assignment_demo --radius 0.50   --record /tmp/assignment2-demo/keyframes.jsonl
~~~

If static TF remains missing after subscriptions start, restart zenoh-bridge
once. Drive using the existing teleop/RViz controls. A stationary robot accepts
no additional motion keyframes; empty keyframes still have saved images.
Stop the robot (k in teleop or cancel Nav2), then Ctrl+C the ingest and capture
terminals. Do not stop an ingest process before the route ends.

## Four parameterized queries

~~~bash
docker compose exec -T detector python -B embedding_ingest.py query --pg-db semantic_assignment_demo nearest chair 0 0
docker compose exec -T detector python -B embedding_ingest.py query --pg-db semantic_assignment_demo region -12.5 12.5 -12.5 12.5
docker compose exec -T detector python -B embedding_ingest.py query --pg-db semantic_assignment_demo histogram
docker compose exec -T detector python -B embedding_ingest.py query --pg-db semantic_assignment_demo poses 1
~~~

Use an existing object ID from the region output. Values are bound through
psycopg2, not interpolated into SQL. The histogram counts objects, not sightings.

## Export and plot the demo

~~~bash
docker compose exec -T detector python -B embedding_ingest.py query   --pg-db semantic_assignment_demo export --output /tmp/assignment2-demo/semantic-map-export.json
docker compose exec -T detector python -B embedding_ingest.py map   /tmp/assignment2-demo/semantic-map-export.json   --keyframes /tmp/assignment2-demo/keyframes.jsonl   --output /tmp/assignment2-demo/semantic-map.png
mkdir -p results/assignment2-demo
docker compose cp detector:/tmp/assignment2-demo/. results/assignment2-demo/
~~~

The full keyframe JSONL includes empty frames and supplies coverage. Raw
observation rows alone cannot establish the full traversal.

## Restore the supplied evaluated snapshot

Use a fresh empty database; do not load the same snapshot twice:

~~~bash
docker compose exec -T vector psql -U postgres -d postgres   -v ON_ERROR_STOP=1 -c 'CREATE DATABASE semantic_assignment_snapshot;'
docker compose exec -T vector psql -U postgres -d semantic_assignment_snapshot   -v ON_ERROR_STOP=1 < results/assignment-2/semantic-map.sql
docker compose exec -T detector python -B embedding_ingest.py query   --pg-db semantic_assignment_snapshot histogram
~~~

Expected snapshot: 94 objects, 207 observations(might vary on your path chosen and time travelled). Query outputs and all eight
passing Task 5 Docker checks are included. Database record counts are distinct
from physical matches; read the evaluation policy and uncertainty in REPORT.md.

## Radius analysis and evidence

The evaluated radius remains 0.50 m, selected from the historical depth,
box/viewpoint and independent localization measurements. The final full-route
chair trials preserve that frozen choice and quantify its errors.
The report identifies native-image uncertainty and the map discontinuity.
World assets include model/geometry definitions; texture reproduction relies
on the course repository's original assets. The exact evaluated SDF is saved.
Clean-clone GPU execution must be verified in the user's/course environment.
