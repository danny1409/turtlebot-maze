# Tutorial: Navigating a TurtleBot with StellaVSLAM + Nav2, without a prior map

**Stack:** ROS 2 Jazzy · Gazebo Harmonic · Nav2 · stella_vslam_ros (RGB-D) · Docker on WSL2
**Simulation:** [`turtlebot-maze`](#) repo, `demo-world-enhanced` (textured walls, ArUco markers),
TurtleBot3 Waffle with a simulated Intel RealSense D435i.

> Status: every command, config value and code reference below was checked against the
> stella_vslam / stella_vslam_ros / Nav2 Jazzy sources and the live simulation, and the
> whole pipeline was run end to end (section 8 shows the reference run).

## 1. What we are building and why

Nav2 normally localises the robot with **AMCL** on a **map recorded beforehand**
(`map_server`). Here there is no prior map. Instead:

| Job | Normally | In this tutorial |
|---|---|---|
| Where is the robot? (`map → odom`) | AMCL particle filter on a saved occupancy grid | **StellaVSLAM**, from camera images, building its own landmark map as it drives |
| Where are the obstacles? (global costmap) | static layer from the saved map + sensors | **rolling window** around the robot, from the LiDAR only |
| Local planning, control, recovery | Nav2 | Nav2 (unchanged) |

```
 Gazebo (D435i RGB-D) ──/camera/color/image_raw──────────┐
                     └─/camera/depth/image_rect_raw──────┤
                                                         ▼
                                           stella_vslam_ros (RGB-D)
                                                         │  TF  map → odom
 Gazebo diff-drive ── TF odom → base_footprint ─┐        │
 Gazebo LiDAR ─────── /scan ───────────┐        ▼        ▼
                                       └──►  Nav2 (no map_server, no AMCL)
                                                         │ /cmd_vel
                                                         ▼
                                                     TurtleBot
```

**Recursive state estimation view.** Wheel odometry gives a smooth, high-rate but
drifting estimate (`odom → base_footprint`). StellaVSLAM corrects it from visual
landmarks every camera frame, and when it recognises a place it has seen before it
closes the loop and re-optimises its whole map, so the correction stays globally
consistent. Nav2 only ever sees the composed result `map → odom → base_footprint`.
Section 6 walks through exactly how this happens, frame by frame.

**Why RGB-D and not monocular.** A single camera cannot observe metric scale
(Task 1 showed the drift this causes). Nav2 plans in metres, so a scale-ambiguous pose
is unusable. The simulated D435i provides depth in metres, so every landmark has
metric depth from the first frame and the map is in metres.

## 2. Prerequisites

- WSL2 + Docker, and the `turtlebot-maze` repo built (`docker compose build`) with
  `ROS_DISTRO=jazzy` in its `.env`.
- A working folder, `~/vslam_nav2`, holding the five files below. Their full contents
  are in **Appendix A**; save each one under the name shown (note the `config/`
  subfolder), then `chmod +x ~/vslam_nav2/*.sh`.

| File | Purpose |
|---|---|
| `Dockerfile.stella_vslam_ros_jazzy` | image with stella_vslam + stella_vslam_ros on ROS 2 Jazzy (no GUI viewer; use RViz) |
| `config/tb3_d435i_rgbd.yaml` | StellaVSLAM camera config for the simulated D435i |
| `make_nav2_params_vslam.py` | turns the repo's `nav2_params.yaml` into a no-map version |
| `run_sim_no_map.sh` | world + robot + RViz + Nav2 without map_server/AMCL |
| `run_vslam.sh` | starts StellaVSLAM and publishes `map → odom` |

## 3. Build the StellaVSLAM image (once, 30-60 min)

```bash
cd ~/vslam_nav2
docker build -t stella_vslam_ros:jazzy -f Dockerfile.stella_vslam_ros_jazzy . \
  --build-arg NUM_THREADS=2
```

`NUM_THREADS=2` matters on WSL: each parallel C++ compile job can use 1-2 GB, and more
jobs crash the WSL VM.

### 3.1 Why a separate container

The simulation already runs in the `turtlebot-maze` images. StellaVSLAM gets its own
image instead of being added to them because:

- its C++ dependencies (g2o, a specific Eigen/OpenCV combination, FBoW) are large and
  only needed by this one node;
- the class images stay untouched, so other assignments keep working;
- ROS 2 does not care which container a node runs in: with `--net=host --ipc=host`
  (the same settings every `turtlebot-maze` service uses) nodes in both containers
  discover each other and exchange messages as if they were in one.

### 3.2 Dockerfile walkthrough

`Dockerfile.stella_vslam_ros_jazzy` follows the structure of the official
stella_vslam_ros Dockerfile (which targets Humble), layer by layer:

| Layer | What it does | Why |
|---|---|---|
| `FROM ros:jazzy` | ROS 2 Jazzy base on Ubuntu 24.04 | must match the simulation's distro (`ROS_DISTRO=jazzy` in the repo's `.env`) so message definitions and DDS versions agree |
| `apt-get install ...` | build tools; `libopencv-dev`, `libeigen3-dev`, `libyaml-cpp-dev`, `libsuitesparse-dev`, `libatlas-base-dev`, `libsqlite3-dev`; `ros-jazzy-libg2o`; `cv-bridge`, `image-transport`, `message-filters`, `tf2-*` | OpenCV: ORB features and image handling. Eigen: all the geometry. g2o (+ SuiteSparse/ATLAS for its sparse solvers): the graph optimiser behind pose optimisation, bundle adjustment and loop closure. yaml-cpp: the camera config. sqlite3: optional map format. The ROS packages are what the wrapper uses to receive images and publish TF |
| `git clone stella_vslam` + `cmake`/`make install` | builds the SLAM library itself into `/usr/local` | `CMAKE_PREFIX_PATH=/opt/ros/jazzy/lib/cmake` makes CMake find the g2o that came from `ros-jazzy-libg2o`. The tracking patch (3.3) is applied before compiling |
| `git clone stella_vslam_ros` + `colcon build` | builds the ROS 2 wrapper (`run_slam`) in `/ros2_ws` | this is the node that turns ROS images into SLAM input and SLAM poses into TF; two Jazzy patches (3.3) are applied first |
| `curl orb_vocab.fbow` | downloads the ORB bag-of-words vocabulary to `/vslam` | used for relocalisation and loop detection (place recognition); `test -s` fails the build if the download is empty |
| `.bashrc` lines | source ROS and the workspace in interactive shells | convenience for `docker run -it ... bash` |

What is deliberately **not** in the image: the Pangolin/socket viewers (RViz shows the
pose and keyframes), and any simulation or Nav2 packages (those live in the
simulation container).

**Layer order matters for rebuild time.** Docker caches each layer and only rebuilds
from the first layer that changed. The order above goes from rarely changing (apt
packages) to most likely to change (the wrapper). When the wrapper needed a fix, the
rebuild took about 100 s; changing the core library patch rebuilt both (about 6 min);
the 75 s apt layer was never repeated.

### 3.3 Patches for Jazzy and for simulation

The official Dockerfiles target Humble. Three patches are applied in the Dockerfile,
each followed by a check (`grep`) that fails the build if the patch did not apply,
so a future upstream change cannot silently produce a broken image:

| Problem | Symptom | Fix |
|---|---|---|
| On Jazzy, `ament_export_targets(... HAS_LIBRARY_TARGET)` already registers the `library_path` environment hook; stella_vslam_ros registers it again | CMake: `Files to be generated by multiple different commands: .../library_path.dsv` | delete the extra `ament_environment_hooks(...)` block |
| `cv_bridge/cv_bridge.h` no longer exists in Jazzy (and the `tf2_ros/*.h` headers are deprecated) | `fatal error: cv_bridge/cv_bridge.h: No such file or directory` | rewrite those includes to `.hpp` |
| In stella_vslam, `search_local_landmarks()` treats "every visible landmark is already matched" as a tracking failure. A robot standing still in simulation produces identical frames, so this happens on the first frame after initialisation and whenever the robot stops | `new map created ...` immediately followed by `projection candidate not found`, `tracking lost`, `resetting system`, repeating every frame | return success instead; the pose optimisation that follows still requires >= 20 tracked landmarks |

The third one is the interesting one for this course: it is invisible with real cameras
(sensor noise makes consecutive frames differ), and only appears with perfect,
noise-free simulated images.

### 3.4 Running the container

`run_vslam.sh` is a single `docker run`:

```bash
docker run -it --rm --net=host --ipc=host -v "$T3":/task3 stella_vslam_ros:jazzy \
  bash -c "source ... && ros2 run stella_vslam_ros run_slam -v /vslam/orb_vocab.fbow \
    -c /task3/config/tb3_d435i_rgbd.yaml --viewer none --map-db-out /task3/maze_map.msg \
    --ros-args -r camera/depth/image_raw:=/camera/depth/image_rect_raw -p ..."
```

| Flag | Purpose |
|---|---|
| `--net=host` | the container shares the WSL network stack, so DDS discovery finds the simulation's nodes |
| `--ipc=host` | shares memory with the host; Fast DDS (ROS 2's default middleware) moves large messages such as images through shared memory, and without this nodes discover each other but no image data arrives |
| `-v "$T3":/task3` | mounts the working folder (`~/vslam_nav2`) as `/task3`: the config comes in and the saved map comes out through it |
| `--rm` | the container is thrown away on exit; nothing inside it needs to persist |

The ROS parameters after `--ros-args` are explained in section 7.

## 4. The camera configuration

Facts measured in the running simulation:

| Check | Command | Result |
|---|---|---|
| image frame | `ros2 topic echo /camera/color/image_raw --once --field header.frame_id` | `camera_depth_frame` |
| colour encoding | `... --field encoding` | `rgb8` |
| depth encoding | `ros2 topic echo /camera/depth/image_rect_raw --once --field encoding` | `32FC1` (metres) |
| QoS | `ros2 topic info /camera/color/image_raw -v` | RELIABLE |
| camera pose | `ros2 run tf2_ros tf2_echo base_footprint camera_depth_frame` | xyz (0.069, -0.037, 0.117), RPY (-90°, 0, -90°) |

RPY (-90°, 0, -90°) is the **optical frame** convention (z forward, x right, y down),
which is what StellaVSLAM assumes, so no extra frame is needed.

Key values in `config/tb3_d435i_rgbd.yaml` and why:

- `fx = fy = 277.13, cx = 160, cy = 120` from the simulated camera: 320 x 240,
  horizontal FOV 1.047 rad, so fx = 320 / (2 tan(1.047/2)). No distortion in simulation,
  so no calibration is needed (compare Task 2, where the real webcam had to be calibrated).
- `setup: "RGBD"`, `color_order: "RGB"` (images arrive as `rgb8`).
- `Preprocessing.depthmap_factor: 1.0` (depth already in metres). Note that stella_vslam
  only reads this key under `Preprocessing`.
- `focal_x_baseline: 13.86` (0.05 m x fx). RGB-D mode turns depth into a virtual stereo
  measurement; with 0 the constraint is lost.
- `depth_threshold: 5.0` m, the Gazebo far clip. Note: in current stella_vslam this
  value is used directly in metres (`camera/perspective.cc` passes it unchanged to
  `depth_thr_`, which `keyframe_inserter.cc` and `local_map_cleaner.cc` compare with
  point depth). Older OpenVSLAM, and parts of the online docs, describe it as a ratio
  of the baseline; that does not apply to the version built here.
- low FAST thresholds and 6 pyramid levels because the image is only 320 x 240.

## 5. Nav2 without a map

Two changes, no edits to the repo:

1. `use_localization:=False` (Nav2 Jazzy `bringup_launch.py` argument) skips `map_server`
   and AMCL. StellaVSLAM becomes the only publisher of `map → odom`. Two publishers of
   the same transform make the robot jump between them, so this is essential.
2. The global costmap has a `static_layer` that would wait forever for `/map`.
   `make_nav2_params_vslam.py` removes it and makes the global costmap a 20 m x 20 m
   rolling window around the robot, filled by the obstacle layer from `/scan`:

```text
global_costmap plugins: ['static_layer', 'obstacle_layer', 'inflation_layer'] -> ['obstacle_layer', 'inflation_layer']
rolling_window: True, size: 20 x 20 m, global_frame: map, robot_base_frame: base_link
```

Why not use the VSLAM map for planning? StellaVSLAM's map is a sparse cloud of
feature points, not an occupancy grid; it says where textured points are, not where
free space is. Obstacles therefore come from the LiDAR, and VSLAM only answers
"where am I".

## 6. How the integration works

This section follows one camera frame from Gazebo to the robot's wheels. Function and
parameter names refer to `stella_vslam_ros/src/stella_vslam_ros.cc` and the
stella_vslam library.

### 6.1 One frame, end to end

1. **Gazebo renders** the D435i's colour and depth images and `ros_gz_bridge`
   publishes them as `/camera/color/image_raw` (`rgb8`) and
   `/camera/depth/image_rect_raw` (`32FC1`, metres), stamped with simulation time.
   The colour image's `frame_id` is `camera_depth_frame` (both images come from the
   same simulated sensor).
2. **The wrapper pairs them.** In RGB-D mode `run_slam` subscribes to both topics through
   a `message_filters` **approximate-time synchroniser** (queue of 10), so it gets
   colour/depth pairs taken at (nearly) the same moment. Our remap
   `camera/depth/image_raw:=/camera/depth/image_rect_raw` connects the wrapper's
   expected depth topic to the one the bridge publishes. On the first pair it remembers
   the image `frame_id` as the camera's optical frame.
3. **`rgbd::callback`** converts both images to OpenCV (`cv_bridge`), replaces NaN depths
   (Gazebo returns NaN where there is no surface), and calls
   `slam_->feed_RGBD_frame(colour, depth, timestamp)`.
4. **Inside stella_vslam** (three threads):
   - *Tracking* (runs inside that call): extract ORB features, give each a depth from
     the depth image, predict the new camera pose from the previous motion
     (constant-velocity model), match features to map landmarks, then refine the pose by
     minimising reprojection error (pose-only optimisation). Decide whether this frame
     becomes a new keyframe.
   - *Mapping module* (background): for new keyframes, create landmarks, fuse duplicates,
     remove bad landmarks and redundant keyframes, and run local bundle adjustment
     (jointly refine nearby keyframe poses and the landmarks they see).
   - *Global optimisation module* (background): look up each new keyframe in the
     bag-of-words database; if it matches an old, non-neighbouring keyframe consistently
     and the geometry checks out, close the loop: pose-graph optimisation, then a global
     bundle adjustment ("loop bundle adjustment" in the log), then propagate the
     correction to newer keyframes.
5. **`feed_RGBD_frame` returns the camera pose** in the SLAM map, or nothing if tracking
   failed. Only a successful pose is published.
6. **`publish_pose`** publishes the pose as `nav_msgs/Odometry` on `/run_slam/camera_pose`
   and, with `publish_tf:=true`, computes and broadcasts **`map → odom`** (6.2).
7. **Nav2** looks up `map → base_link` through TF whenever it plans or controls (6.4).

### 6.2 How `map → odom` is computed

ROS frames follow REP 105: `map` (globally consistent, may jump) → `odom` (smooth,
drifts) → `base_footprint`/`base_link`. Gazebo's wheel odometry already publishes
`odom → base_footprint`, and a TF frame may only have one parent, so VSLAM cannot
publish `map → base_link` directly. Instead it publishes the **correction** between
the two:

```text
T_map_odom = T_map_camera * T_camera_odom
             (from VSLAM)    (from TF: URDF camera mount + wheel odometry,
                              looked up at the image's timestamp)
```

Details that matter:

- **Coordinate conventions.** stella_vslam uses the computer-vision convention
  (x right, y down, z forward). The wrapper converts to ROS (x forward, y left, z up)
  with a fixed rotation (`rot_ros_to_cv_map_frame_`).
- **`odom2d:=true`** projects both `T_map_camera` and `T_camera_odom` onto the floor
  plane (keeping x, y and yaw) before multiplying. Small roll/pitch errors from the
  camera then cannot tilt the `map` frame, which a 2-D planner would not handle.
- **`transform_tolerance:=0.5`** stamps the transform 0.5 s in the future, the same trick
  AMCL uses: TF lookups by Nav2 keep succeeding between camera frames. If VSLAM stops
  publishing (tracking lost), the transform becomes stale after that and Nav2 reports
  TF errors instead of driving on an outdated position.
- **Where the `map` origin is.** The SLAM map's origin is the camera pose at
  initialisation, with `map` x along the camera's viewing direction. The camera sits
  0.069 m ahead of and 0.037 m to the right of `base_footprint` (measured with
  `tf2_echo` in section 4), so the robot's start position in `map` is (-0.069, 0.037).
  Nav2's log for the first goal says `Begin navigating from current location (-0.07,
  0.04)`: exactly that offset.

### 6.3 Why the composition works: prediction and correction

The camera runs at about 10 Hz; wheel odometry and Nav2's controller run faster.
Between two camera frames, `map → odom` stays fixed and `odom → base_footprint` keeps
moving with the wheels, so `map → base_link` is propagated by odometry (prediction).
When the next frame is tracked, VSLAM publishes a new `map → odom` that absorbs whatever
drift the wheels accumulated (correction). Over a long run, the wheels can drift
without bound, but `map → base_link` stays anchored to the visual landmarks.

### 6.4 Recursive state estimation, mapped onto this system

| RSE concept | In this system |
|---|---|
| State | robot pose (and, in the SLAM back end, all keyframe poses and landmark positions) |
| Motion model / prediction | wheel odometry via TF between frames; inside tracking, a constant-velocity model to predict where landmarks will appear |
| Measurement model | projection of 3-D landmarks into the image (pinhole model with the intrinsics from section 4), plus the depth of each feature |
| Correction / update | pose-only optimisation every frame; local bundle adjustment per keyframe |
| Map | landmarks + keyframes + covisibility graph + bag-of-words database |
| Global consistency | loop detection and pose-graph optimisation + global bundle adjustment |

Two differences from the filters seen earlier in the course: stella_vslam does not keep
a Gaussian (Kalman) or particle (AMCL) belief; it computes maximum-a-posteriori
estimates by non-linear least squares over a window of keyframes (and over the whole
map after a loop closure). And there is no a-priori map: the landmarks it localises
against are the ones it created itself moments earlier, which is the "simultaneous" in
SLAM.

### 6.5 What Nav2 does with it

- **Global costmap** (`global_frame: map`, rolling 20 m window, obstacle + inflation
  layers from `/scan`): obstacles are placed in `map` using VSLAM's pose, so they stay
  where they belong after the robot moves.
- **Planner** (NavFn) plans from the robot's `map` pose to the goal on that costmap,
  and the behaviour tree replans about once per second (`Passing new path to
  controller` in the log). After a loop closure shifts `map → odom`, the next replan
  uses the corrected pose.
- **Controller** (MPPI) and the **local costmap** work in `odom`, which is smooth, so
  the jumps from loop closure never reach the wheel commands directly.

## 7. Run it

Terminal 1, simulation + Nav2 (no map):

```bash
TB_REPO=~/robotics/turtlebot-maze ~/vslam_nav2/run_sim_no_map.sh
```

Wait until Gazebo and RViz are up. Nav2 will log that it is waiting for the
`map` frame; that is expected until VSLAM starts.

Terminal 2, StellaVSLAM:

```bash
~/vslam_nav2/run_vslam.sh
```

Key parameters in `run_vslam.sh`:

| Parameter | Value | Why |
|---|---|---|
| `--net=host --ipc=host` | | same as every turtlebot-maze service; IPC is needed for Fast DDS shared memory |
| `-r camera/depth/image_raw:=/camera/depth/image_rect_raw` | | stella subscribes to `camera/depth/image_raw`; the bridge publishes `image_rect_raw` |
| `use_sim_time` | true | timestamps come from the Gazebo clock |
| `publish_tf` | true | publish `map → odom` |
| `odom2d` | true | project the pose onto the floor plane (ground robot, 2-D Nav2) |
| `camera_frame` | `camera_depth_frame` | only the `child_frame_id` of the published `~/camera_pose`. The `map → odom` computation instead looks up TF from the RGB image's `header.frame_id` to `odom` at the image timestamp; here both are `camera_depth_frame` |
| `robot_base_frame` | `base_link` | Nav2's base frame |
| `transform_tolerance` | 0.5 s | lets Nav2 use the slightly delayed VSLAM transform |

## 8. Verify before navigating

From any turtlebot-maze container (`docker exec -it $(docker ps -qf name=demo-world-enhanced) bash`,
then `source /opt/ros/jazzy/setup.bash`):

```bash
ros2 node list | grep -E "amcl|map_server" || echo "OK: no AMCL / map_server"
ros2 topic hz /run_slam/camera_pose            # VSLAM pose is being published
timeout 5 ros2 run tf2_ros tf2_echo map odom   # map -> odom exists
ros2 run tf2_tools view_frames                 # map -> odom -> base_footprint -> ... in one tree
```

Expected: no AMCL/map_server, a steady pose rate, and a single TF tree rooted at `map`.
If AMCL is still listed, `use_localization` was not passed through to Nav2; see
Troubleshooting.

### Reference run (observed)

One complete session, started from scratch (`run_sim_no_map.sh`, then `run_vslam.sh`),
VSLAM localization only, no prior map.

**Startup**

- VSLAM terminal: a single `new map created with 193 points: frame 0` about one second
  after `startup SLAM system`, and no resets afterwards. (In another session the first
  image was empty and the log showed one `preprocess: cannot extract any keypoints`
  before the map was created on frame 1; that is harmless.)
- Nav2 terminal: `Timed out waiting for transform from base_link to map` repeats until
  VSLAM publishes `map -> odom` (the global costmap cannot activate without it), then
  `global_costmap: start` and `Managed nodes are active`. No `amcl`, `map_server` or
  `lifecycle_manager_localization` appears anywhere in the log.

**Navigation.** Eight RViz goals through the maze:

| # | From (m) | To (m) | Result |
|---|---|---|---|
| 1 | (-0.07, 0.04) | (-0.62, 2.76) | succeeded |
| 2 | (-0.54, 2.67) | (-1.53, 1.88) | succeeded |
| 3 | (-1.45, 1.86) | (0.58, -0.66) | succeeded |
| 4 | (0.45, -0.75) | (2.58, -0.81) | replaced by goal 5 while driving (`Received goal preemption request`) |
| 5 | (2.59, -1.16) | (2.19, 3.18) | succeeded |
| 6 | (2.27, 3.28) | (1.86, 1.69) | succeeded |
| 7 | (2.02, 1.69) | (2.09, 3.57) | succeeded |
| 8 | (2.24, 3.40) | (0.42, 0.21), back near the start | succeeded |

7 of 7 completed goals succeeded (goal 4 was not a failure: a new goal was sent before
it finished). No `tracking lost` at any point. On goals 6-8 the collision monitor
briefly slowed the robot near walls (`Robot to approach for 1.2 seconds away from
collision`), then resumed normal operation.

**Loop closure.** Two loops were closed with the default loop-detector settings:

| Time | Loop detected | Loop bundle adjustment |
|---|---|---|
| 10:15:33 | keyframe 244 - keyframe 345 | 0.77 s |
| 10:15:45 | keyframe 265 - keyframe 370 | 0.90 s |

Each follows the same sequence in the log: `detect loop` -> `pause mapping module` ->
`start/finish loop bundle adjustment` -> `updating the map with pose propagation` ->
`updated the map`. The correction propagates to the current keyframe and therefore to
the `map -> odom` transform Nav2 uses: this is the step that makes the localization
globally consistent rather than drifting odometry. (An earlier session closed three
loops: keyframes 147-278, 379-470 and 244-536.)

Why no loop closes on short routes: the loop detector only searches keyframes outside
the current keyframe's covisibility graph, and needs the place recognised on 3
consecutive keyframes with at least 40 inlier matches. While the robot stays within
sight of the start, the start keyframes remain covisible and there is nothing to close.
Loops appeared once the routes left that area and came back.

**Saved map.** Stopping VSLAM with Ctrl+C wrote the map:

```text
encoding 456 keyframes to store
encoding 23322 landmarks to store
save the MessagePack file of database to /task3/maze_map.msg
```

In RViz add: **TF**, **Odometry** on `/run_slam/camera_pose`, **PoseArray** on
`/run_slam/keyframes_2d` (the VSLAM keyframe trajectory), and the global/local costmaps.

## 9. Demo: navigate without a map

1. Drive the robot slowly for a few metres with RViz goals close by, so VSLAM adds
   keyframes and the LiDAR fills the rolling costmap.
2. Send **Nav2 Goal**s further away, including through parts of the maze not yet seen.
3. Return to the start area: watch for a loop closure in the VSLAM log
   (`detect loop`); the `map → odom` correction jumps slightly as drift is removed.
4. Stop VSLAM with **Ctrl+C**; the landmark map is saved to `~/vslam_nav2/maze_map.msg`
   (owned by root, because it is written from inside the container).
5. Stop the simulation with **Ctrl+C**. Nav2 in Jazzy often logs a `CRITICAL FAILURE:
   SERVER controller_server IS DOWN` heartbeat error and RViz/the component container
   exit with code -6/-11 during this shutdown; that is shutdown noise, not a problem
   with the run.

Record the screen (Gazebo + RViz + the VSLAM terminal) for the demo video.

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `new map created` then `projection candidate not found` / `tracking lost` every frame | stationary simulated camera; build the image with the tracking patch (section 3) |
| `run_slam` prints nothing after start | no images: check `ros2 topic hz /camera/color/image_raw` from the VSLAM container; both containers must use `--net=host --ipc=host` and the same `ROS_DOMAIN_ID` (default 0; the VS Code dev container uses 30) |
| `Transform failed: ... camera_depth_frame` | TF not available yet or clock mismatch; make sure `use_sim_time:=true` |
| Nav2 "Timed out waiting for transform from base_link to map" | VSLAM has not initialised yet or lost tracking; move the robot a little, look at a textured wall |
| AMCL still running | `use_localization:=False` did not reach Nav2's bringup; add `use_localization` as a pass-through argument in `tb_demo_world.launch.py` (`launch_arguments` of `bringup_cmd`) |
| `CRITICAL FAILURE ... heartbeat`, rviz2 exit -6, component container exit -11 on Ctrl+C | normal when the launch is interrupted; ignore |
| Robot pose jumps | two `map → odom` publishers (AMCL still on) or a loop closure; the second is expected |
| "tracking lost" | fast turns or close to an untextured surface; slow the controller (`max_vel_theta`) or keep textured walls in view |

## 11. Limitations

- VSLAM gives localisation only; obstacle mapping is from the LiDAR rolling window, so
  the robot does not remember obstacles outside the 20 m window.
- The landmark map is in StellaVSLAM's frame: the `map` origin is wherever VSLAM
  initialised, not the Gazebo world origin.
- Tracking depends on texture; the enhanced world's textured walls are what make this work.
- Real hardware: the same pipeline applies with a real D435i (`realsense2_camera`
  publishes the same topic names), but needs calibrated intrinsics (Task 2's method) and
  `depthmap_factor` for 16-bit millimetre depth (1000.0).

## 12. Demo video

https://youtu.be/2G6iNnKbq1c

## Appendix A: Files

Save these in `~/vslam_nav2` with exactly these names.

### A.1 `Dockerfile.stella_vslam_ros_jazzy`

```dockerfile
# stella_vslam + stella_vslam_ros on ROS 2 Jazzy, no GUI viewer (visualise in RViz).
#
# Derived from the official stella_vslam_ros Dockerfile (which targets Humble):
# same apt dependencies, g2o from ros-$ROS_DISTRO-libg2o, stella_vslam built with
# CMAKE_PREFIX_PATH=/opt/ros/$ROS_DISTRO/lib/cmake. Pangolin/viewers are optional in
# stella_vslam_ros and are left out.
#
# Build (NUM_THREADS=2: more parallel C++ jobs crashed WSL out of memory):
#   docker build -t stella_vslam_ros:jazzy -f Dockerfile.stella_vslam_ros_jazzy . \
#     --build-arg NUM_THREADS=2
# If your simulation image uses another distro, also pass --build-arg ROS_DISTRO=<distro>.

ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}
ARG NUM_THREADS=2
ENV DEBIAN_FRONTEND=noninteractive DEBCONF_NOWARNINGS=yes
SHELL ["/bin/bash", "-c"]

RUN apt-get update -y -qq && \
    apt-get install -y -qq --no-install-recommends \
      build-essential cmake git pkg-config curl \
      libyaml-cpp-dev libopencv-dev libeigen3-dev libgoogle-glog-dev \
      libatlas-base-dev libsuitesparse-dev libsqlite3-dev sqlite3 \
      ros-${ROS_DISTRO}-libg2o \
      ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-image-transport \
      ros-${ROS_DISTRO}-message-filters ros-${ROS_DISTRO}-tf2-eigen \
      ros-${ROS_DISTRO}-tf2-geometry-msgs ros-${ROS_DISTRO}-rosbag2-cpp \
      ros-${ROS_DISTRO}-tf2-ros ros-${ROS_DISTRO}-tf2-tools \
      python3-colcon-common-extensions && \
    rm -rf /var/lib/apt/lists/*

# stella_vslam core library
# Simulation fix: tracking_module::search_local_landmarks() returns false when every
# visible landmark is already matched ("projection candidate not found"). With a robot
# standing still in Gazebo, consecutive frames are identical, so tracking is lost right
# after initialisation and every time the robot stops. Continue instead: the following
# pose optimisation still requires >= 20 tracked landmarks.
RUN git clone --depth 1 https://github.com/stella-cv/stella_vslam.git /opt/stella_vslam && \
    cd /opt/stella_vslam && git submodule update --init --recursive && \
    perl -0pi -e 's/spdlog::warn\("projection candidate not found"\);\s*return false;/spdlog::debug("projection candidate not found (all visible landmarks already matched)");\n        return true;/' src/stella_vslam/tracking_module.cc && \
    grep -q "all visible landmarks already matched" src/stella_vslam/tracking_module.cc && \
    mkdir -p build && cd build && \
    CMAKE_PREFIX_PATH=/opt/ros/${ROS_DISTRO}/lib/cmake cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .. && \
    make -j${NUM_THREADS} && make install && ldconfig

# stella_vslam_ros (ROS 2 branch)
# Jazzy fix: ament_export_targets(... HAS_LIBRARY_TARGET) already registers the
# library_path environment hook, so the extra ament_environment_hooks() block in
# stella_vslam_ros' CMakeLists.txt registers it twice and CMake fails with
# "Files to be generated by multiple different commands: .../library_path.dsv".
# Delete that block (the only `if(NOT WIN32) ... endif()` in the file).
WORKDIR /ros2_ws
RUN git clone --recursive -b ros2 --depth 1 https://github.com/stella-cv/stella_vslam_ros.git src/stella_vslam_ros && \
    sed -i '/^if(NOT WIN32)/,/^endif()/d' src/stella_vslam_ros/CMakeLists.txt && \
    ! grep -q ament_environment_hooks src/stella_vslam_ros/CMakeLists.txt && \
    : "Jazzy fix 2: the old .h ROS headers are gone (cv_bridge) or deprecated (tf2_ros); use .hpp" && \
    sed -i 's#cv_bridge/cv_bridge\.h>#cv_bridge/cv_bridge.hpp>#; s#tf2_ros/\(transform_listener\|transform_broadcaster\|buffer\)\.h>#tf2_ros/\1.hpp>#; s#geometry_msgs/msg/transform_stamped\.h>#geometry_msgs/msg/transform_stamped.hpp>#' \
      src/stella_vslam_ros/src/*.cc src/stella_vslam_ros/src/*.h && \
    ! grep -rq "cv_bridge/cv_bridge.h>" src/stella_vslam_ros/src && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    MAKEFLAGS=-j${NUM_THREADS} colcon build --parallel-workers 1 \
      --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo

# ORB vocabulary
RUN mkdir -p /vslam && \
    curl -sL "https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow" -o /vslam/orb_vocab.fbow && \
    test -s /vslam/orb_vocab.fbow

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc
CMD ["bash"]
```

### A.2 `config/tb3_d435i_rgbd.yaml`

```yaml
# stella_vslam RGB-D config for the TurtleBot3 Waffle's simulated RealSense D435i
# (turtlebot-maze repo, demo-world-enhanced), used through stella_vslam_ros.
#
# Intrinsics: same as the repo's slam/config/turtlebot_realsense.yaml, derived from
# gz_waffle.sdf.xacro (horizontal_fov = 1.047 rad, 320 x 240):
#   fx = fy = 320 / (2 tan(1.047 / 2)) = 277.13,  cx = 160,  cy = 120
# Differences from that file (checked against the stella_vslam source):
#   - color_order RGB: stella_vslam_ros passes the ROS image through unchanged
#     (encoding parameter empty), and the Gazebo bridge publishes rgb8.
#   - focal_x_baseline set: RGB-D mode builds a virtual stereo pair from depth;
#     0 (the default when the key is missing) disables that constraint.
#     0.05 m (roughly the D435 baseline) x fx = 13.86.
#   - depthmap_factor moved to Preprocessing (stella_vslam only reads it there).
#     Gazebo depth is 32FC1 in metres, so 1.0.
#   - depth_threshold is in metres in current stella_vslam; 5.0 = Gazebo far clip.
#   - Loop closure enabled, map saved as msgpack.

Camera:
  name: "TurtleBot3 Waffle simulated D435i (RGBD, ROS 2)"
  setup: "RGBD"
  model: "perspective"

  fx: 277.13
  fy: 277.13
  cx: 160.0
  cy: 120.0

  k1: 0.0
  k2: 0.0
  p1: 0.0
  p2: 0.0
  k3: 0.0

  fps: 10.0
  cols: 320
  rows: 240

  focal_x_baseline: 13.86
  depth_threshold: 5.0

  color_order: "RGB"

Preprocessing:
  min_size: 200
  depthmap_factor: 1.0

Feature:
  name: "ORB for low-res simulation camera"
  scale_factor: 1.2
  num_levels: 6
  ini_fast_threshold: 10
  min_fast_threshold: 3

Mapping:
  baseline_dist_thr: 0.1
  redundant_obs_ratio_thr: 0.9

LoopDetector:
  enabled: true

System:
  map_format: "msgpack"
```

### A.3 `make_nav2_params_vslam.py`

```python
#!/usr/bin/env python3
"""Turn a Nav2 params file that expects a prior map into one that works without it.

Changes only the global costmap:
  - removes the static_layer (it would wait forever for a /map that never comes)
  - makes the costmap a rolling window centred on the robot, built from live
    sensor data (obstacle/voxel + inflation layers, which stay as configured)
Everything else (controller, planner, behaviours, local costmap) is untouched.
Localization (map -> odom) comes from stella_vslam_ros instead of AMCL.

    python3 make_nav2_params_vslam.py nav2_params.yaml nav2_params_vslam.yaml
"""
import argparse

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--size", type=float, default=20.0,
                    help="rolling global costmap width/height in metres")
    a = ap.parse_args()

    with open(a.src) as f:
        params = yaml.safe_load(f)

    g = params["global_costmap"]["global_costmap"]["ros__parameters"]
    before = list(g.get("plugins", []))
    g["plugins"] = [p for p in before if p != "static_layer"]
    g.pop("static_layer", None)
    g["rolling_window"] = True
    g["width"] = int(a.size)
    g["height"] = int(a.size)
    g["track_unknown_space"] = True

    with open(a.dst, "w") as f:
        f.write("# Generated by make_nav2_params_vslam.py from " + a.src + "\n"
                "# global_costmap: static_layer removed, rolling_window enabled.\n")
        yaml.safe_dump(params, f, sort_keys=False)

    print(f"global_costmap plugins: {before} -> {g['plugins']}")
    print(f"rolling_window: True, size: {g['width']} x {g['height']} m, "
          f"global_frame: {g.get('global_frame')}, robot_base_frame: {g.get('robot_base_frame')}")
    print("wrote", a.dst)


if __name__ == "__main__":
    main()
```

### A.4 `run_sim_no_map.sh`

```bash
#!/usr/bin/env bash
# Start the turtlebot-maze enhanced world (textured walls + ArUco markers) with Nav2
# but WITHOUT a prior map: no map_server, no AMCL (use_localization:=False), and a
# global costmap that is a rolling window built from the LiDAR.
#
# Usage:  TB_REPO=~/robotics/turtlebot-maze ./run_sim_no_map.sh
set -euo pipefail
T3="$(cd "$(dirname "$0")" && pwd)"
TB_REPO="${TB_REPO:-$HOME/robotics/turtlebot-maze}"
SHARE=/overlay_ws/install/tb_worlds/share/tb_worlds
cd "$TB_REPO"

# 1. derive the no-map Nav2 params from the repo's own nav2_params.yaml
docker compose run --rm -v "$T3":/task3 demo-world-enhanced \
  python3 /task3/make_nav2_params_vslam.py "$SHARE/configs/nav2_params.yaml" /task3/nav2_params_vslam.yaml

# 2. world + robot + RViz + Nav2 (navigation only, no localization)
docker compose run --rm -v "$T3":/task3 demo-world-enhanced \
  ros2 launch tb_worlds tb_demo_world.launch.py \
    world_name:=sim_house_enhanced.sdf.xacro \
    use_aruco:=True \
    rviz_config_file:="$SHARE/rviz/nav2_turtlebot_view.rviz" \
    use_localization:=False \
    params_file:=/task3/nav2_params_vslam.yaml
```

### A.5 `run_vslam.sh`

```bash
#!/usr/bin/env bash
# Run stella_vslam_ros (RGB-D) next to the turtlebot-maze simulation and publish
# map -> odom for Nav2. Host network + host IPC, like every turtlebot-maze service,
# so DDS discovery and Fast DDS shared-memory transport both work.
#
# Usage:  ./run_vslam.sh            (map saved to task3/maze_map.msg on Ctrl+C)
set -euo pipefail
T3="$(cd "$(dirname "$0")" && pwd)"

docker run -it --rm --net=host --ipc=host \
  -v "$T3":/task3 \
  stella_vslam_ros:jazzy \
  bash -c "source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash && \
    ros2 run stella_vslam_ros run_slam \
      -v /vslam/orb_vocab.fbow \
      -c /task3/config/tb3_d435i_rgbd.yaml \
      --viewer none \
      --map-db-out /task3/maze_map.msg \
      --ros-args \
        -r camera/depth/image_raw:=/camera/depth/image_rect_raw \
        -p use_sim_time:=true \
        -p map_frame:=map \
        -p odom_frame:=odom \
        -p robot_base_frame:=base_link \
        -p camera_frame:=camera_depth_frame \
        -p publish_tf:=true \
        -p odom2d:=true \
        -p publish_keyframes:=true \
        -p transform_tolerance:=0.5"
```
*Used AI for this documentation 
