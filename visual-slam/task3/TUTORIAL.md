# Tutorial: Navigating a TurtleBot with StellaVSLAM + Nav2, without a prior map

**Stack:** ROS 2 Jazzy · Gazebo Harmonic · Nav2 · stella_vslam_ros (RGB-D) · Docker on WSL2
**Simulation:** [`turtlebot-maze`](#) repo, `demo-world-enhanced` (textured walls, ArUco markers),
TurtleBot3 Waffle with a simulated Intel RealSense D435i.

> Status: commands and configs were checked against the stella_vslam / stella_vslam_ros /
> Nav2 Jazzy sources and against the live topics and TF of the simulation. Items marked
> **[verify]** depend on runtime behaviour; check them with the commands given.

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

**Recursive state estimation view.** The robot state (pose) is estimated recursively:
wheel odometry *predicts* motion (`odom → base_footprint`, smooth but drifting), and
visual landmarks *correct* it. StellaVSLAM tracks ORB features against its landmark
map (pose optimisation), refines recent keyframes and landmarks jointly (local bundle
adjustment), and when it recognises a place seen before (bag-of-words loop detection)
it runs pose-graph optimisation so the whole trajectory becomes globally consistent.
The correction is published as `map → odom`, so Nav2 sees one consistent pose
`map → odom → base_footprint` without ever being given a map.

**Why RGB-D and not monocular.** A single camera cannot observe metric scale
(Task 1 showed the drift this causes). Nav2 plans in metres, so a scale-ambiguous pose
is unusable. The simulated D435i provides depth in metres, so every landmark has
metric depth from the first frame and the map is in metres.

## 2. Prerequisites

- WSL2 + Docker, and the `turtlebot-maze` repo built (`docker compose build`) with
  `ROS_DISTRO=jazzy` in its `.env`.
- This folder (`assignment-3/task3/`):

| File | Purpose |
|---|---|
| `Dockerfile.stella_vslam_ros_jazzy` | image with stella_vslam + stella_vslam_ros on ROS 2 Jazzy (no GUI viewer; use RViz) |
| `config/tb3_d435i_rgbd.yaml` | StellaVSLAM camera config for the simulated D435i |
| `make_nav2_params_vslam.py` | turns the repo's `nav2_params.yaml` into a no-map version |
| `run_sim_no_map.sh` | world + robot + RViz + Nav2 without map_server/AMCL |
| `run_vslam.sh` | starts StellaVSLAM and publishes `map → odom` |

## 3. Build the StellaVSLAM image (once, 30-60 min)

```bash
cd ~/visual-slam/task3        # this folder
docker build -t stella_vslam_ros:jazzy -f Dockerfile.stella_vslam_ros_jazzy . \
  --build-arg NUM_THREADS=2
```

`NUM_THREADS=2` matters on WSL: each parallel C++ compile job can use 1-2 GB and more
jobs crash the WSL VM. The image follows the official stella_vslam_ros Dockerfile
(g2o from `ros-jazzy-libg2o`, OpenCV/Eigen from apt) but leaves out the Pangolin
viewer; RViz is the viewer.

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
- `depth_threshold: 5.0` m, the Gazebo far clip.
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

## 6. Run it

Terminal 1, simulation + Nav2 (no map):

```bash
TB_REPO=~/robotics/turtlebot-maze ~/visual-slam/task3/run_sim_no_map.sh
```

Wait until Gazebo and RViz are up. Nav2 will log that it is waiting for the
`map` frame; that is expected until VSLAM starts.

Terminal 2, StellaVSLAM:

```bash
~/visual-slam/task3/run_vslam.sh
```

Key parameters in `run_vslam.sh`:

| Parameter | Value | Why |
|---|---|---|
| `--net=host --ipc=host` | | same as every turtlebot-maze service; IPC is needed for Fast DDS shared memory |
| `-r camera/depth/image_raw:=/camera/depth/image_rect_raw` | | stella subscribes to `camera/depth/image_raw`; the bridge publishes `image_rect_raw` |
| `use_sim_time` | true | timestamps come from the Gazebo clock |
| `publish_tf` | true | publish `map → odom` |
| `odom2d` | true | project the pose onto the floor plane (ground robot, 2-D Nav2) |
| `camera_frame` | `camera_depth_frame` | the image frame from section 4 |
| `robot_base_frame` | `base_link` | Nav2's base frame |
| `transform_tolerance` | 0.5 s | lets Nav2 use the slightly delayed VSLAM transform |

## 7. Verify before navigating

From any turtlebot-maze container (`docker exec -it $(docker ps -qf name=demo-world-enhanced) bash`,
then `source /opt/ros/jazzy/setup.bash`):

```bash
ros2 node list | grep -E "amcl|map_server" || echo "OK: no AMCL / map_server"
ros2 topic hz /run_slam/camera_pose            # VSLAM pose is being published
timeout 5 ros2 run tf2_ros tf2_echo map odom   # map -> odom exists
ros2 run tf2_tools view_frames                 # map -> odom -> base_footprint -> ... in one tree
```

**[verify]** Expected: no AMCL/map_server, a steady pose rate, and a single TF tree
rooted at `map`. If AMCL is still listed, `use_localization` was not passed through to
Nav2; see Troubleshooting.

In RViz add: **TF**, **Odometry** on `/run_slam/camera_pose`, **PoseArray** on
`/run_slam/keyframes_2d` (the VSLAM keyframe trajectory), and the global/local costmaps.

## 8. Demo: navigate without a map

1. Drive the robot slowly for a few metres with RViz goals close by, so VSLAM adds
   keyframes and the LiDAR fills the rolling costmap.
2. Send **Nav2 Goal**s further away, including through parts of the maze not yet seen.
3. Return to the start area: watch for a loop closure in the VSLAM log
   (`detect loop`); the `map → odom` correction jumps slightly as drift is removed.
4. Stop VSLAM with **Ctrl+C**; the landmark map is saved to `task3/maze_map.msg`.

Record the screen (Gazebo + RViz + the VSLAM terminal) for the demo video.

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `run_slam` prints nothing after start | no images: check `ros2 topic hz /camera/color/image_raw` from the VSLAM container; both containers must use `--net=host --ipc=host` and the same `ROS_DOMAIN_ID` (default 0; the VS Code dev container uses 30) |
| `Transform failed: ... camera_depth_frame` | TF not available yet or clock mismatch; make sure `use_sim_time:=true` |
| Nav2 "Timed out waiting for transform from base_link to map" | VSLAM has not initialised yet or lost tracking; move the robot a little, look at a textured wall |
| AMCL still running | `use_localization:=False` did not reach Nav2's bringup; add `use_localization` as a pass-through argument in `tb_demo_world.launch.py` (`launch_arguments` of `bringup_cmd`) |
| Robot pose jumps | two `map → odom` publishers (AMCL still on) or a loop closure; the second is expected |
| "tracking lost" | fast turns or close to an untextured surface; slow the controller (`max_vel_theta`) or keep textured walls in view |

## 10. Limitations

- VSLAM gives localisation only; obstacle mapping is from the LiDAR rolling window, so
  the robot does not remember obstacles outside the 20 m window.
- The landmark map is in StellaVSLAM's frame: the `map` origin is wherever VSLAM
  initialised, not the Gazebo world origin.
- Tracking depends on texture; the enhanced world's textured walls are what make this work.
- Real hardware: the same pipeline applies with a real D435i (`realsense2_camera`
  publishes the same topic names), but needs calibrated intrinsics (Task 2's method) and
  `depthmap_factor` for 16-bit millimetre depth (1000.0).

## 11. Demo video

Link: TODO
