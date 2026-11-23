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
