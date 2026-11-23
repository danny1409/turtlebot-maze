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
