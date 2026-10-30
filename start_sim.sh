#!/bin/bash
# シミュレーション起動（teleop_node なし）。USE_TELEOP=true で従来どおり
export DISPLAY=:0
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
exec ros2 launch nav_dev nav_teleop.launch.py use_teleop:=${USE_TELEOP:-false} > /tmp/launch_$(whoami).log 2>&1
