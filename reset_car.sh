#!/bin/bash
# 使い方: ./reset_car.sh [x y yaw_deg]   引数なし = 初期位置
X=${1:-0.0}; Y=${2:-0.0}; YAW=${3:-0}; Z=0.1   # ← 1. で確認した値に書き換える
read QZ QW < <(awk -v d="$YAW" 'BEGIN{r=d*3.14159265/180/2; printf "%.6f %.6f", sin(r), cos(r)}')
# 念のため停止指令を送る
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{}" >/dev/null 2>&1
ign service -s /world/car_world/set_pose \
  --reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean --timeout 2000 \
  --req "name: \"vehicle_blue\", position: {x: $X, y: $Y, z: $Z}, orientation: {z: $QZ, w: $QW}"