#!/bin/bash
# 使い方: [MODEL=cnn] ./shoot.sh <label> [秒数] ["x,y,yaw"]   （MODEL 省略時は ACT v4）
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
pkill -f "[d]ynamic_pose/info" 2>/dev/null; sleep 1
LABEL=$1; SEC=${2:-80}
[ -n "$3" ] && export EVAL_START="$3"
./set_view.sh A
sleep 1
if [ "$MODEL" = "cnn" ]; then
  export DRIVE_NODE=drive_node            # driving_model_v2.pth (= CNN r3)
else
  export DRIVE_NODE=act_drive_node ACT_MODEL=~/ros2_ws/act_model_v4.pth ACT_TE=1 ACT_TE_ORDER=new ACT_TE_M=0.5
fi
python3 eval_run.py $SEC $LABEL 2>&1 | tee logs/eval_$LABEL.out
