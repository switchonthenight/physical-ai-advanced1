#!/bin/bash
# ずれた姿勢からの復帰評価: 固定15姿勢(logs/recovery_starts.csv) × CNN r3 / ACT v3 / ACT v4、各60秒
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
rm -f logs/batch_rec_done
run() {  # $1=ラベル接頭辞  残りは環境変数
  tail -n +2 logs/recovery_starts.csv | while IFS=, read id name x y yaw; do
    EVAL_START="$x,$y,$yaw" python3 eval_run.py 60 ${1}_$id 2>&1 | grep -v "^\[" > logs/eval_${1}_$id.out
  done
}
DRIVE_NODE=drive_node run recCNN
export DRIVE_NODE=act_drive_node ACT_TE=1 ACT_TE_ORDER=new ACT_TE_M=0.5
ACT_MODEL=~/ros2_ws/act_model_v3.pth run recV3
ACT_MODEL=~/ros2_ws/act_model_v4.pth run recV4
touch logs/batch_rec_done
