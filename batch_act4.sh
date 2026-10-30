#!/bin/bash
# v4: v3 のデータ + 復帰デモ(dataset_recovery) で学習 → 案A設定で10回評価
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
rm -f logs/batch_act4_done
ACT_AUG=1 ACT_FREEZE=layer2 ACT_EPOCHS=15 ACT_MODEL=act_model_v4.pth python3 src/nav_ai/nav_ai/train_act.py dataset_act dataset_recovery dataset dataset.04 > train_act_v4.log 2>&1
export ACT_MODEL=~/ros2_ws/act_model_v4.pth DRIVE_NODE=act_drive_node
for k in $(seq 1 10); do ACT_TE=1 ACT_TE_ORDER=new ACT_TE_M=0.5 python3 eval_run.py 80 act4A_$k 2>&1 | grep -v "^\[" > logs/eval_act4A_$k.out; done
touch logs/batch_act4_done
