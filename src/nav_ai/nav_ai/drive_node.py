"""
学習済み DrivingCNN で走る推論ノード（安全オーバーライド付き）

- 学習時と同じ前処理（120x160・正規化・履歴3フレーム）を必ず適用
- 予測クラスの確率から旋回量を決め、指数移動平均でならして首振りを抑える
- 安全策: 画面下・中央寄りに球が大きく写ったら旋回を強める。
  向きは「モデルが行きたい側」に合わせる（球の位置だけで決めるとスラロームの進路と逆になるため）
- スタック検出: 画像がほぼ変化しない状態が続いたら一定時間後退して切り返す
- 毎フレームの判断を CSV に記録（DRIVE_LOG で保存先を指定可）
- DRIVE_HEADLESS=1 で画面表示なし（評価スクリプトから起動する場合）
"""
import csv
import json
import os
import time
from collections import deque

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from nav_ai.train_driving import DrivingCNN   # nav_ai パッケージ内に置いた場合
except ImportError:
    from train_driving import DrivingCNN          # 同じフォルダから直接実行した場合

ANG_MAX = float(os.environ.get("DRIVE_ANG_MAX", "0.5"))      # 旋回の最大角速度 [rad/s]
LIN_MIN, LIN_MAX = 0.2, 0.6
EMA_ALPHA = float(os.environ.get("DRIVE_EMA", "1.0"))       # 旋回量の平滑化（1.0=平滑化なし）
# 学習時のクラス重みの影響を打ち消す（事前確率補正）。1=有効
PRIOR_CORR = os.environ.get("DRIVE_PRIOR_CORR", "1") == "1"
USE_OVERRIDE = os.environ.get("DRIVE_OVERRIDE", "1") == "1"
DANGER_AREA = 0.08       # 危険領域内で球の色が占める割合のしきい値
DANGER_CX = (0.25, 0.75) # 球の重心がこの範囲（画面中央寄り）のときだけオーバーライド
STUCK_GAP = 10           # 何フレーム前の画像と比べるか（10Hzで1秒前）
STUCK_DIFF = 0.5         # 1秒前との平均輝度差がこれ未満なら「動いていない」
STUCK_FRAMES = 10        # 「動いていない」がこのフレーム数続いたらスタック判定
BACKUP_FRAMES = 12       # 後退するフレーム数

# HSV での球の色範囲（赤は色相が0付近と180付近の2つに分かれる）
BALL_HSV = [
    ((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255)),   # 赤
    ((40, 120, 70), (80, 255, 255)),                                    # 緑
    ((100, 150, 50), (130, 255, 255)),                                  # 青
]


class DriveNode(Node):
    def __init__(self):
        super().__init__("drive_node")
        model_path = os.path.expanduser("~/ros2_ws/driving_model_v2.pth")
        with open(os.path.splitext(model_path)[0] + "_config.json") as f:
            self.cfg = json.load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DrivingCNN(history_len=self.cfg["history_len"]).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()   # ★BatchNorm/Dropout を推論モードに

        self.mean = np.array(self.cfg["mean"], np.float32).reshape(3, 1, 1)
        self.std = np.array(self.cfg["std"], np.float32).reshape(3, 1, 1)
        self.hist = deque(maxlen=self.cfg["history_len"])
        self.br = CvBridge()
        self.ang_s = 0.0
        self.small_hist = deque(maxlen=STUCK_GAP + 1)
        self.still_count = 0
        self.backup_left = 0
        self.headless = os.environ.get("DRIVE_HEADLESS", "0") == "1"

        log_path = os.environ.get("DRIVE_LOG") or os.path.expanduser(
            time.strftime("~/ros2_ws/logs/drive_%Y%m%d_%H%M%S.csv"))
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_f = open(log_path, "w", newline="")
        self.log = csv.writer(self.log_f)
        self.log.writerow(["t", "p_left", "p_straight", "p_right", "danger", "cx",
                           "mode", "ang_raw", "ang", "lin"])
        self.get_logger().info(f"ログ: {log_path}")

        self.sub = self.create_subscription(Image, "/camera/image_raw", self.cb, 10)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def preprocess(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.cfg["img_w"], self.cfg["img_h"]), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        return (x - self.mean) / self.std

    def danger(self, bgr):
        """画面下半分・中央60%の領域で球の色が占める割合と、その重心が左右どちらか"""
        h, w = bgr.shape[:2]
        roi = bgr[h // 2:, int(w * 0.2):int(w * 0.8)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in BALL_HSV:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        ratio = mask.mean() / 255.0
        if ratio == 0:
            return 0.0, 0.5
        cx = np.nonzero(mask)[1].mean() / mask.shape[1]   # 0=左端, 1=右端
        return ratio, cx

    def stamp(self, msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def cb(self, msg):
        bgr = self.br.imgmsg_to_cv2(msg, "bgr8")
        twist = Twist()
        t = self.stamp(msg)

        # --- スタック検出（画像がほとんど変わらない） ---
        # 単色の床と壁では1フレームごとの差がもともと小さいので、1秒前の画像と比べる
        small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (40, 30)).astype(np.float32)
        self.small_hist.append(small)
        if len(self.small_hist) > STUCK_GAP and np.abs(small - self.small_hist[0]).mean() < STUCK_DIFF:
            self.still_count += 1
        else:
            self.still_count = 0
        if self.still_count > STUCK_FRAMES and self.backup_left == 0:
            self.backup_left = BACKUP_FRAMES
            self.get_logger().warn("スタック検出 → 後退して切り返し")
        if self.backup_left > 0:
            self.backup_left -= 1
            twist.linear.x = -0.3
            twist.angular.z = 0.0
            self.pub.publish(twist)
            self.hist.clear()
            self.small_hist.clear()
            self.still_count = 0
            self.log.writerow([f"{t:.2f}", "", "", "", "", "", "backup", "",
                               f"{twist.angular.z:.3f}", f"{twist.linear.x:.2f}"])
            return

        # --- 学習モデルの推論 ---
        self.hist.append(self.preprocess(bgr))
        while len(self.hist) < self.hist.maxlen:
            self.hist.append(self.hist[-1])
        x = torch.from_numpy(np.concatenate(list(self.hist), 0)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit, lin = self.model(x)
            p = torch.softmax(logit, 1)[0].cpu().numpy()   # [left, straight, right]
        if PRIOR_CORR and "class_weight" in self.cfg:
            p = p / np.array(self.cfg["class_weight"])     # 重み付けで盛った分を戻す
            p = p / p.sum()
        ang_raw = ANG_MAX * (p[0] - p[2])                   # 確率の差を連続値に
        lin_x = float(np.clip(lin.item(), LIN_MIN, LIN_MAX))
        mode = "model"

        # --- 安全オーバーライド（中央寄りに球が大きく写ったら旋回を強める） ---
        ratio, cx = self.danger(bgr)
        if USE_OVERRIDE and ratio > DANGER_AREA and DANGER_CX[0] < cx < DANGER_CX[1]:
            pref = p[0] - p[2]
            if abs(pref) > 0.1:
                side = 1.0 if pref > 0 else -1.0          # モデルが行きたい側
            else:
                side = -1.0 if cx < 0.5 else 1.0          # 迷っているときだけ球の反対側
            ang_raw = side * ANG_MAX
            lin_x = LIN_MIN
            mode = "override"

        self.ang_s = EMA_ALPHA * ang_raw + (1 - EMA_ALPHA) * self.ang_s
        twist.linear.x = lin_x
        twist.angular.z = float(self.ang_s)
        self.pub.publish(twist)
        self.log.writerow([f"{t:.2f}", f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}",
                           f"{ratio:.3f}", f"{cx:.2f}", mode, f"{ang_raw:.3f}",
                           f"{self.ang_s:.3f}", f"{lin_x:.2f}"])
        self.log_f.flush()

        if not self.headless:
            vis = bgr.copy()
            cv2.putText(vis, f"L{p[0]:.2f} S{p[1]:.2f} R{p[2]:.2f} ang{self.ang_s:+.2f} {mode}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Drive", vis)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = DriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())   # 終了時に停止
        node.log_f.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
