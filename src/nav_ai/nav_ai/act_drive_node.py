"""
簡易 ACT で走る推論ノード

- 毎フレーム CHUNK 個先までの行動（3クラス確率＋前進速度）を予測
- Temporal ensembling: 時刻 t の行動 = 過去の各予測チャンクが「時刻 t」について出した確率の重み付き平均
  重み w_i = exp(-m * i)（i=0 が最も古い予測。元論文 ACT と同じ向き）。ACT_TE=0 で無効（毎回 step0 のみ）
- 旋回量・安全オーバーライド・スタック検出・ログ形式は drive_node（CNN）と同一にして公平に比較する
環境変数: ACT_MODEL（既定 ~/ros2_ws/act_model.pth） ACT_TE ACT_TE_M DRIVE_* は drive_node と共通
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
    from nav_ai import drive_node as dn
    from nav_ai.train_act import ACTPolicy
except ImportError:
    import drive_node as dn
    from train_act import ACTPolicy

USE_TE = os.environ.get("ACT_TE", "1") == "1"
TE_M = float(os.environ.get("ACT_TE_M", "0.1"))
# old: 元論文どおり古い予測ほど重い / new: 新しい予測ほど重い（反応の遅れを減らす）
TE_ORDER = os.environ.get("ACT_TE_ORDER", "old")
# 確率の温度（<1 で鋭くする。1=そのまま）。p ∝ p^(1/T)
TEMP = float(os.environ.get("ACT_TEMP", "1.0"))


class ActDriveNode(Node):
    def __init__(self):
        super().__init__("act_drive_node")
        model_path = os.path.expanduser(os.environ.get("ACT_MODEL", "~/ros2_ws/act_model.pth"))
        with open(os.path.splitext(model_path)[0] + "_config.json") as f:
            self.cfg = c = json.load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ACTPolicy(chunk=c["chunk"], d=c["d_model"], n_head=c["n_head"], n_enc=c["n_enc"],
                               n_dec=c["n_dec"], use_cvae=c["use_cvae"], latent=c["latent"],
                               pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.K = c["chunk"]
        self.mean = np.array(c["mean"], np.float32).reshape(3, 1, 1)
        self.std = np.array(c["std"], np.float32).reshape(3, 1, 1)
        self.cw = np.array(c["class_weight"], np.float32)

        self.buf = deque(maxlen=self.K)     # 過去の予測チャンク (probs[K,3], lin[K])、新しいものが右
        self.br = CvBridge()
        self.ang_s = 0.0
        self.small_hist = deque(maxlen=dn.STUCK_GAP + 1)
        self.still_count = 0
        self.backup_left = 0
        self.headless = os.environ.get("DRIVE_HEADLESS", "0") == "1"

        log_path = os.environ.get("DRIVE_LOG") or os.path.expanduser(
            time.strftime("~/ros2_ws/logs/act_drive_%Y%m%d_%H%M%S.csv"))
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_f = open(log_path, "w", newline="")
        self.log = csv.writer(self.log_f)
        self.log.writerow(["t", "p_left", "p_straight", "p_right", "danger", "cx",
                           "mode", "ang_raw", "ang", "lin"])
        self.get_logger().info(f"ACT モデル: {model_path}  TE={USE_TE} order={TE_ORDER} m={TE_M} temp={TEMP}  ログ: {log_path}")

        self.sub = self.create_subscription(Image, "/camera/image_raw", self.cb, 10)
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)

    danger = dn.DriveNode.danger

    def preprocess(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.cfg["img_w"], self.cfg["img_h"]), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        return (x - self.mean) / self.std

    def predict(self, bgr):
        x = torch.from_numpy(self.preprocess(bgr)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit, lin, _, _ = self.model(x)
        probs = torch.softmax(logit[0], -1).cpu().numpy()     # [K,3]
        lin = lin[0].cpu().numpy()                             # [K]
        if not USE_TE:
            return probs[0], float(lin[0])
        self.buf.append((probs, lin))
        n = len(self.buf)
        # buf[i] は (n-1-i) フレーム前の予測 → 今の行動はそのチャンクの index (n-1-i)
        ps = np.stack([self.buf[i][0][n - 1 - i] for i in range(n)])
        ls = np.array([self.buf[i][1][n - 1 - i] for i in range(n)])
        age = np.arange(n)[::-1] if TE_ORDER == "new" else np.arange(n)
        w = np.exp(-TE_M * age)               # old: i=0（最古）が最大 / new: 最新が最大
        w /= w.sum()
        return (w[:, None] * ps).sum(0), float((w * ls).sum())

    def cb(self, msg):
        bgr = self.br.imgmsg_to_cv2(msg, "bgr8")
        twist = Twist()
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # --- スタック検出（drive_node と同じ） ---
        small = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (40, 30)).astype(np.float32)
        self.small_hist.append(small)
        if len(self.small_hist) > dn.STUCK_GAP and np.abs(small - self.small_hist[0]).mean() < dn.STUCK_DIFF:
            self.still_count += 1
        else:
            self.still_count = 0
        if self.still_count > dn.STUCK_FRAMES and self.backup_left == 0:
            self.backup_left = dn.BACKUP_FRAMES
            self.get_logger().warn("スタック検出 → 後退して切り返し")
        if self.backup_left > 0:
            self.backup_left -= 1
            twist.linear.x = -0.3
            self.pub.publish(twist)
            self.buf.clear()
            self.small_hist.clear()
            self.still_count = 0
            self.log.writerow([f"{t:.2f}", "", "", "", "", "", "backup", "", "0.000", "-0.30"])
            return

        # --- ACT 推論 ---
        p, lin = self.predict(bgr)
        if dn.PRIOR_CORR:
            p = p / self.cw
            p = p / p.sum()
        if TEMP != 1.0:
            p = np.power(np.clip(p, 1e-8, 1), 1.0 / TEMP)
            p = p / p.sum()
        ang_raw = dn.ANG_MAX * (p[0] - p[2])
        lin_x = float(np.clip(lin, dn.LIN_MIN, dn.LIN_MAX))
        mode = "model"

        # --- 安全オーバーライド（drive_node と同じ） ---
        ratio, cx = self.danger(bgr)
        if dn.USE_OVERRIDE and ratio > dn.DANGER_AREA and dn.DANGER_CX[0] < cx < dn.DANGER_CX[1]:
            pref = p[0] - p[2]
            if abs(pref) > 0.1:
                side = 1.0 if pref > 0 else -1.0
            else:
                side = -1.0 if cx < 0.5 else 1.0
            ang_raw = side * dn.ANG_MAX
            lin_x = dn.LIN_MIN
            mode = "override"

        self.ang_s = dn.EMA_ALPHA * ang_raw + (1 - dn.EMA_ALPHA) * self.ang_s
        twist.linear.x = lin_x
        twist.angular.z = float(self.ang_s)
        self.pub.publish(twist)
        self.log.writerow([f"{t:.2f}", f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}",
                           f"{ratio:.3f}", f"{cx:.2f}", mode, f"{ang_raw:.3f}",
                           f"{self.ang_s:.3f}", f"{lin_x:.2f}"])
        self.log_f.flush()

        if not self.headless:
            vis = bgr.copy()
            cv2.putText(vis, f"ACT L{p[0]:.2f} S{p[1]:.2f} R{p[2]:.2f} ang{self.ang_s:+.2f} {mode}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("ACT Drive", vis)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = ActDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.log_f.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
