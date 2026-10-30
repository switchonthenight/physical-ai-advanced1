"""
ACT 用のエピソード収集ノード

1エピソード = スタートからゴールまでの連続した走行。
- 車を初期位置に戻した後、最初に操作した瞬間から録画を開始
- 車体が x > GOAL_X に達したら自動で保存し、初期位置に戻して次のエピソードを待つ
- このノードのターミナルでキー操作:
    d : 今のエピソードを破棄して初期位置に戻す（失敗したとき）
    r : 録画していないときに初期位置に戻す
    q : 終了
保存先: ~/ros2_ws/dataset_act/（環境変数 ACT_DATASET_DIR で変更可）
  images/ep003_00012.jpg
  data.csv : episode, frame, image_name, linear_x, angular_z, sim_time
"""
import csv
import math
import os
import random
import select
import shutil
import signal
import subprocess
import sys
import termios
import tty

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage

GOAL_X = 13.0
MAX_SEC = 150.0
POSE_TOPIC = "/world/car_world/dynamic_pose/info"

# ---- 復帰デモ用のスタート姿勢（ACT_START=recovery のとき）----
# 評価で見られた「スタックしかけ・周回しかけ」の状態から始め、人が切り返してゴールまで走る。
# (説明, x範囲, y範囲, yaw候補範囲のリスト)  yaw は rad、0 がゴール方向(+x)
BALLS = [("red", 3.5, 0.5), ("green", 7.0, -0.5), ("blue", 10.5, 0.5)]
RECOVERY_STARTS = [
    ("赤玉手前で大きくずれ", (2.0, 3.0), (-1.2, 1.2), [(0.8, 1.5), (-1.5, -0.8)]),
    ("赤玉付近で周回しかけ", (3.0, 4.5), (-0.6, 1.5), [(1.2, 2.9), (-2.3, -0.8)]),
    ("緑玉手前で大きくずれ", (5.5, 6.5), (-1.2, 1.2), [(0.8, 1.5), (-1.5, -0.8)]),
    ("緑玉付近で周回しかけ", (6.0, 8.0), (-1.5, 0.6), [(1.2, 2.6), (-2.6, -1.2)]),
    ("青玉手前で大きくずれ", (9.0, 10.0), (-1.2, 1.2), [(0.8, 1.5), (-1.5, -0.8)]),
]
BALL_CLEAR = 1.0   # 球の中心からこれ以上離して置く [m]
Y_LIMIT = 1.6      # 壁に近すぎる位置は避ける [m]


def sample_recovery_start():
    for _ in range(200):
        name, xr, yr, yaws = random.choice(RECOVERY_STARTS)
        x, y = random.uniform(*xr), random.uniform(*yr)
        yaw = random.uniform(*random.choice(yaws))
        cx, cy = x + 0.5 * math.cos(yaw), y + 0.5 * math.sin(yaw)   # 車体中心
        if abs(cy) > Y_LIMIT:
            continue
        if all(math.hypot(cx - bx, cy - by) > BALL_CLEAR for _, bx, by in BALLS):
            return name, x, y, yaw
    return "初期位置", 0.0, 0.0, 0.0


class EpisodeCollector(Node):
    def __init__(self):
        super().__init__("episode_collector")
        self.root = os.path.expanduser(os.environ.get("ACT_DATASET_DIR", "~/ros2_ws/dataset_act"))
        self.img_dir = os.path.join(self.root, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.csv_path = os.path.join(self.root, "data.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["episode", "frame", "image_name", "linear_x", "angular_z", "sim_time"])
        self.episode = self.next_episode()
        self.recovery = os.environ.get("ACT_START", "") == "recovery"
        self.start = ("初期位置", 0.0, 0.0, 0.0)
        self.starts_path = os.path.join(self.root, "starts.csv")

        self.br = CvBridge()
        self.cmd = Twist()
        self.rows = []          # 録画中エピソードの行（保存は終了時にまとめて）
        self.recording = False
        self.armed = True       # リセット直後: 次の操作で録画開始
        self.x = None
        self.t0 = None
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Twist, "/cmd_vel", self.cmd_cb, 10)
        self.create_subscription(Image, "/camera/image_raw", self.img_cb, 10)
        self.create_subscription(TFMessage, POSE_TOPIC, self.pose_cb, 10)
        self.create_timer(0.05, self.key_cb)

        self.reset_pose()
        self.info(f"保存先 {self.root}  次のエピソード番号 {self.episode}")
        self.info("teleop で走らせると録画開始。ゴール(x>13)で自動保存→初期位置へ。 d=破棄 r=リセット q=終了")

    def info(self, s):
        self.get_logger().info(s)

    def next_episode(self):
        eps = []
        with open(self.csv_path, newline="") as f:
            for row in csv.DictReader(f):
                eps.append(int(row["episode"]))
        return max(eps) + 1 if eps else 0

    def reset_pose(self):
        for _ in range(3):
            self.cmd_pub.publish(Twist())
        self.start = sample_recovery_start() if self.recovery else ("初期位置", 0.0, 0.0, 0.0)
        _, x, y, yaw = self.start
        req = (f'name: "vehicle_blue", position: {{x: {x:.3f}, y: {y:.3f}, z: 0.05}}, '
               f'orientation: {{z: {math.sin(yaw / 2):.5f}, w: {math.cos(yaw / 2):.5f}}}')
        subprocess.run(["bash", "-lc",
                        "ign service -s /world/car_world/set_pose --reqtype ignition.msgs.Pose "
                        f"--reptype ignition.msgs.Boolean --timeout 3000 --req '{req}'"],
                       capture_output=True)
        self.recording = False
        self.armed = True
        self.rows = []
        if self.recovery:
            self.info(f"スタート: {self.start[0]}  x={x:.2f} y={y:.2f} yaw={math.degrees(yaw):.0f}°"
                      "（r で引き直し）")

    def cmd_cb(self, msg):
        self.cmd = msg
        if self.armed and (msg.linear.x != 0.0 or msg.angular.z != 0.0):
            self.armed = False
            self.recording = True
            self.t0 = None
            self.info(f"エピソード {self.episode} 録画開始")

    def pose_cb(self, msg):
        if not msg.transforms:
            return
        tr = msg.transforms[0].transform
        q = tr.rotation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        self.x = tr.translation.x + 0.5 * np.cos(yaw)
        if self.recording and self.x > GOAL_X:
            self.finish(save=True)

    def img_cb(self, msg):
        if not self.recording:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        if t - self.t0 > MAX_SEC:
            self.info("時間切れ → 破棄します")
            self.finish(save=False)
            return
        frame = len(self.rows)
        name = f"ep{self.episode:03d}_{frame:05d}.jpg"
        cv2.imwrite(os.path.join(self.img_dir, name), self.br.imgmsg_to_cv2(msg, "bgr8"))
        self.rows.append([self.episode, frame, name, self.cmd.linear.x, self.cmd.angular.z, f"{t:.2f}"])

    def finish(self, save):
        self.recording = False
        if save and self.rows:
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerows(self.rows)
            moving = sum(1 for r in self.rows if r[3] != 0.0 or r[4] != 0.0)
            new = not os.path.exists(self.starts_path)
            with open(self.starts_path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["episode", "start", "x", "y", "yaw"])
                n, x, y, yaw = self.start
                w.writerow([self.episode, n, f"{x:.3f}", f"{y:.3f}", f"{yaw:.3f}"])
            self.info(f"✅ エピソード {self.episode} 保存: {len(self.rows)}フレーム（走行中 {moving}）")
            self.episode += 1
        else:
            for r in self.rows:
                p = os.path.join(self.img_dir, r[2])
                if os.path.exists(p):
                    os.remove(p)
            self.info(f"🗑 エピソード {self.episode} を破棄")
        self.reset_pose()
        self.info(f"初期位置に戻しました。操作するとエピソード {self.episode} を開始します")

    def key_cb(self):
        if not sys.stdin.isatty():
            return
        if select.select([sys.stdin], [], [], 0)[0]:
            k = sys.stdin.read(1)
            if k == "d" and self.recording:
                self.finish(save=False)
            elif k == "r" and not self.recording:
                self.reset_pose()
                self.info("スタート位置を再設定しました")
            elif k == "q":
                raise KeyboardInterrupt


def main():
    rclpy.init()
    bridge = subprocess.Popen(
        ["ros2", "run", "ros_gz_bridge", "parameter_bridge",
         f"{POSE_TOPIC}@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    old = termios.tcgetattr(sys.stdin) if sys.stdin.isatty() else None
    if old:
        tty.setcbreak(sys.stdin.fileno())
    node = EpisodeCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.recording:
            node.info("録画中のエピソードは破棄します")
            node.finish(save=False)
        if old:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        os.killpg(bridge.pid, signal.SIGINT)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
