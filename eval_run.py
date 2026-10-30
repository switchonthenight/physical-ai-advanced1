"""
自動走行の評価: 車を初期位置に戻す → drive_node で走らせる → 真値の軌跡を記録 → 図と判定を出す

使い方（Gazebo を起動した状態で）:
  python3 ~/ros2_ws/eval_run.py [タイムアウト秒=60] [ラベル]
出力: ~/ros2_ws/logs/eval_<ラベル>.png / .csv / _drive.csv
"""
import csv
import math
import os
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from tf2_msgs.msg import TFMessage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

TIMEOUT = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
LABEL = sys.argv[2] if len(sys.argv) > 2 else time.strftime("%H%M%S")
LOGDIR = os.path.expanduser("~/ros2_ws/logs")
GOAL_X = 13.0
BALLS = [("red", 3.5, 0.5), ("green", 7.0, -0.5), ("blue", 10.5, 0.5)]
WANT = {"red": "right", "green": "left", "blue": "right"}   # 想定の通過側
POSE_TOPIC = "/world/car_world/dynamic_pose/info"


def sh(cmd):
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)


def start_pose():
    """ENV: EVAL_START="x,y,yaw[deg]"（モデル原点の姿勢。episode_collector と同じ定義）。未指定なら初期位置"""
    v = os.environ.get("EVAL_START", "")
    if not v:
        return 0.0, 0.0, 0.0
    x, y, yaw_deg = (float(a) for a in v.split(","))
    return x, y, math.radians(yaw_deg)


def reset_pose():
    x, y, yaw = start_pose()
    req = (f'name: "vehicle_blue", position: {{x: {x:.3f}, y: {y:.3f}, z: 0.05}}, '
           f'orientation: {{z: {math.sin(yaw / 2):.5f}, w: {math.cos(yaw / 2):.5f}}}')
    r = sh("ign service -s /world/car_world/set_pose --reqtype ignition.msgs.Pose "
           f"--reptype ignition.msgs.Boolean --timeout 3000 --req '{req}'")
    return "true" in r.stdout


class Recorder(Node):
    def __init__(self):
        super().__init__("eval_recorder")
        self.traj = []
        self.create_subscription(TFMessage, POSE_TOPIC, self.cb, 10)
        self.stop_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def cb(self, msg):
        if msg.transforms:
            tr = msg.transforms[0].transform       # 先頭が車体（vehicle_blue）
            q = tr.rotation
            yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
            st = msg.transforms[0].header.stamp
            self.traj.append((st.sec + st.nanosec * 1e-9, tr.translation.x, tr.translation.y, yaw))


def chassis_xy(x, y, yaw):
    # モデル原点から車体中心(+0.5m前)へ
    return x + 0.5 * np.cos(yaw), y + 0.5 * np.sin(yaw)


def teleop_pids():
    """teleop_node はキーを押していない間も /cmd_vel にゼロを10Hzで送り続け、推論ノードの指令を打ち消す。
    評価中は SIGSTOP で一時停止し、終了後に SIGCONT で再開する（ENV: EVAL_KEEP_TELEOP=1 で無効）"""
    if os.environ.get("EVAL_KEEP_TELEOP") == "1":
        return []
    out = subprocess.run(["pgrep", "-f", "lib/nav_dev/teleop_node"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def main():
    os.makedirs(LOGDIR, exist_ok=True)
    rclpy.init()
    rec = Recorder()
    bridge = subprocess.Popen(
        ["ros2", "run", "ros_gz_bridge", "parameter_bridge",
         f"{POSE_TOPIC}@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(5):
        rec.stop_pub.publish(Twist())
        rclpy.spin_once(rec, timeout_sec=0.1)
    ok = reset_pose()
    t0 = time.time()
    while time.time() - t0 < 2.0:
        rec.stop_pub.publish(Twist())
        rclpy.spin_once(rec, timeout_sec=0.05)
    rec.traj.clear()

    paused = teleop_pids()
    for pid in paused:
        os.kill(pid, signal.SIGSTOP)
    print(f"teleop一時停止: {paused}")

    drive_log = os.path.join(LOGDIR, f"eval_{LABEL}_drive.csv")
    env = dict(os.environ, DRIVE_HEADLESS="1", DRIVE_LOG=drive_log)
    drive = subprocess.Popen(["ros2", "run", "nav_ai", os.environ.get("DRIVE_NODE", "drive_node")], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    t0 = time.time()
    result = "timeout"
    while time.time() - t0 < TIMEOUT:
        rclpy.spin_once(rec, timeout_sec=0.1)
        if rec.traj:
            cx, _ = chassis_xy(*rec.traj[-1][1:])
            if cx > GOAL_X:
                result = "goal"
                break
    os.killpg(drive.pid, signal.SIGINT)
    try:
        drive.wait(5)
    except subprocess.TimeoutExpired:
        os.killpg(drive.pid, signal.SIGKILL)
    for _ in range(5):
        rec.stop_pub.publish(Twist())
        rclpy.spin_once(rec, timeout_sec=0.05)
    os.killpg(bridge.pid, signal.SIGINT)
    for pid in paused:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass

    tr = np.array(rec.traj) if rec.traj else np.zeros((0, 4))
    with open(os.path.join(LOGDIR, f"eval_{LABEL}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "yaw"])
        w.writerows(tr.round(3).tolist())

    # --- 判定 ---
    sx, sy, syaw = start_pose()
    lines = [f"start x={sx:.2f} y={sy:.2f} yaw={math.degrees(syaw):.0f}deg  "
             f"reset={'ok' if ok else 'FAILED'}  result={result}  wall={time.time()-t0:.1f}s  samples={len(tr)}"]
    if len(tr):
        cxy = np.array([chassis_xy(x, y, yaw) for _, x, y, yaw in tr])
        lines.append(f"最終位置 x={cxy[-1,0]:.2f} y={cxy[-1,1]:.2f}  最大x={cxy[:,0].max():.2f}")
        lines.append(f"球との最小距離(全体)={min(np.min(np.hypot(cxy[:, 0] - bx, cxy[:, 1] - by)) for _, bx, by in BALLS):.2f}m")
        lines.append(f"壁への最小距離(車体中心y) 左={2.4-cxy[:,1].max():.2f}m 右={cxy[:,1].min()+2.4:.2f}m（車幅半分0.7未満なら接触）")
        for name, bx, by in BALLS:
            near = np.abs(cxy[:, 0] - bx) < 0.3
            dmin = np.min(np.hypot(cxy[:, 0] - bx, cxy[:, 1] - by))
            if near.any():
                side = "left" if cxy[near, 1].mean() > by else "right"
                mark = "OK" if side == WANT[name] else "NG"
                lines.append(f"{name:5s}: {side:5s}側を通過 [{mark}]  最小距離={dmin:.2f}m")
            else:
                lines.append(f"{name:5s}: 未到達  最小距離={dmin:.2f}m")

    # --- 図 ---
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.add_patch(Rectangle((-1, 2.4), 16, 0.2, color="0.5"))
    ax.add_patch(Rectangle((-1, -2.6), 16, 0.2, color="0.5"))
    for name, bx, by in BALLS:
        ax.add_patch(Circle((bx, by), 0.3, color=name))
    if len(tr):
        ax.plot(*chassis_xy(*start_pose()), "k*", ms=12)
        sc = ax.scatter(cxy[:, 0], cxy[:, 1], c=tr[:, 0] - tr[0, 0], s=4, cmap="viridis")
        plt.colorbar(sc, ax=ax, label="sim time [s]")
    ax.axvline(GOAL_X, ls="--", c="k", lw=0.8)
    ax.set_xlim(-1.5, 15.5); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"eval {LABEL}: {result}")
    png = os.path.join(LOGDIR, f"eval_{LABEL}.png")
    plt.tight_layout(); plt.savefig(png, dpi=110)
    lines.append(f"図: {png}")
    print("\n".join(lines))
    rec.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
