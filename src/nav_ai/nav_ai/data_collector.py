import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import os
import csv

# 前回の続きから番号を振るときに空ける番号の幅。
# 番号が連続しない所が「別の走行」の区切りとして学習スクリプトに認識される
SESSION_GAP = 100


class DataCollector(Node):
    """操作中のカメラ画像と操作量を保存する。既存データがあれば続きの番号で追記する。

    保存先は環境変数 DATASET_DIR で変更可（既定: ~/ros2_ws/dataset）
    """

    def __init__(self):
        super().__init__('data_collector')

        self.save_dir = os.path.expanduser(os.environ.get('DATASET_DIR', '~/ros2_ws/dataset'))
        self.image_dir = os.path.join(self.save_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)

        self.csv_path = os.path.join(self.save_dir, 'data.csv')
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_name', 'linear_x', 'angular_z'])

        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)

        self.br = CvBridge()
        self.current_cmd = Twist()
        self.image_count = self.next_index()
        self.saved = 0
        self.is_recording = False

        self.get_logger().info(f'データ収集ノードが起動しました。保存先: {self.save_dir}')
        self.get_logger().info(f'画像番号 {self.image_count:05d} から追記します。操作すると録画を開始します。')

    def next_index(self):
        """画像フォルダと CSV の両方を見て、既存の最大番号 + SESSION_GAP を返す（空なら 0）"""
        mx = -1
        for f in os.listdir(self.image_dir):
            stem = os.path.splitext(f)[0]
            if stem.isdigit():
                mx = max(mx, int(stem))
        with open(self.csv_path, newline='') as f:
            for row in csv.reader(f):
                stem = os.path.splitext(row[0])[0] if row else ''
                if stem.isdigit():
                    mx = max(mx, int(stem))
        return 0 if mx < 0 else mx + 1 + SESSION_GAP

    def cmd_callback(self, msg):
        self.current_cmd = msg
        if msg.linear.x != 0.0 or msg.angular.z != 0.0:
            self.is_recording = True

    def image_callback(self, msg):
        if self.is_recording:
            cv_image = self.br.imgmsg_to_cv2(msg, "bgr8")

            image_name = f"{self.image_count:05d}.jpg"
            image_path = os.path.join(self.image_dir, image_name)
            if os.path.exists(image_path):   # 念のため: 既存画像は絶対に上書きしない
                self.image_count = self.next_index()
                image_name = f"{self.image_count:05d}.jpg"
                image_path = os.path.join(self.image_dir, image_name)
            cv2.imwrite(image_path, cv_image)

            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([image_name, self.current_cmd.linear.x, self.current_cmd.angular.z])

            self.image_count += 1
            self.saved += 1
            if self.saved % 50 == 0:
                self.get_logger().info(f'データ保存中... (今回 {self.saved}枚, 最新 {image_name})')


def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()





# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist
# from gazebo_msgs.srv import SetEntityState
# from cv_bridge import CvBridge
# import cv2
# import os
# import csv
# import sys
# import termios
# import tty
# import select

# class DataCollector(Node):
#     def __init__(self):
#         super().__init__('data_collector')
        
#         # データの保存先ディレクトリを作成
#         self.save_dir = os.path.expanduser('~/ros2_ws/dataset')
#         self.image_dir = os.path.join(self.save_dir, 'images')
#         os.makedirs(self.image_dir, exist_ok=True)
        
#         # CSVファイルの準備
#         self.csv_path = os.path.join(self.save_dir, 'data.csv')
#         if not os.path.exists(self.csv_path):
#             with open(self.csv_path, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['image_name', 'linear_x', 'angular_z'])

#         # サブスクライバーとクライアント
#         self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
#         self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
#         # Gazeboのモデル位置リセット用クライアント
#         self.reset_client = self.create_client(SetEntityState, '/set_entity_state')
        
#         self.br = CvBridge()
#         self.current_cmd = Twist()
#         self.image_count = 0
#         self.is_recording = False
        
#         # ターミナルのキー入力設定（非ブロッキング用）
#         self.settings = termios.tcgetattr(sys.stdin)
        
#         self.get_logger().info('=== データ収集ノードが起動しました ===')
#         self.get_logger().info('操作が検知されると自動で録画を開始します。')
#         self.get_logger().info('[R]キー または [スペース]キー を押すと、ロボットが自動で初期位置に戻ります！')

#     def reset_robot_position(self):
#         if not self.reset_client.wait_for_service(timeout_sec=1.0):
#             self.get_logger().warn('Gazeboのリセットサービスが見つかりません')
#             return

#         request = SetEntityState.Request()
#         request.state.name = 'vehicle'  # Gazebo上のロボット名（環境に合わせて調整）
#         request.state.pose.position.x = 0.0
#         request.state.pose.position.y = 0.0
#         request.state.pose.position.z = 0.1
#         # 向き（クォータニオン）の初期化
#         request.state.pose.orientation.x = 0.0
#         request.state.pose.orientation.y = 0.0
#         request.state.pose.orientation.z = 0.0
#         request.state.pose.orientation.w = 1.0

#         future = self.reset_client.call_async(request)
#         self.get_logger().info('-> ロボットを初期位置にテレポートしました！')
        
#         # 録画フラグを一時オフにする
#         self.is_recording = False

#     def cmd_callback(self, msg):
#         self.current_cmd = msg
#         if msg.linear.x != 0.0 or msg.angular.z != 0.0:
#             self.is_recording = True

#     def image_callback(self, msg):
#         # ターミナルからのキー入力を監視（Rキーでリセット）
#         if select.select([sys.stdin], [], [], 0.0)[0]:
#             key = sys.stdin.read(1)
#             if key == 'r' or key == 'R' or key == ' ':
#                 self.reset_robot_position()

#         if self.is_recording:
#             cv_image = self.br.imgmsg_to_cv2(msg, "bgr8")
            
#             image_name = f"{self.image_count:05d}.jpg"
#             image_path = os.path.join(self.image_dir, image_name)
#             cv2.imwrite(image_path, cv_image)
            
#             with open(self.csv_path, 'a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([image_name, self.current_cmd.linear.x, self.current_cmd.angular.z])
            
#             self.image_count += 1
#             if self.image_count % 50 == 0:
#                 self.get_logger().info(f'データ保存中... ({self.image_count}枚)')

# def main(args=None):
#     rclpy.init(args=args)
#     node = DataCollector()
    
#     # ターミナルの入力モード変更
#     tty.setcbreak(sys.stdin.fileno())
    
#     try:
#         rclpy.spin(node)
#     finally:
#         termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.settings)
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()





# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist
# from cv_bridge import CvBridge
# import cv2
# import os
# import csv

# class DataCollector(Node):
#     def __init__(self):
#         super().__init__('data_collector')
        
#         # データの保存先ディレクトリを作成
#         self.save_dir = os.path.expanduser('~/ros2_ws/dataset')
#         self.image_dir = os.path.join(self.save_dir, 'images')
#         os.makedirs(self.image_dir, exist_ok=True)
        
#         # CSVファイルの準備
#         self.csv_path = os.path.join(self.save_dir, 'data.csv')
#         # ファイルが存在しない場合のみヘッダーを書き込む
#         if not os.path.exists(self.csv_path):
#             with open(self.csv_path, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['image_name', 'linear_x', 'angular_z'])

#         # サブスクライバー
#         self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
#         self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
#         self.br = CvBridge()
#         self.current_cmd = Twist()
#         self.image_count = 0
#         self.is_recording = False
        
#         self.get_logger().info('データ収集ノードが起動しました。')
#         self.get_logger().info('teleopの操作が検知されると自動で録画(保存)を開始します。')

#     def cmd_callback(self, msg):
#         self.current_cmd = msg
#         # 操作が0でない（動かしている）場合、録画フラグをオンにする
#         if msg.linear.x != 0.0 or msg.angular.z != 0.0:
#             self.is_recording = True

#     def image_callback(self, msg):
#         # 操作中のみデータを保存する（停止中の無駄な画像を省くため）
#         if self.is_recording:
#             cv_image = self.br.imgmsg_to_cv2(msg, "bgr8")
            
#             # 画像を保存
#             image_name = f"{self.image_count:05d}.jpg"
#             image_path = os.path.join(self.image_dir, image_name)
#             cv2.imwrite(image_path, cv_image)
            
#             # CSVに「画像名」「直進速度」「旋回速度」を記録
#             with open(self.csv_path, 'a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([image_name, self.current_cmd.linear.x, self.current_cmd.angular.z])
            
#             self.image_count += 1
#             if self.image_count % 50 == 0:
#                 self.get_logger().info(f'データ保存中... ({self.image_count}枚)')

# def main(args=None):
#     rclpy.init(args=args)
#     node = DataCollector()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()