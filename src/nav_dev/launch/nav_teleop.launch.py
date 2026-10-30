import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share_dir = get_package_share_directory('nav_dev')
    # 今回はモデル(world.sdf)をパッケージのルートに置いている前提
    model_path = pkg_share_dir
    
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    # use_teleop:=false で xterm の teleop_node を起動しない
    # （キー無しの間も /cmd_vel にゼロを10Hzで送り続け、自動走行や teleop_twist_keyboard の指令を打ち消すため）
    use_teleop = LaunchConfiguration('use_teleop')
    declare_use_teleop = DeclareLaunchArgument('use_teleop', default_value='true',
                                               description='xterm で teleop_node (WASD) を起動するか')

    # Gazeboがモデルを見つけられるように環境変数をセット
    ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[os.path.join("/opt/ros/humble", "share"), ":" + model_path]
    )

    # Gazeboの起動
    ign_gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('ros_ign_gazebo'), 'launch', 'ign_gazebo.launch.py')]
        ),
        launch_arguments=[('ign_args', ['-r -v 3 ', os.path.join(model_path, "world.sdf")])]
    )

    # ブリッジノードの起動（先ほど作ったyamlを読み込む）
    bridge = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': os.path.join(pkg_share_dir, 'config', 'nav_teleop.yaml'),
        }, {'use_sim_time': use_sim_time}],
        output='screen'
    )

    # 操作用ノードの起動（xtermの別窓で起動）
    teleop_node = Node(
        package='nav_dev',
        executable='teleop_node',
        output='screen',
        prefix="xterm -e",
        condition=IfCondition(use_teleop)
    )

    return LaunchDescription([
        declare_use_teleop,
        ign_resource_path,
        ign_gz,
        bridge,
        teleop_node,
    ])