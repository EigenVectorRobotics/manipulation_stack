"""Arm bringup: the robot interface, the viewer bridge, and the arm servo stack.

    ros2 launch humanoid_arm_skills arm_bringup.launch.py

This is the arm-only path. It deliberately brings up nothing from the
navigation or perception side: no realsense, no nav stack, no gadgets.

RViz comes up with the state publishers, both guarded by PublishTF. On the
robot there is no display and foxglove is the viewer, so pass PublishTF:=false
there; that drops the TF publishers along with it.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'PublishTF', default_value='true',
            description='Start the state publishers and RViz'),
        DeclareLaunchArgument(
            'foxglove_port', default_value='8765',
            description='Port the foxglove bridge listens on'),
        DeclareLaunchArgument(
            'unitree_executable', default_value='unitree_interface_node',
            description='Set to unitree_interface_tuning_node to tune gains'),
    ]

    # Speaks /position_control and /feedback, which is what move_to_pose
    # expects; the remappings live in that launch file.
    unitree_interface = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('unitree_interface'),
                                  'launch', 'unitree_interface.launch.py'])
        ),
        launch_arguments={
            'executable': LaunchConfiguration('unitree_executable'),
        }.items(),
    )

    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'port': LaunchConfiguration('foxglove_port'),
            'send_buffer_limit': 10000000,
            'num_threads': 2,
        }],
        output='screen',
    )

    # move_to_pose, pose_filtering and the visual servo skill.
    arm_visual_servoing = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('humanoid_arm_skills'),
                                  'launch',
                                  'humanoid_arm_visual_servoing.launch.py'])
        ),
        launch_arguments={
            'PublishTF': LaunchConfiguration('PublishTF'),
        }.items(),
    )

    return LaunchDescription(
        args + [unitree_interface, foxglove_bridge, arm_visual_servoing])
