from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument('namespace', default_value=''))
    ld.add_action(DeclareLaunchArgument('move_to_pose_action',
                                        default_value='/move_to_pose'))
    ld.add_action(DeclareLaunchArgument('feedback_topic', default_value='feedback'))
    ld.add_action(DeclareLaunchArgument('position_control_topic',
                                        default_value='position_control'))
    ld.add_action(DeclareLaunchArgument('left_ee_pose_topic',
                                        default_value='left_ee_pose'))
    ld.add_action(DeclareLaunchArgument('right_ee_pose_topic',
                                        default_value='right_ee_pose'))
    ld.add_action(DeclareLaunchArgument('traj_topic', default_value='traj'))

    ld.add_action(Node(
        package='g1_pilot',
        executable='move_to_pose',
        name='move_to_pose_server',
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[{
            'action_name': LaunchConfiguration('move_to_pose_action'),
            'feedback_topic': LaunchConfiguration('feedback_topic'),
            'position_control_topic': LaunchConfiguration('position_control_topic'),
            'left_ee_pose_topic': LaunchConfiguration('left_ee_pose_topic'),
            'right_ee_pose_topic': LaunchConfiguration('right_ee_pose_topic'),
            'traj_topic': LaunchConfiguration('traj_topic'),
        }],
    ))
    return ld
