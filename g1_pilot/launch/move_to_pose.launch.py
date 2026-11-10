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
    ld.add_action(DeclareLaunchArgument('traj_topic', default_value='traj'))
    ld.add_action(DeclareLaunchArgument('planning_frame',
                                        default_value='left_ankle_roll_link'))
    for axis in 'xyz':
        ld.add_action(DeclareLaunchArgument(f'arm_safety_distance_{axis}', default_value='0.0'))

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
            'traj_topic': LaunchConfiguration('traj_topic'),
            'planning_frame': LaunchConfiguration('planning_frame'),
            **{f'arm_safety_distance_{axis}': LaunchConfiguration(f'arm_safety_distance_{axis}')
               for axis in 'xyz'},
        }],
    ))
    return ld
