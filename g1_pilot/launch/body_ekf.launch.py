from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    package_name = 'g1_pilot'
    package_path = FindPackageShare(package_name)

    ld.add_action(DeclareLaunchArgument('robot', default_value='stanley', description='robot name: loads config/body_ekf_<robot>.yaml'))
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='false'))

    config_file = PathJoinSubstitution([package_path, 'config', ['body_ekf_', LaunchConfiguration('robot'), '.yaml']])

    ld.add_action(Node(
        package=package_name,
        executable='body_ekf',
        name='body_ekf',
        output='screen',
        parameters=[config_file, {
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    ))
    return ld
