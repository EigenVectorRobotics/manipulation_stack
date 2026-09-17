"""Bring up move_to_pose against the MuJoCo sim, with RViz.

    ros2 launch g1_pilot move_to_pose_sim.launch.py

The sim runs in position mode (g1_mujoco/config.py TEST_MODE), so it subscribes
to position_control and publishes feedback — exactly what move_to_pose speaks.
PublishTF brings up robot_state_publisher and RViz via g1_description.

Send a different target without editing anything:

    ros2 launch g1_pilot move_to_pose_sim.launch.py target:="[0.4, -0.4, 0.2]"
    ros2 launch g1_pilot move_to_pose_sim.launch.py send_goal:=false

Pass target_topic:=/some/pose to have move_to_pose follow a moving target. It
stores the latest pose today; nothing replans off it yet.
"""

import ast
import os
import sys

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, OpaqueFunction, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _goal(context):
    if context.launch_configurations['send_goal'].lower() not in ('true', '1'):
        return []

    x, y, z = ast.literal_eval(context.launch_configurations['target'])
    qw, qx, qy, qz = ast.literal_eval(context.launch_configurations['orientation'])
    frame = context.launch_configurations['target_frame']
    left = context.launch_configurations['left_arm'].lower() in ('true', '1')
    topic = context.launch_configurations['target_topic']

    goal_yaml = (
        f"{{target_pose: {{header: {{frame_id: {frame}}}, pose: {{"
        f"position: {{x: {x}, y: {y}, z: {z}}}, "
        f"orientation: {{w: {qw}, x: {qx}, y: {qy}, z: {qz}}}"
        f"}}}}, left_arm: {str(left).lower()}, target_topic: '{topic}'}}"
    )

    # Delayed so the model has loaded and joint feedback has arrived: planning
    # is refused outright until move_to_pose has seen a joint state.
    return [TimerAction(
        period=float(context.launch_configurations['goal_delay']),
        actions=[ExecuteProcess(
            cmd=['ros2', 'action', 'send_goal', '/move_to_pose',
                 'humanoid_manipulation_interfaces/action/MoveToPose',
                 goal_yaml, '--feedback'],
            output='screen',
        )],
    )]


def generate_launch_description():
    # ros2 launch may run under a different interpreter than the sim needs.
    venv = os.environ.get('VIRTUAL_ENV')
    python_exec = os.path.join(venv, 'bin', 'python3') if venv else sys.executable

    args = [
        DeclareLaunchArgument('PublishTF', default_value='true',
                              description='Start robot_state_publisher and RViz'),
        DeclareLaunchArgument('send_goal', default_value='true'),
        DeclareLaunchArgument('target', default_value='[0.4, -0.4, 0.2]',
                              description='Target position [x, y, z]'),
        DeclareLaunchArgument('orientation', default_value='[1.0, 0.0, 0.0, 0.0]',
                              description='Target orientation [qw, qx, qy, qz]'),
        DeclareLaunchArgument('target_frame', default_value='pelvis'),
        DeclareLaunchArgument('left_arm', default_value='false'),
        DeclareLaunchArgument('target_topic', default_value='',
                              description='Topic publishing updated target '
                                          'poses; empty for a static target'),
        DeclareLaunchArgument('goal_delay', default_value='8.0',
                              description='Seconds to wait before sending the goal'),
    ]

    sim = ExecuteProcess(
        cmd=[python_exec, '-m', 'g1_mujoco.mujoco_test'],
        additional_env={'G1_TEST_MODE': 'position'},
        output='screen',
    )

    display = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('g1_description'), 'launch',
                                  'display.launch.py'])
        ),
        condition=IfCondition(LaunchConfiguration('PublishTF')),
    )

    move_to_pose = Node(
        package='g1_pilot',
        executable='move_to_pose',
        name='move_to_pose_server',
        output='screen',
    )

    return LaunchDescription(args + [sim, display, move_to_pose,
                                     OpaqueFunction(function=_goal)])
