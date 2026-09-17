from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        # pose_filtering
        DeclareLaunchArgument('normal_pose_topic',
                              default_value='track_3d/selected_normal_pose'),
        DeclareLaunchArgument('target_frame_arm',
                              default_value='pelvis',
                              description='TF frame to transform normal pose into'),
        DeclareLaunchArgument('filter_window_size', default_value='10'),

        # humanoid_arm_visual_servoing
        DeclareLaunchArgument('keyboard_input_topic',
                              default_value='/keyboard_input'),
        DeclareLaunchArgument('left_ee_pose_topic',
                              default_value='/left_ee_pose'),
        DeclareLaunchArgument('right_ee_pose_topic',
                              default_value='/right_ee_pose'),
        DeclareLaunchArgument('move_to_pose_action',
                              default_value='/move_to_pose'),
        DeclareLaunchArgument('arm_safety_distance_x', default_value='-0.0'),
        DeclareLaunchArgument('arm_safety_distance_y', default_value='0.0'),
        DeclareLaunchArgument('arm_safety_distance_z', default_value='0.0'),

        # g1_pilot controller / plan_trajectory
        DeclareLaunchArgument('namespace', default_value=''),
        DeclareLaunchArgument('PublishTF', default_value='true',
                              description='Whether to include display.launch.py for TF publishing'),
        DeclareLaunchArgument('feedback_topic', default_value='feedback'),
        DeclareLaunchArgument('position_control_topic',
                              default_value='position_control'),
        DeclareLaunchArgument('traj_topic', default_value='traj'),
    ]

    g1_pilot_share = FindPackageShare('g1_pilot')
    g1_description_share = FindPackageShare('g1_description')

    display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([g1_description_share, 'launch',
                                  'display.launch.py'])
        ),
        condition=IfCondition(LaunchConfiguration('PublishTF')),
    )

    move_to_pose_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([g1_pilot_share, 'launch',
                                  'move_to_pose.launch.py'])
        ),
        launch_arguments={
            'namespace': LaunchConfiguration('namespace'),
            'move_to_pose_action': LaunchConfiguration('move_to_pose_action'),
            'feedback_topic': LaunchConfiguration('feedback_topic'),
            'position_control_topic': LaunchConfiguration('position_control_topic'),
            'left_ee_pose_topic': LaunchConfiguration('left_ee_pose_topic'),
            'right_ee_pose_topic': LaunchConfiguration('right_ee_pose_topic'),
            'traj_topic': LaunchConfiguration('traj_topic'),
        }.items(),
    )

    pose_filter_node = Node(
        package='humanoid_arm_skills',
        executable='pose_filtering',
        name='arm_pose_filter',
        output='screen',
        parameters=[{
            'input_topic':  LaunchConfiguration('normal_pose_topic'),
            'output_topic': PythonExpression(
                ["'", LaunchConfiguration('normal_pose_topic'), "_filtered/arm'"]),
            'msg_type':     'PoseStamped',
            'target_frame_arm': LaunchConfiguration('target_frame_arm'),
            'window_size':  LaunchConfiguration('filter_window_size'),
        }]
    )

    # pose_filtering defaults output to {input_topic}_filtered
    servo_arm_node = Node(
        package='humanoid_arm_skills',
        executable='humanoid_arm_vs',
        name='humanoid_arm_vs',
        output='screen',
        parameters=[{
            'filtered_pose_upright_topic': PythonExpression(
                ["'", LaunchConfiguration('normal_pose_topic'),
                 "_filtered/arm_upright'"]),
            'keyboard_input_topic': LaunchConfiguration(
                'keyboard_input_topic'),
            'left_ee_pose_topic': LaunchConfiguration('left_ee_pose_topic'),
            'right_ee_pose_topic': LaunchConfiguration('right_ee_pose_topic'),
            'move_to_pose_action': LaunchConfiguration('move_to_pose_action'),
            'arm_safety_distance_x': LaunchConfiguration(
                'arm_safety_distance_x'),
            'arm_safety_distance_y': LaunchConfiguration(
                'arm_safety_distance_y'),
            'arm_safety_distance_z': LaunchConfiguration(
                'arm_safety_distance_z'),
        }]
    )

    return LaunchDescription(args + [
        display_launch,
        move_to_pose_launch,
        pose_filter_node,
        servo_arm_node,
    ])
