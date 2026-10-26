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
                              default_value='left_ankle_roll_link',
                              description='TF frame to transform normal pose into'),
        DeclareLaunchArgument('filter_alpha', default_value='0.2',
                              description='EMA new-observation weight in (0, 1]'),
        DeclareLaunchArgument('filter_type', default_value='kalman',
                              description='Target filter: ema or kalman'),
        DeclareLaunchArgument('kf_q_pos', default_value='0.003',
                              description='Kalman target random walk (m/sqrt(s))'),
        DeclareLaunchArgument('kf_sig_lat', default_value='0.004',
                              description='Kalman lateral camera noise (m)'),
        DeclareLaunchArgument('kf_sig_depth', default_value='0.015',
                              description='Kalman depth camera noise along the ray (m)'),
        DeclareLaunchArgument('kf_sig_kin', default_value='0.002',
                              description='Kalman isotropic kinematic noise (m)'),
        DeclareLaunchArgument('kf_q_ang_deg', default_value='0.5',
                              description='Kalman normal random walk (deg/sqrt(s))'),
        DeclareLaunchArgument('kf_sig_ang_deg', default_value='3.0',
                              description='Kalman normal measurement noise (deg)'),
        DeclareLaunchArgument('kf_gate', default_value='16.27',
                              description='Kalman NIS outlier gate (chi2 3 DoF)'),

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
            'planning_frame': LaunchConfiguration('target_frame_arm'),
            **{f'arm_safety_distance_{axis}': LaunchConfiguration(f'arm_safety_distance_{axis}')
               for axis in 'xyz'},
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
            'alpha': LaunchConfiguration('filter_alpha'),
            'filter_type': LaunchConfiguration('filter_type'),
            **{name: LaunchConfiguration(name)
               for name in ('kf_q_pos', 'kf_sig_lat', 'kf_sig_depth', 'kf_sig_kin',
                            'kf_q_ang_deg', 'kf_sig_ang_deg', 'kf_gate')},
        }]
    )

    # pose_filtering defaults output to {input_topic}_filtered
    servo_arm_node = Node(
        package='humanoid_arm_skills',
        executable='humanoid_arm_vs',
        name='humanoid_arm_vs',
        output='screen',
        parameters=[{
            'filtered_pose_topic': PythonExpression(
                ["'", LaunchConfiguration('normal_pose_topic'),
                 "_filtered/arm_approach'"]),
            'keyboard_input_topic': LaunchConfiguration(
                'keyboard_input_topic'),
            'left_ee_pose_topic': LaunchConfiguration('left_ee_pose_topic'),
            'right_ee_pose_topic': LaunchConfiguration('right_ee_pose_topic'),
            'move_to_pose_action': LaunchConfiguration('move_to_pose_action'),
        }]
    )

    return LaunchDescription(args + [
        display_launch,
        move_to_pose_launch,
        pose_filter_node,
        servo_arm_node,
    ])
