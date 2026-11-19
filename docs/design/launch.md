# Launch files

Every launch file in the repo, what it includes and which nodes it starts. Use `arm_bringup.launch.py` on the robot and `move_to_pose_sim.launch.py` against the MuJoCo sim.

## The press stack

`arm_bringup.launch.py` is the arm-only entry point. It brings up nothing from navigation or perception.

```mermaid
flowchart TB
  AB["humanoid_arm_skills/arm_bringup.launch.py"]
  UIL["unitree_interface/unitree_interface.launch.py<br/>other repo"]
  FOX["foxglove_bridge"]
  AV["humanoid_arm_skills/humanoid_arm_visual_servoing.launch.py"]
  DISP["g1_description/display.launch.py<br/>if PublishTF"]
  EKFL["g1_pilot/body_ekf.launch.py<br/>if body_ekf"]
  MTPL["g1_pilot/move_to_pose.launch.py"]
  PF["arm_pose_filter<br/>pose_filtering"]
  VS["humanoid_arm_vs"]
  EKF["body_ekf<br/>config body_ekf_ROBOT.yaml"]
  MTP["move_to_pose_server"]
  IK["ik_server"]
  RSP["robot_state_publisher, joint_state_publisher<br/>feedback view and control_viz view"]
  STP["static_transform_publisher<br/>pelvis_reference to control/pelvis_reference"]
  RV["rviz2"]
  AB --> UIL
  AB --> FOX
  AB --> AV
  AV --> DISP
  AV --> EKFL
  AV --> MTPL
  AV --> PF
  AV --> VS
  EKFL --> EKF
  MTPL --> MTP
  MTPL --> IK
  DISP --> RSP
  DISP --> STP
  DISP --> RV
```

Pass `PublishTF:=false` on the robot, where there is no display. `body_ekf.launch.py` loads `config/body_ekf_<robot>.yaml`; only `stanley` has a file in the repo.

## Key launch arguments

The press launch forwards most arguments to its includes. Defaults below come from `humanoid_arm_visual_servoing.launch.py`.

| Argument | Default | Goes to |
|---|---|---|
| `normal_pose_topic` | `track_3d/selected_normal_pose` | filter input; `_filtered/arm` output; `_filtered/arm_upright` skill input |
| `target_frame_arm` | `foot_world` | filter frame and `move_to_pose_server` `planning_frame` |
| `filter_type` | `kalman` | filter, with `filter_alpha` and the `kf_*` noise values |
| `move_to_pose_action` | `/move_to_pose` | server action name and skill client |
| `arm_safety_distance_x`, `_y`, `_z` | `0.0` | `move_to_pose_server` and `ik_server` press offset |
| `feedback_topic`, `position_control_topic`, `traj_topic` | `feedback`, `position_control`, `traj` | `move_to_pose_server` (feedback also to `ik_server`) |
| `body_ekf` | `true` | starts `body_ekf` |
| `PublishTF` | `true` | starts `display.launch.py` |

`move_to_pose.launch.py` alone defaults `planning_frame` to `left_ankle_roll_link` and `check_reach_service` to `/check_reach`.

## Simulation

`move_to_pose_sim.launch.py` runs the planner against the MuJoCo sim in position mode. It can send one goal after a delay.

```mermaid
flowchart TB
  SIM["move_to_pose_sim.launch.py"]
  MJ["python -m g1_mujoco.mujoco_test<br/>G1_TEST_MODE=position"]
  DISP["g1_description/display.launch.py<br/>if PublishTF"]
  MTP["move_to_pose_server"]
  GOAL["ros2 action send_goal /move_to_pose<br/>after goal_delay 8 s, if send_goal"]
  SIM --> MJ
  SIM --> DISP
  SIM --> MTP
  SIM --> GOAL
```

The sim speaks `/position_control` and `/feedback`. This launch starts no `ik_server`.

## Demos and older tools

The remaining `g1_pilot` launch files each start one node, most with `display.launch.py` when `PublishTF` is true.

```mermaid
flowchart LR
  C["controller.launch.py"] --> JC["joint_controller as trajectory_controller"]
  P["plan_trajectory.launch.py"] --> PT["plan_trajecory as plan_trajectory_server"]
  VSL["visual_servo.launch.py"] --> VSN["visual_servo"]
  VSA["visual_servo_aruco.launch.py"] --> VSN2["visual_servo"]
  VSA --> RS["realsense2_camera rs_launch.py"]
  VSA --> AR["hand_eye_calibration aruco_pose_detector"]
  IKD["ik_demo.launch.py"] --> IKP["ik_joint_state_publisher"]
  GF["grav_ff.launch.py"] --> GFN["grav_ff"]
  IA["impede_arms_demo.launch.py"] --> IAN["impede_arms"]
  JT["joint_traj_demo.launch.py"] --> JTN["joint_traj_publisher"]
  BE["body_ekf.launch.py"] --> BEN["body_ekf"]
```

`visual_servo.launch.py` remaps the right-arm goal topic to `/track_3d/selected_normal_pose_filtered/arm_upright`, the filtered pose the press stack uses.
