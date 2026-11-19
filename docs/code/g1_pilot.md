# g1_pilot

The C++ arm library `libg1_pilot` and its ROS 2 nodes. The library builds without ROS; the nodes build when a ROS environment is sourced.

- **`include/g1_pilot/g1_pilot.h`**: The public API: `Humanoid`, `JointState`, `JointTrajectory` and `RobotConfig`.
- **`src/core/`**: The library itself: model loading, IK, controllers, planners and the body EKF. See [core](g1_pilot/core.md).
- **`src/move_to_pose.cpp`, `src/move_to_pose/`**: The MoveToPose action server that plans, replans and commands the arm. See [move_to_pose](g1_pilot/move_to_pose.md).
- **`src/ik_server.cpp`**: The `/check_reach` service; it plans as `move_to_pose_server` does and commands nothing.
- **`src/body_ekf.cpp`**: The `body_ekf` node; it fuses the IMUs and leg encoders and publishes the `foot_world` frame.
- **`src/joint_controller.cpp`**: The `trajectory_controller` action server that tracks a given joint trajectory.
- **`src/plan_trajecory.cpp`**: The `plan_trajectory` service that returns a joint trajectory between two poses.
- **`src/visual_servo.cpp`**: An older in-process servo node driven by `goal_pose/left` and `goal_pose/right` topics.
- **`src/impede_arms.cpp`, `src/grav_ff.cpp`**: Impedance control and gravity feedforward on `effort_control`.
- **`src/joint_traj_publisher.cpp`, `src/joint_state_publisher.cpp`**: Planning and IK demos that publish on `position_control`.
- **`src/helper_funcs.h`**: Conversions between Eigen, trajectories, paths and poses, workspace limits, and the `left_ee`/`right_ee` static TF.
- **`config/g1.yaml`**: Locked joints, end-effector frames, joint limit overrides, joint dynamics and the collision margin.
- **`config/body_ekf_stanley.yaml`**: Per-robot `body_ekf` parameters; the node does not read `camera_imu_topic`, `tf_root_frame`, `publish_tf` or `use_header_stamps`.
- **`launch/`**: Eleven launch files; see [Launch files](../design/launch.md).
- **`test/`**: GTest suites, benchmarks and replay tools. See [test](g1_pilot/test.md).
