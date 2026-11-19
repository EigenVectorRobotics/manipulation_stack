# Classes and ownership

The C++ classes of `g1_pilot`, who owns which, and the base classes between the parts. The Python nodes close the page; their full API is in the API reference.

## The Humanoid facade

`Humanoid` is the one public class of `libg1_pilot`. Every node talks to the library through it, and a private `Impl` owns the solvers.

```mermaid
classDiagram
  class Humanoid {
    +Humanoid(RobotConfig* config)
    +grav_ff(JointState) JointState
    +inverse_dynamics(JointState, acceleration) JointState
    +get_joint_names() vector~string~
    +solve_ik(left_wrist, right_wrist, ...) JointState
    +solve_ik(wrist, left, ...) JointState
    +check_collision(q) bool
    +update(JointState)
    +control_both_arms(state, left_pose, right_pose) JointState
    +control_left_arm(state, pose) JointState
    +control_right_arm(state, pose) JointState
    +planTrajectory(start, goal, final_leg_steps, retreat, ...) JointTrajectory
    +replanTrajectory(start, goal, final_leg_steps, retreat, ...) JointTrajectory
    +planFinalLegStep(start, goal, step, final_leg_steps, retreat, ...) JointTrajectory
    +brake(state, ...) JointTrajectory
    +forward_kinematics(q, left) MatrixXd
    +frame_transform(state, base_frame, tip_frame) Matrix4d
    +setLeftArm(bool)
    +plan_failure() string
    +get_left_ee() EEFrame
    +get_right_ee() EEFrame
  }
  class Impl {
    model_full, model
    geom_full, geom
    data, data_full
  }
  class HumanoidIK
  class ImpedanceController
  class VisualServoPlanner
  class RobotConfig {
    asset_file
    asset_root
    config_file
    NUM_DOF
  }
  class JointState {
    name
    position
    velocity
    effort
  }
  class JointTrajectory {
    name
    position, velocity, acceleration
    time_step
    lock_time
  }
  Humanoid *-- Impl : impl_
  Humanoid ..> RobotConfig : reads
  Impl o-- HumanoidIK : shared_ptr ik
  Impl *-- ImpedanceController : controller
  Impl *-- VisualServoPlanner : motion_planner
  VisualServoPlanner o-- HumanoidIK : same ik
  Humanoid ..> JointState
  Humanoid ..> JointTrajectory
```

`Impl::initialize` builds one `HumanoidIK` and hands it to the planner. The planning calls, `brake`, `setLeftArm` and `plan_failure` all forward to that one `VisualServoPlanner`.

## Controllers and planners

Two small hierarchies sit under the facade. Both share the IK solver by `shared_ptr`.

```mermaid
classDiagram
  class ArmController {
    +ArmController(Model)
    +update(JointState)
    +get_current_left_ee_vel() VectorXd
    +get_current_right_ee_vel() VectorXd
    +get_current_left_ee_error() double
    +get_current_right_ee_error() double
    +control_no_arms(JointState) JointState
    #compute_ee_error(...)
  }
  class ImpedanceController {
    +control_both_arms(...) JointState
    +control_left_arm(...) JointState
    +control_right_arm(...) JointState
    +set_nullspace_target(q_d)
    +reset()
    +Kp_linear, Kd_linear, Kp_nullspace
  }
  class IKTrajTracker {
    +control_both_arms(...) JointState
    +control_left_arm(...) JointState
    +control_right_arm(...) JointState
  }
  class JointSpacePlanner {
    +brake(state, ...)$ JointTrajectory
    +planTrajectory(start_state, goal_pose, ...) JointTrajectory
    +planPath(start_state, goal_pose)
    +setLeftArm(bool)
    +failure() string
    #runRRTStar(start_q, goal_q, path) bool
    #timeParameterize(waypoints, ...) JointTrajectory
    #solveGoalConfig(pose, start, passes) JointState
    #motionValid(a, b) bool
  }
  class VisualServoPlanner {
    +planPath(start_state, goal_pose, final_leg_step, retreat)
    +planTrajectory(start_state, goal_pose, final_leg_steps, retreat, ...) JointTrajectory
    +replanTrajectory(start_state, goal_pose, final_leg_steps, retreat, ...) JointTrajectory
    +planFinalLegStep(start_state, goal_pose, step, final_leg_steps, retreat, ...) JointTrajectory
    #computeApproachLeg(goal_pose, leg_size)
    #closestOnPath(q)
  }
  class RRTData {
    OMPL RRTstar planner
  }
  class HumanoidIK {
    +solve_ik(left_wrist, right_wrist, ...) JointState
    +solve_ik(wrist, left, ..., use_dls) JointState
    +check_collision(q) bool
    +set_active_joints(joint_moves)
    +reset()
  }
  class PolynomialTrajectoryGenerator {
    +planTrajectory(goal_pose, start_pose, ...) vector~MatrixXd~
    +planPath(goal_pose, start_pose, ..., steps) vector~MatrixXd~
  }
  ArmController <|-- ImpedanceController
  ArmController <|-- IKTrajTracker
  JointSpacePlanner <|-- VisualServoPlanner
  JointSpacePlanner *-- RRTData : rrt_
  JointSpacePlanner o-- HumanoidIK : ik_handle_
  IKTrajTracker o-- HumanoidIK : ik_handle_
```

`HumanoidIK` solves with CasADi and checks self-collision with pinocchio geometry. `PolynomialTrajectoryGenerator` is compiled into the library, but no node or class uses it.

## Body attitude EKF

The `body_ekf` node owns one `OnlineBodyEstimator`. It resamples the IMUs on a grid and feeds the filter.

```mermaid
classDiagram
  class OnlineBodyEstimator {
    +add_imu(body, t, gyro, acc)
    +process() ProcessReport
    +estimator() BodyAttitudeEstimator
    +reset()
  }
  class BodyAttitudeEstimator {
    +initialize(still, init)
    +initialized() bool
    +reset()
    +step(GridSample) Matrix4d
    +calibration() StillCalibration
  }
  class Body2EKF {
    +propagate(gyro_p, gyro_t, gyro_s, dt)
    +gravity(body, up, sig)
    +waist(W, sig)
    +leg(L, sig_tilt, sig_yaw)
    +R_p, R_t, P
  }
  class StillPeriodCollector {
    +add(sample, speed) bool
    +samples()
    +clear()
  }
  class SampleBuffer {
    +append(t, row)
    +covers(t, margin) bool
    +boxcar(t, window) VectorXd
    +interp(t) VectorXd
  }
  class RelationLookup {
    function of t to Relations
  }
  OnlineBodyEstimator *-- BodyAttitudeEstimator : estimator_
  OnlineBodyEstimator *-- StillPeriodCollector : collector_
  OnlineBodyEstimator *-- "3" SampleBuffer : pelvis_, torso_, camera_
  OnlineBodyEstimator o-- RelationLookup : lookup_
  BodyAttitudeEstimator *-- Body2EKF : optional ekf_
```

The node supplies `RelationLookup` as a TF lookup of the waist and leg relations. The still-period collector decides when calibration ends and the EKF starts.

## ROS nodes

Each node owns its own `Humanoid` instances. `MoveToPoseServer` holds two, so planning and control never share pinocchio data.

```mermaid
classDiagram
  class MoveToPoseServer {
    node move_to_pose_server
    -planner_ Humanoid
    -controller_model_ Humanoid
    -worker_ thread
    -on_goal_() / on_cancel_() / on_accepted_()
    -run_() / monitor_() / try_replan_()
    -step_final_leg_() / check_reach_()
    -control_tick_() / install_() / brake_locked_()
  }
  class IkServer {
    node ik_server
    -planner_ Humanoid
    -check_(request, response)
  }
  class BodyEKFNode {
    node body_ekf
    -estimator_ OnlineBodyEstimator
  }
  class TrajectoryControllerActionServer {
    node trajectory_controller_action_server
    -arm_handle_ Humanoid
  }
  class PlanTrajectoryServer {
    node plan_trajectory_server
    -arm_handle_ Humanoid
  }
  class VisualServo {
    node visual_servoing
    -arm_handle_ Humanoid
  }
  class JointTrajPublisher
  class ImpedeArms
  class GravFF
  class JointStatePublisher
  class Humanoid
  class OnlineBodyEstimator
  MoveToPoseServer *-- "2" Humanoid
  IkServer *-- Humanoid
  TrajectoryControllerActionServer *-- Humanoid
  PlanTrajectoryServer *-- Humanoid
  VisualServo *-- Humanoid
  JointTrajPublisher *-- Humanoid
  ImpedeArms *-- Humanoid
  GravFF *-- Humanoid
  JointStatePublisher *-- Humanoid
  BodyEKFNode *-- OnlineBodyEstimator
```

The default node names above come from the constructors. Launch files rename some: `joint_controller` runs as `trajectory_controller`, and `plan_trajecory` as `plan_trajectory_server`.

## MoveToPoseServer threads and locks

Three mutexes split the move-to-pose node. The worker thread plans, the 100 Hz timer commands, and action callbacks only hand over goals.

```mermaid
flowchart LR
  subgraph CB["Executor callbacks"]
    G["on_goal_, on_accepted_, on_cancel_"]
    T["on_live_target_"]
    F["feedback subscription"]
    TICK["control_tick_ timer"]
  end
  W["worker_ thread<br/>run_, monitor_, replan, final leg"]
  GM[("goal_mutex_<br/>active_, pending_")]
  TM[("target_mutex_<br/>live_target_")]
  CM[("ctrl_mutex_<br/>state_, mode_, trajectory")]
  G --> GM
  T --> TM
  F --> CM
  TICK --> CM
  W --> GM
  W --> TM
  W --> CM
  W -.->|"plans with planner_, no lock held"| P["planner_ Humanoid"]
  TICK -.->|"feedforward torque"| C["controller_model_ Humanoid"]
```

Helpers ending in `_locked_` expect `ctrl_mutex_`. A cancel clears the trajectory under both goal and control locks, so commands stop at once.

## Python nodes

The Python side has four nodes. Each owns its helpers directly.

```mermaid
classDiagram
  class VisualServoArmNode {
    node humanoid_arm_vs
    -_execute_cb(goal_handle)
    -_send_motion_goal()
    -_cancel_motion_goal()
    -_request_target(target)
    +ee_pose(left_arm) PoseStamped
  }
  class PoseFilterNode {
    node pose_filter
    -_pose_stamped_cb(msg)
    -_update_kalman(...)
    -_update_filter(...)
    -_publish_pose(header)
  }
  class TargetKF {
    +reset()
    +initialized bool
    +measurement_covariance(R) ndarray
    +update(t, z_p, z_n, R) tuple
  }
  class GripperPoseEstimator {
    node gripper_pose_estimator
    -_work()
    -_fit(arm, msg)
  }
  class GripperFit {
    +roi(k, camera_from_base, width, height)
    +fit(depth_mm, k, camera_from_base, start)
  }
  class NoisyPosePublisher {
    +publish()
  }
  PoseFilterNode *-- TargetKF : target_kf
  GripperPoseEstimator *-- "per arm" GripperFit : fitter
  VisualServoArmNode ..> PoseFilterNode : /reset_filter
```

The press launch names the filter node `arm_pose_filter`; its constructor default is `pose_filter`. `GripperFit` runs Warp kernels on the GPU from the estimator's worker thread.
