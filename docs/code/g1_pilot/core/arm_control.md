# g1_pilot/src/core/arm_control

Task-space arm controllers. `Humanoid` uses the impedance controller; the IK tracker is an alternative.

- **`base_controller.h`, `arm_control.cpp`**: `ArmController` keeps joint state, end-effector poses, velocities and Jacobians, and computes end-effector errors.
- **`impedance_control.h`, `impedance_control.cpp`**: `ImpedanceController` computes joint torques for one or both arms, with null-space stiffness and a torque rate limit.
- **`ik_traj_tracker.h`, `ik_traj_tracker.cpp`**: `IKTrajTracker` turns end-effector errors into IK targets through the shared `HumanoidIK`.
- **`arm_control.h`**: An umbrella header for both controllers.
