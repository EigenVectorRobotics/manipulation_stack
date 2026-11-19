# g1_pilot/src/core

The library behind `Humanoid`. The top-level files load the model; each subfolder holds one solver family.

- **`humanoid.cpp`**: `Humanoid` and its private `Impl`, which owns the pinocchio models, the IK solver, the impedance controller and the planner.
- **`io.cpp`**: Reads `g1.yaml` into the model setup and converts between `JointState` and pinocchio vectors.
- **`interfaces.h`**: Declares the `JointState` conversion helpers for pinocchio models.
- **`arm_ik/`**: The CasADi IK solver with self-collision checks. See [arm_ik](core/arm_ik.md).
- **`arm_control/`**: Task-space controllers. See [arm_control](core/arm_control.md).
- **`arm_mp/`**: Joint-space and approach planners. See [arm_mp](core/arm_mp.md).
- **`body_ekf/`**: The body attitude EKF. See [body_ekf](core/body_ekf.md).
