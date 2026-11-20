# g1_pilot/src/core/arm_ik

Inverse kinematics for one or both arms. The solver is shared by the planner and the IK tracker.

- **`robot_arm_ik.h`, `robot_arm_ik.cpp`**: `HumanoidIK` solves arm IK as a CasADi optimisation, with an optional damped-least-squares path. It checks self-collision on pinocchio geometry and can freeze joints.
- **`arm_ik.h`**: An umbrella header that includes `robot_arm_ik.h`.
- **`README.md`**: The older library usage notes, listed under Guides.
