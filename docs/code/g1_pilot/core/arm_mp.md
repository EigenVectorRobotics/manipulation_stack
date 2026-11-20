# g1_pilot/src/core/arm_mp

Arm motion planning. `Humanoid` owns one `VisualServoPlanner`, which every plan, replan and brake goes through.

- **`joint_space_planner.h`, `joint_space_planner.cpp`**: `JointSpacePlanner` plans with OMPL RRT* in joint space and time-parameterises the path. Its static `brake` stops all joints at the earliest shared sample.
- **`visual_servo_planner.h`, `visual_servo_planner.cpp`**: `VisualServoPlanner` adds a pre-grasp 0.10 m back from the goal and a straight approach leg. It sets `lock_time`, reuses cached paths and plans single final-leg steps.
- **`poly_traj_gen.h`, `motion_planner.cpp`**: `PolynomialTrajectoryGenerator` builds quintic SE(3) trajectories; nothing in the library calls it.
- **`arm_mp.h`**: An umbrella header for both planners.
