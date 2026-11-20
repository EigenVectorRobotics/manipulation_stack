# g1_pilot/test

Tests and tools for the library, built when `BUILD_TESTING` is on. Only the suites named in `CMakeLists.txt` build.

- **`arm_ik/test_ik.cpp`**: The `test_ik` program; it exercises the IK solver.
- **`arm_control/test_control.cpp`**: The `test_control` program; it exercises the controllers.
- **`arm_mp/test_brake.cpp`**: GTest suite for `brake`: every joint stops at one sample within the limits.
- **`arm_mp/benchmark_brake.cpp`**: Times `brake` on random states.
- **`arm_mp/test_traj_gen.cpp`**: The `test_mp_ik` program; it plans trajectories on the G1 model.
- **`arm_mp/test_joint_traj.cpp`, `line_unittest.cpp`, `math_unittest.cpp`, `line_viz.py`**: Older planner tests and a plot script; `CMakeLists.txt` builds none of them.
- **`body_ekf/test_body_ekf.cpp`**: GTest suite for the filter, calibration, buffers and on-line runs.
- **`body_ekf/replay_body_ekf.cpp`**: The `body_ekf_replay` tool; it runs exported bag data through the estimator.
