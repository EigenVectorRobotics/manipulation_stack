# g1_pilot/src/move_to_pose

The MoveToPose action server, split by concern. [Move-to-pose node](../../g1_pilot/src/move_to_pose.md) explains its execution and locking.

- **`../move_to_pose.cpp`**: Creates `MoveToPoseServer` and spins it on a multi-threaded executor.
- **`server.h`**: Declares the node, its control modes, goal states and the three mutexes with what each protects.
- **`server.cpp`**: Declares parameters, builds the two `Humanoid` models and the ROS interfaces, and handles goals, cancels and the live target topic.
- **`planning.cpp`**: Runs the worker thread: start state, planning, reach check, replanning, final-leg stepping and failure rules.
- **`control.cpp`**: Runs the 100 Hz control tick: installs, brakes, holds and tracks trajectories with feedforward torque.
