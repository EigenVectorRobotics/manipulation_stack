# humanoid_manipulation_interfaces

The action and service definitions of the arm stack. [System](../design/system.md) lists who serves and calls each.

- **`action/VisualServoingArm.action`**: Press a target named by text with one arm; returns the closest distance and orientation error.
- **`action/MoveToPose.action`**: Plan to a stamped pose and track it; an optional `target_topic` streams target updates.
- **`action/ControlJointTrajectory.action`**: Track a given joint trajectory with one arm.
- **`srv/CheckReach.srv`**: Whether the arm reaches a target as `move_to_pose` would plan it, with the miss in pelvis and target axes.
- **`srv/PlanJointTrajectory.srv`**: Plan a joint trajectory between a start and a target pose.

The MoveToPose comment says target updates are only stored; `move_to_pose_server` does replan from them.
