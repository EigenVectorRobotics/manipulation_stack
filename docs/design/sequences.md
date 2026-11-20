# Sequences

How the arm stack starts, how a press goal unfolds and how the orchestrator's reach check and press offset reach the planner. Read [Classes and ownership](classes.md) for the objects named here.

## Startup

`arm_bringup.launch.py` starts the robot interface, the viewer bridge and the press stack. Launch starts the actions of a file in list order, and each node then initialises on its own.

```mermaid
sequenceDiagram
  participant L as arm_bringup.launch.py
  participant UI as unitree_interface
  participant FX as foxglove_bridge
  participant AV as humanoid_arm_visual_servoing.launch.py
  participant D as display.launch.py
  participant E as body_ekf
  participant M as move_to_pose_server
  participant I as ik_server
  participant F as arm_pose_filter
  participant V as humanoid_arm_vs
  L->>UI: include unitree_interface.launch.py
  L->>FX: start on foxglove_port 8765
  L->>AV: include with PublishTF
  AV->>D: include when PublishTF is true
  AV->>E: include body_ekf.launch.py when body_ekf is true
  AV->>M: include move_to_pose.launch.py, planning_frame foot_world
  AV->>I: same include, service /check_reach
  AV->>F: start, input track_3d/selected_normal_pose
  AV->>V: start, filtered_pose_topic .../arm_upright
  M->>M: load two Humanoid models, send left_ee and right_ee static TF
  M->>M: start the worker thread and the 100 Hz control timer
  I->>I: load one Humanoid model, serve /check_reach
  E->>E: wait for a still period, then publish torso_link to foot_world
```

`move_to_pose_server` refuses to plan until it has fresh `/feedback`. Arm targets need `foot_world`, so `body_ekf` must calibrate before the first press.

## A press goal

The orchestrator sends one VisualServoingArm goal. The skill seeds the tracker, hands one MoveToPose goal to the planner and waits for its result.

```mermaid
sequenceDiagram
  participant O as orchestrator press_button
  participant V as humanoid_arm_vs
  participant T as track_3d
  participant F as arm_pose_filter
  participant M as move_to_pose_server
  participant U as unitree_interface
  O->>V: goal target_description, left_arm
  V->>T: /track_3d/reset, then /keyboard_input text
  T-->>V: /track_3d/is_tracking false
  V->>F: /reset_filter true
  T-->>V: /track_3d/is_tracking true
  T->>F: selected_normal_pose
  F-->>V: arm_upright pose
  F-->>M: arm_upright pose (after the goal names it)
  V->>M: MoveToPose goal: pose, left_arm, target_topic arm_upright
  M->>M: take a start at rest, transform to pelvis, add press offset
  M->>M: plan, check reach, install
  loop 100 Hz
    M->>U: /position_control sample with feedforward torque
    U-->>M: /feedback
  end
  loop every 0.1 s
    M->>M: replan or step the final leg on target updates
    M-->>V: feedback tracking, current_error, waypoints
    V-->>O: feedback distance, orientation error
  end
  M-->>V: result success, final_error
  V->>M: cancel all goals
  V->>T: /track_3d/reset
  V-->>O: result success, final_distance, orientation_error_deg
```

A failed MoveToPose result aborts the press with its message. The cancel-all request stops any motion the skill started, even one whose acceptance it never saw.

## The reach check

The orchestrator asks `ik_server` whether the arm reaches a target. The service plans as `move_to_pose_server` would and commands nothing.

```mermaid
sequenceDiagram
  participant O as orchestrator check_reach
  participant I as ik_server
  participant H as Humanoid planner_
  O->>I: CheckReach target_pose, left_arm
  I->>I: reject when /feedback is missing or older than 0.5 s
  I->>I: transform the target to pelvis, add ik_server's own offset
  I->>H: setLeftArm, planTrajectory from the current joints
  H-->>I: trajectory, or empty with plan_failure()
  I->>H: forward_kinematics on each sample
  I-->>O: reachable, position_error, orientation_error, error in pelvis and target axes, target_in_pelvis, message
```

`reachable` uses the same 5 mm and 0.1 rad tolerances as `move_to_pose_server`. `ik_server` keeps its launch-time offset, so the orchestrator adds the press offset to the target itself.

## Setting the press offset

The orchestrator sets three parameters on `move_to_pose_server`. The node reads them at every plan, so the next press uses them.

```mermaid
sequenceDiagram
  participant O as orchestrator set_press_offset
  participant M as move_to_pose_server
  O->>M: /move_to_pose_server/set_parameters arm_safety_distance_x, _y, _z
  M-->>O: SetParameters results
  Note over M: to_pelvis_frame_ translates each target by (x, y, z) in its own axes
  Note over M: a change during final-leg stepping moves the frozen goal by the difference
```

The values live only in the running node. A relaunch resets them to the launch arguments, which default to zero.
