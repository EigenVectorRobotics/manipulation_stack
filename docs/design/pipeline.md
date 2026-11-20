# Press pipeline

How a press goal becomes arm motion: the target filter, the skill's seeding, the planned trajectory's shape and the control modes of `move_to_pose_server`. Read [System](system.md) first for the topic names.

## Target path

The tracker's surface pose reaches the planner through one filter. The filter moves it into `foot_world`, smooths it and publishes three variants.

```mermaid
flowchart LR
  TRK["track_3d<br/>/track_3d/selected_normal_pose<br/>camera frame"]
  TF["TF lookup at the pose's stamp<br/>into target_frame_arm = foot_world"]
  KF["TargetKF<br/>position random walk, normal random walk,<br/>NIS gate 16.27"]
  EMA["EMA, alpha 0.2"]
  OUT["/track_3d/selected_normal_pose_filtered/arm"]
  UP[".../arm_upright<br/>yaw only, turned 180 deg"]
  AP[".../arm_approach<br/>normal reversed"]
  TRK --> TF
  TF -->|"filter_type kalman (launch default)"| KF
  TF -->|"filter_type ema"| EMA
  KF --> OUT
  EMA --> OUT
  OUT --> UP
  OUT --> AP
```

A pose without a stamp, a TF miss or an outlier is dropped. A gap over `lost_timeout` (5 s), a frame change or a `/reset_filter` message restarts the filter.

## Skill goal and seeding

`humanoid_arm_vs` accepts one goal at a time. It asks the tracker for the target by text, then waits for a fresh track before it trusts any pose.

```mermaid
stateDiagram-v2
  [*] --> reset: goal arrives, publish /track_3d/reset, then the text on /keyboard_input
  reset --> seeding: /track_3d/is_tracking false
  seeding --> tracking: /track_3d/is_tracking true
  tracking --> tracking: filtered pose stored as last_goal
  reset --> reset: seed_timeout 30 s, ask again (2 requests at most)
  seeding --> reset: seed_timeout 30 s, ask again
```

Poses count only in `tracking` while `require_seed` is true. The first stored pose triggers the single MoveToPose goal.

## Skill goal outcomes

The skill loop runs at `goal_rate` (2 Hz) and ends the goal on the first matching rule. Every exit cancels all `move_to_pose` goals and resets the tracker.

```mermaid
flowchart TB
  L["loop at 2 Hz"]
  C{"cancel requested?"}
  S{"move_to_pose succeeded?"}
  MF{"motion failure:<br/>rejected, failed, send error?"}
  SEED{"no seed after<br/>seed_max_requests?"}
  FP{"no pose within<br/>first_pose_timeout 10 s?"}
  SV{"server missing 5 s before,<br/>or lost 2 s during?"}
  TO{"over action_timeout 600 s?"}
  FB["publish feedback:<br/>distance, orientation error, tracking_lost"]
  L --> C
  C -->|yes| CAN["canceled"]
  C -->|no| S
  S -->|yes| OK["succeeded, closest approach as result"]
  S -->|no| MF
  MF -->|yes| AB["aborted"]
  MF -->|no| SEED
  SEED -->|yes| AB
  SEED -->|no| FP
  FP -->|yes| AB
  FP -->|no| SV
  SV -->|yes| AB
  SV -->|no| TO
  TO -->|yes| AB
  TO -->|no| FB
  FB --> L
```

The result reports the closest approach of `left_ee` or `right_ee` to the goal. A press ends with a retreat, so the last pose would understate the reach.

## Planned trajectory

`VisualServoPlanner` plans in joint space to a pre-grasp, then walks a straight approach leg to the goal. The target locks once the approach leg starts.

```mermaid
flowchart LR
  S["start<br/>held or current joints"]
  PG["pre-grasp<br/>goal moved 0.10 m back along its x"]
  G["goal<br/>pelvis frame, plus press offset"]
  R["retreat<br/>approach leg reversed"]
  S -->|"OMPL RRT*, max_joint_vel"| PG
  PG -->|"pause, then final_leg_steps DLS IK poses<br/>half speed, quarter acceleration"| G
  G -.->|"last final-leg step only"| R
```

`lock_time` marks the start of the approach leg. `replanTrajectory` reuses the cached path and recomputes only the approach leg when the goal moved a little and the arm stayed near the path.

## Control modes of move_to_pose

The 100 Hz control timer acts on one mode. The worker thread moves the node between modes as targets update and steps finish.

```mermaid
stateDiagram-v2
  [*] --> Idle
  Idle --> Tracking: install_ a plan
  Tracking --> Braking: target moved, newer goal, or failure
  Braking --> Holding: brake ends
  Braking --> Idle: brake ends after a failed goal
  Holding --> Tracking: replan or next final-leg step installed
  Tracking --> Holding: final-leg step ends, hold_between_steps_
  Tracking --> Complete: last sample reached
  Complete --> [*]: goal succeeds or fails on joint error
  Tracking --> Idle: cancel
  Holding --> Idle: cancel or failure
```

`Tracking` and `Braking` publish the current trajectory sample with inverse-dynamics feedforward torque. `Holding` republishes the endpoint with zero velocity.

## Replanning and the final leg

The worker checks the live target every `feedback_period` (0.1 s). What it does depends on where the arm is in the plan.

```mermaid
flowchart TB
  U["latest pose on target_topic"]
  I{"replan_min_interval<br/>0.25 s passed?"}
  LT{"tracking and past lock_time?"}
  ST{"final_leg_stepping?"}
  STEP["start stepping: freeze the goal<br/>in left_ankle_roll_link"]
  ONCE["replan the final leg once"]
  MOVED{"holding, or moved 1 cm or 0.05 rad<br/>since the last plan?"}
  RP["brake, hold, replan from the held endpoint"]
  SK["keep the current plan"]
  U --> I
  I -->|no| SK
  I -->|yes| LT
  LT -->|yes| ST
  ST -->|yes| STEP
  ST -->|no| ONCE
  LT -->|no| MOVED
  MOVED -->|yes| RP
  MOVED -->|no| SK
```

Each of the 10 final-leg steps waits for the hold and a 0.5 s settle. It re-maps the frozen goal into the current pelvis and subtracts the depth-fitted gripper error.

## Failure rules of move_to_pose

A goal fails fast when its inputs go stale or the arm stops following. Near the target, a large joint error counts as pressing and is allowed.

| Rule | Parameter (default) | Effect |
|---|---|---|
| Joint feedback older than the limit | `feedback_timeout` (0.5 s) | goal aborts at once |
| One joint off its command while tracking | `max_tracking_error` (0.2 rad) for `tracking_error_timeout` (1 s) | goal aborts, unless the tip is within `near_target_distance` (2 cm) |
| Plan ends outside the goal tolerance | `goal_position_tolerance` (5 mm), `goal_orientation_tolerance` (0.1 rad) | first plan aborts; replans count as failures |
| Replan or final-leg step fails repeatedly | `max_plan_failures` (3) | goal aborts |
| Gripper estimate stale or implausible | `gripper_estimate_max_age` (0.5 s), `gripper_estimate_max_error` (3 cm) | correction is skipped |

`ik_server` applies the same plan and tolerance check without moving the arm. The orchestrator's `check_reach` uses it before a press.
