# Move-to-pose node

The node owns goal handling, planning, and trajectory execution while accessing the core exclusively through `g1_pilot.h`. `src/move_to_pose.cpp` starts the node and executor.

## File map

| File | Responsibility |
|---|---|
| `server.h` | Declares the node, ROS message aliases, control states, helper methods, and mutex ownership. |
| `server.cpp` | Creates ROS interfaces and models; handles goals, cancellation, and live targets. |
| `planning.cpp` | Runs the planning worker, monitors actions, transforms frames, and shares trajectory generation between initial plans and replans. |
| `control.cpp` | Installs trajectories, generates braking through `Humanoid::brake()`, executes samples, and holds the braking endpoint. |

## Execution

Control follows `Tracking → Braking → Holding → Tracking` when a live target moves. The worker plans from the held endpoint; a failed replan keeps the hold, `max_plan_failures` in a row end the goal, and a successful replan installs the replacement.

## Concurrency

Separate mutexes protect goal ownership, target updates, and control state; `_locked_` helpers require the control mutex. Planning runs without those locks, and separate `Humanoid` instances serve planning and control.

## Goal lifecycle

A newer goal brakes the previous motion before it plans. The standard action cancel clears control commands at once; a cancel request with a zero goal id cancels every goal. A failed goal brakes and then stops commanding, as a finished one does. Installation checks ownership and cancellation before replacing a trajectory.

## Frames

Goal poses transform into pelvis coordinates for the core; published paths and end-effector poses use `planning_frame`. Target conversion retains the existing latest-TF behavior.

The final leg freezes its goal in `final_leg_anchor_frame`, read from TF before every step. The robot launch sets it to the body EKF's `foot_world`, which holds still while the body leans into a press.

## Target offset

`arm_safety_distance_{x,y,z}` defaults to zero metres and translates each target along its local axes. Shared target conversion applies it once to the initial action pose and every streamed update before drift checks and planning.
