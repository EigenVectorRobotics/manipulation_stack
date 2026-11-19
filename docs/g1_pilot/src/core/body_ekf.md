# Body attitude EKF

`body_ekf.{h,cpp}` estimates pelvis and torso orientation from the pelvis and torso IMUs (optionally the camera IMU) and the leg and waist encoders. The `body_ekf` node publishes `torso_link -> foot_world`, so every live TF frame resolves in a gravity-aligned frame tied to the left foot.

## File map

| File | Responsibility |
|---|---|
| `body_ekf.h` | Declares `Body2EKF`, the per-robot `BodyConfig`, `BodyAttitudeEstimator`, the resampling buffer, the still collector and `OnlineBodyEstimator`. |
| `body_ekf.cpp` | Implements the 15-state multiplicative EKF, still-period calibration and on-line resampling on a 200 Hz grid (camera-IMU or torso-IMU stamps). Leg and waist relations come from a lookup at each grid stamp, and their rate of change detects the still period. |

## Node

`src/body_ekf.cpp` reads `sensor_msgs/Imu` topics for the pelvis and torso and looks up `pelvis <- torso_link` and `<foot> <- pelvis` on TF from `robot_state_publisher`, whose `joint_state_publisher` must run at 200 Hz. With `use_camera_imu: true` it also fuses the camera IMU. It publishes `torso_link -> foot_world` at 200 Hz and the torso pose on `/body_ekf/torso_pose`. Per-robot values live in `config/body_ekf_<robot>.yaml`; only `body_ekf_stanley.yaml` exists.

## Testing

`test_body_ekf` covers the filter, TF-URDF kinematics, the relation speed, buffers and on-line runs with a lagging encoder TF. `body_ekf_replay` runs exported bag data through the core, with relations from FK on the recorded joints.
