# Body attitude EKF

`body_ekf.{h,cpp}` estimates pelvis and torso orientation from the pelvis and torso IMUs (optionally the camera IMU) and the leg and waist encoders. The `body_ekf` node publishes `foot_world -> sensor` so every live TF frame resolves in a gravity-aligned frame tied to the left foot.

## File map

| File | Responsibility |
|---|---|
| `body_ekf.h` | Declares `Body2EKF`, the per-robot `BodyConfig`, `BodyAttitudeEstimator`, the resampling buffer, the still collector and `OnlineBodyEstimator`. |
| `body_ekf.cpp` | Implements the 15-state multiplicative EKF, still-period calibration and on-line resampling on a 200 Hz grid (camera-IMU or torso-IMU stamps). Kinematics come from `Humanoid::frame_transform` on the full model. |

## Node

`src/body_ekf.cpp` reads a `sensor_msgs/JointState` topic (`/feedback`) and `sensor_msgs/Imu` topics for the pelvis and torso. With `use_camera_imu: true` it also fuses the camera IMU. It publishes `foot_world -> tf_root_frame` at 200 Hz and the torso pose on `/body_ekf/torso_pose`. Per-robot values live in `config/body_ekf_{stanley,patricia}.yaml`.

## Testing

`test_body_ekf` covers the filter, `Humanoid` kinematics, buffers and an on-line run on a still robot. `body_ekf_replay` runs exported bag data through the core for comparison with the Python reference.
