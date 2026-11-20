# humanoid_arm_skills

The Python side of the press: the VisualServoingArm skill, the target filter and the gripper estimator. The [API reference](https://eigenvectorrobotics.github.io/manipulation_stack/reference/humanoid_arm_skills/) has every function.

- **`humanoid_arm_visual_servoing.py`**: `humanoid_arm_vs`, the `/humanoid_arm_visual_servoing` action server; it seeds the tracker and drives one MoveToPose goal.
- **`pose_filtering.py`**: `pose_filtering`, the EMA or Kalman target filter that publishes the `arm`, `arm_upright` and `arm_approach` poses.
- **`target_kf.py`**: `TargetKF`, a Kalman filter on target position and surface normal with an NIS outlier gate.
- **`gripper_pose_estimator.py`**: Fits each gripper mesh to the depth image and publishes `<side>_dex1_base_link_estimate` TF.
- **`gripper_fit.py`**: `GripperFit`, the Warp GPU kernels for the translation-only mesh fit.
- **`noisy_pose_publisher.py`**: Publishes a fixed pose with noise, to test the filter without a tracker.
- **`utils.py`**: Quaternion and pose helpers: distances, averaging, surface normals, approach and upright poses.
- **`launch/`**: `arm_bringup.launch.py` and `humanoid_arm_visual_servoing.launch.py`; see [Launch files](../design/launch.md).
- **`test/`**: Unit tests for `TargetKF` and surface targets, an offline filter run, and scripts that save and replay button trajectories.

`setup.py` installs `meshes/*.STL`, which the estimator needs, but the repo tracks no `meshes/` folder.
