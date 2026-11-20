# g1_description

The G1 robot model and its viewer launch. Every C++ node loads its URDF from this package's share folder.

- **`assets/g1/*.urdf`**: The 29-DoF G1 with Dex1 grippers; the `_ros` and `_ros_ctrl_viz` variants feed the feedback and control views.
- **`assets/g1/meshes/`**: Visual and collision meshes.
- **`assets/g1/dex1_1/`**: The Dex1 gripper's license and notice.
- **`launch/display.launch.py`**: Two state publishers, one on `/feedback` and one on `/position_control` under `control/`, plus RViz.
- **`rviz/manipulation.rviz`**: Shows the robot, `/traj` and the filtered `arm_upright` target.
