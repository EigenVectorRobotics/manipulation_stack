# Dex1-1 gripper assets

The three `g1_29dof_with_hand_rev_1_0*.urdf` files use Dex1-1 parallel grippers. Their legacy filenames and robot name are retained because the launch files and pilot nodes reference them directly. Body links, joints, collision geometry, limits and inertias retain their previous definitions. The control visualization retains its translucent purple material.

## Model and coordinates

Each gripper is attached to its wrist yaw link at `xyz="0.0415 0 0"`, with no rotation. The upstream link inertias, visual meshes, collision meshes, joint axes and limits are preserved. The five mesh files are installed from the adjacent `../meshes/` directory.

Each side has a `*_dex1_base_link` and two `*_dex1_finger_link_1/2` links. The `*_dex1_finger_joint_1/2` joints are prismatic, with opposite Y axes. Both positions use metres: increasing both positions opens the gripper symmetrically. Their position limits are `[-0.020, 0.0245]`, effort limit is `20 N`, and velocity limit is `0.2 m/s`. These are the portable source URDF limits; the MuJoCo repository's optional `-0.023 m` closure adjustment is not applied here. At the lower source limit, the supplied collision meshes still have approximately `5.88 mm` between their inner faces.

The source model represents both fingers independently, without a mimic joint. This preserves its 29 body plus 4 finger coordinates. Real Dex1-1 hardware uses one motor per gripper, so these two model coordinates are not two independently controllable hardware motors. A hardware bridge must map motor position to symmetric finger displacement; motor radians must not be copied directly into these metre-valued joints. The official service publishes on `rt/dex1/{left,right}/{cmd,state}`. Updating this description does not implement that bridge or modify the simulator's DDS transport.

`g1_pilot/config/g1.yaml` locks the new finger joints in both existing arm profiles. Its TCP is the midpoint between the distal fingertip ends: `[0.1847585, 0, 0]` relative to each wrist yaw joint. This is a mesh-derived convention, from the `0.0415 m` mount plus the finger meshes' maximum X coordinate of `0.1432585 m`, rather than a calibrated hardware grasp frame.

## Sources and attribution

- Model and meshes: [lerobot/unitree-g1-mujoco](https://huggingface.co/lerobot/unitree-g1-mujoco/tree/68459ed68f6f68e1f661091dfcb6ebce44681aec), revision `68459ed68f6f68e1f661091dfcb6ebce44681aec`; `assets/g1_29dof_with_dex1_1.urdf` and `assets/dex1_1/meshes/`.
- Hardware service: [unitreerobotics/dex1_1_service](https://github.com/unitreerobotics/dex1_1_service).
- The model repository credits [Hxxxz0/HIW-500-controoler](https://github.com/Hxxxz0/HIW-500-controoler). Its accompanying `LICENSE` and `NOTICE` are retained in this directory. As that notice states, the Unitree model assets retain their applicable upstream terms and are not relicensed by HIW-500's Apache 2.0 license or this package's MIT declaration.
