import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import rclpy
import rosbag2_py
from geometry_msgs.msg import PoseStamped, Quaternion
from rclpy.serialization import serialize_message
from scipy.spatial.transform import Rotation

from humanoid_arm_skills import pose_filtering, utils
from offline_pose_filter import filter_poses, read_inputs


def orientation_from_normal(normal: np.ndarray) -> np.ndarray:
    x = normal / np.linalg.norm(normal)
    up = np.array([0., 0., 1.]) if abs(x[2]) < .99 else np.array([0., 1., 0.])
    z = up - np.dot(up, x) * x
    z /= np.linalg.norm(z)
    return Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])).as_quat()


def wall_poses(messages, wall):
    result = deepcopy(messages)
    for message in result:
        stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
        q = orientation_from_normal(np.array(wall[stamp]['normal']))
        message.pose.orientation = Quaternion(**dict(zip(('x', 'y', 'z', 'w'), q.tolist())))
    return result


def pose_message(row: dict) -> PoseStamped:
    message = PoseStamped()
    message.header.frame_id = row['frame']
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(row['stamp_ns'], 10**9)
    message.pose.position.x, message.pose.position.y, message.pose.position.z = row['position']
    message.pose.orientation = Quaternion(**dict(zip(('x', 'y', 'z', 'w'), row['quaternion'])))
    return message


def save_variant(root, outputs, metadata):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'trajectory.json').write_text(json.dumps(dict(metadata=metadata, outputs=outputs), indent=2))
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(root / 'bag'), storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    topics = {kind: '/analysis/button_' + kind for kind in ('filtered', 'approach', 'upright')}
    for index, topic in enumerate(topics.values()):
        writer.create_topic(rosbag2_py.TopicMetadata(id=index, name=topic,
                            type='geometry_msgs/msg/PoseStamped', serialization_format='cdr'))
    for row in outputs:
        writer.write(topics[row['kind']], serialize_message(pose_message(row)), row['stamp_ns'])
    del writer


def main():
    # (1) Read the observation poses, recorded transforms, and matching wall normals.
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    parser.add_argument('wall_normals')
    parser.add_argument('output_dir')
    parser.add_argument('--input-topic', default='/track_3d/selected_normal_pose')
    parser.add_argument('--start-time', type=float, default=0.0)
    args = parser.parse_args()
    wall = {r['stamp_ns']: r for r in json.loads(Path(args.wall_normals).read_text())}
    root = Path(args.output_dir)
    rclpy.init(args=['--ros-args', '-p', 'target_frame_arm:=left_ankle_roll_link', '-p', 'tf_timeout:=0.0'])
    buffer, messages = read_inputs(args)
    first_stamp = messages[0].header.stamp.sec * 10**9 + messages[0].header.stamp.nanosec
    variants = {'wall_orientation': wall_poses(messages, wall), 'recorded_orientation': messages}
    # (2) Preserve provenance and observation timing for the saved simulation inputs.
    metadata = dict(source_bag=args.bag, input_topic=args.input_topic, alpha=0.2, outlier_rejection=False,
                    frame='left_ankle_roll_link', position_source='original selected_normal_pose position',
                    quaternion_order='xyzw', position_units='m', source_samples=len(messages),
                    trim_start_offset_s=args.start_time, source_first_stamp_ns=first_stamp,
                    timing='observation timestamps; bag storage times equal header times',
                    code_sha256={Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                 for module in (pose_filtering, utils)},
                    wall_normals_sha256=hashlib.sha256(Path(args.wall_normals).read_bytes()).hexdigest())
    # (3) Call the production filter and export all three output pose types.
    try:
        for name, poses in variants.items():
            outputs = filter_poses(buffer, poses, 0.2)
            assert len(outputs) == 3 * len(messages)
            outputs = [row for row in outputs if (row['stamp_ns'] - first_stamp) / 1e9 >= args.start_time]
            if not outputs:
                raise ValueError('No observations remain after the requested start time')
            save_variant(root / name, outputs, dict(metadata, orientation_source=name, samples=len(outputs) // 3))
            print(f'{name}: {len(outputs) // 3} samples, {len(outputs)} poses -> {root / name}')
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
