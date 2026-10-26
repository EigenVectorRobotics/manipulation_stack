import argparse
import json
from pathlib import Path

import rclpy
import rosbag2_py
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer

from humanoid_arm_skills.pose_filtering import PoseFilterNode


class PoseCollector:
    def __init__(self, kind, rows):
        self.kind = kind
        self.rows = rows

    def publish(self, message: PoseStamped) -> None:
        p, q = message.pose.position, message.pose.orientation
        self.rows.append(dict(
            kind=self.kind,
            stamp_ns=message.header.stamp.sec * 10**9 + message.header.stamp.nanosec,
            frame=message.header.frame_id,
            position=[p.x, p.y, p.z], quaternion=[q.x, q.y, q.z, q.w]))


def read_inputs(args):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag), rosbag2_py.ConverterOptions('', ''))
    duration = reader.get_metadata().duration.nanoseconds * 1e-9 + 1.0
    buffer = Buffer(cache_time=Duration(seconds=duration))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.input_topic, '/tf', '/tf_static']))
    poses = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == args.input_topic:
            poses.append(deserialize_message(data, PoseStamped))
        else:
            setter = buffer.set_transform_static if topic == '/tf_static' else buffer.set_transform
            for transform in deserialize_message(data, TFMessage).transforms:
                setter(transform, 'bag')
    return buffer, poses


def filter_poses(buffer, poses, alpha, filter_type='ema'):
    node = PoseFilterNode(tf_buffer=buffer)
    node.set_parameters([Parameter('alpha', value=alpha), Parameter('filter_type', value=filter_type)])
    rows = []
    node.pub = PoseCollector('filtered', rows)
    node.approach_pub = PoseCollector('approach', rows)
    node.upright_pub = PoseCollector('upright', rows)
    try:
        for pose in poses:
            node._pose_stamped_cb(pose)
    finally:
        node.destroy_node()
    return rows


def run_filter(args):
    buffer, poses = read_inputs(args)
    stamps = set(json.loads(Path(args.stamps).read_text())) if args.stamps else None
    poses = [pose for pose in poses if stamps is None
             or pose.header.stamp.sec * 10**9 + pose.header.stamp.nanosec in stamps]
    inputs = [pose.header.stamp.sec * 10**9 + pose.header.stamp.nanosec for pose in poses]
    rows = filter_poses(buffer, poses, args.alpha, args.filter_type)
    Path(args.output).write_text(json.dumps(dict(input_stamps=inputs, outputs=rows)))
    print(f'Processed {len(inputs)} poses; wrote {len(rows)} outputs to {args.output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    parser.add_argument('output')
    parser.add_argument('--input-topic', default='/track_3d/selected_normal_pose')
    parser.add_argument('--frame', default='left_ankle_roll_link')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--filter-type', choices=('ema', 'kalman'), default='ema')
    parser.add_argument('--stamps')
    args = parser.parse_args()
    rclpy.init(args=['--ros-args', '-p', f'target_frame_arm:={args.frame}',
                     '-p', f'alpha:={args.alpha}',
                     '-p', f'filter_type:={args.filter_type}', '-p', 'tf_timeout:=0.0'])
    try:
        run_filter(args)
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
