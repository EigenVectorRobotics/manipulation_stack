import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PosePath
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from humanoid_manipulation_interfaces.action import VisualServoingArm
from save_button_trajectory import pose_message


def await_future(node, future, timeout):
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done():
        raise TimeoutError('ROS request timed out')
    return future.result()


def observe(node, records):
    start = time.monotonic()

    def receive(message, topic):
        row = dict(time_s=time.monotonic() - start, topic=topic)
        if isinstance(message, PoseStamped):
            p, q = message.pose.position, message.pose.orientation
            row.update(frame=message.header.frame_id, position=[p.x, p.y, p.z], quaternion=[q.x, q.y, q.z, q.w])
        elif isinstance(message, JointState):
            row.update(names=list(message.name), position=list(message.position), velocity=list(message.velocity))
        else:
            row['waypoints'] = len(message.poses)
        records.append(row)

    subscriptions = []
    for topic, kind in [('/left_ee_pose', PoseStamped), ('/right_ee_pose', PoseStamped),
                        ('/feedback', JointState), ('/position_control', JointState), ('/traj', PosePath)]:
        subscriptions.append(node.create_subscription(kind, topic, lambda msg, topic=topic: receive(msg, topic), 100))
    return subscriptions


def stream(node, rows, publishers, tracking_pub):
    start = time.monotonic()
    first_stamp = rows[0]['stamp_ns']
    for row in rows:
        deadline = start + (row['stamp_ns'] - first_stamp) / 1e9
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=max(0.0, min(.01, deadline - time.monotonic())))
        message = pose_message(row)
        message.header.stamp = node.get_clock().now().to_msg()
        publishers[row['kind']].publish(message)
        if row['kind'] == 'approach':
            tracking_pub.publish(Bool(data=True))
        rclpy.spin_once(node, timeout_sec=0.0)


def run(args, node, report):
    # (1) Connect the real visual-servo action to replayed perception outputs.
    data = json.loads(Path(args.trajectory).read_text())
    first = data['outputs'][0]['stamp_ns']
    rows = [r for r in data['outputs'] if (r['stamp_ns'] - first) / 1e9 >= args.start_time]
    publishers = {kind: node.create_publisher(PoseStamped, '/analysis/button_' + kind, 10)
                  for kind in ('filtered', 'approach', 'upright')}
    tracking_pub = node.create_publisher(Bool, '/track_3d/is_tracking', 10)
    subscriptions = observe(node, report['observations'])
    client = ActionClient(node, VisualServoingArm, '/humanoid_arm_visual_servoing')
    if not client.wait_for_server(timeout_sec=15):
        raise TimeoutError('Visual-servo action is unavailable')
    request = VisualServoingArm.Goal(target_description='recorded elevator button', left_arm=args.arm == 'left')
    handle = await_future(node, client.send_goal_async(request), 10)
    if not handle.accepted:
        raise RuntimeError('Visual-servo goal was rejected')
    result_future = handle.get_result_async()
    # (2) Replay every saved sample once, then wait for the terminal action result.
    try:
        stream(node, rows, publishers, tracking_pub)
        result = await_future(node, result_future, args.timeout)
        report['result'] = dict(status=result.status, success=result.result.success,
                                message=result.result.message, final_distance=result.result.final_distance,
                                orientation_error_deg=result.result.orientation_error_deg)
    finally:
        if not result_future.done():
            await_future(node, handle.cancel_goal_async(), 10)
        tracking_pub.publish(Bool(data=False))
    report['samples'] = sum(r['kind'] == 'approach' for r in rows)
    report['source'] = data['metadata']
    report['arm'] = args.arm
    report['start_time_s'] = args.start_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trajectory')
    parser.add_argument('output')
    parser.add_argument('--arm', choices=['left', 'right'], default='left')
    parser.add_argument('--start-time', type=float, default=0.0)
    parser.add_argument('--timeout', type=float, default=45.0)
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node('button_trajectory_replay')
    report = dict(observations=[])
    try:
        run(args, node, report)
    finally:
        Path(args.output).write_text(json.dumps(report))
        node.destroy_node()
        rclpy.try_shutdown()
    print(json.dumps({key: value for key, value in report.items() if key != 'observations'}, indent=2))


if __name__ == '__main__':
    main()
