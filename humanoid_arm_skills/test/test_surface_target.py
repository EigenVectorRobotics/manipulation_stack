import threading
import time
import unittest
from unittest.mock import Mock

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Quaternion, TransformStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage

from humanoid_arm_skills.pose_filtering import PoseFilterNode
from humanoid_arm_skills.utils import align_surface_z, approach_pose


def rotation(pose):
    q = pose.orientation
    return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()


def observation(seconds=100, nanoseconds=0):
    message = PoseStamped()
    message.header.frame_id = 'camera'
    message.header.stamp.sec = seconds
    message.header.stamp.nanosec = nanoseconds
    message.pose.position.x = 0.5
    q = Rotation.from_euler('xyz', [0.3, -0.4, 0.2]).as_quat()
    message.pose.orientation = Quaternion(**dict(zip(('x', 'y', 'z', 'w'), q.tolist())))
    return message


def transform(seconds, yaw, translation):
    message = TransformStamped()
    message.header.frame_id = 'pelvis'
    message.child_frame_id = 'camera'
    message.header.stamp.sec = seconds
    message.transform.translation.x = float(translation)
    message.transform.rotation.z = float(np.sin(yaw / 2))
    message.transform.rotation.w = float(np.cos(yaw / 2))
    return message


class SurfaceTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = PoseFilterNode()
        self.node.set_parameters([Parameter('tf_timeout', value=0.0)])
        self.node.pub = Mock()
        self.node.approach_pub = Mock()
        self.node.upright_pub = Mock()
        self.node.tf_buffer.set_transform(transform(100, 0, 1), 'test')
        self.node.tf_buffer.set_transform(transform(101, np.pi / 2, 9), 'test')

    def tearDown(self):
        self.node.destroy_node()

    def test_approach_preserves_tilt_and_reverses_normal(self):
        source = observation().pose
        result = approach_pose(source)
        expected = rotation(source) @ np.diag([-1, -1, 1])
        np.testing.assert_allclose(rotation(result), expected, atol=1e-12)
        self.assertEqual(result.position, source.position)
        self.assertGreater(abs(rotation(result)[2, 0]), 0.1)
        result.position.x = 10.0
        self.assertEqual(source.position.x, 0.5)

    def test_invalid_orientation_is_rejected(self):
        pose = Pose()
        pose.orientation.w = 0.0
        with self.assertRaises(ValueError):
            approach_pose(pose)

    def test_historical_transform_and_output_stamp(self):
        source = observation()
        self.node._pose_stamped_cb(source)
        filtered = self.node.pub.publish.call_args.args[0]
        approach = self.node.approach_pub.publish.call_args.args[0]
        self.assertAlmostEqual(filtered.pose.position.x, 1.5)
        np.testing.assert_allclose(rotation(filtered.pose)[:, 0], rotation(source.pose)[:, 0], atol=1e-12)
        self.assert_z_closest_to_up(rotation(filtered.pose))
        np.testing.assert_allclose(rotation(approach.pose)[:, 0], -rotation(source.pose)[:, 0], atol=1e-12)
        self.assertEqual(filtered.header.stamp, source.header.stamp)
        self.assertEqual(approach.header.stamp.sec, 100)
        self.assertEqual(approach.header.frame_id, 'pelvis')
        self.assertEqual(source.header.frame_id, 'camera')
        self.assertEqual(source.header.stamp.sec, 100)

    def test_missing_or_unavailable_stamp_never_uses_latest_tf(self):
        for seconds in (0, 99, 102):
            with self.subTest(seconds=seconds):
                self.node._pose_stamped_cb(observation(seconds))
                self.node.pub.publish.assert_not_called()
                self.node.approach_pub.publish.assert_not_called()
        self.assertIsNone(self.node.filtered_position)

    def test_z_alignment_uses_the_filtering_frame(self):
        self.node.target_frame = 'left_ankle_roll_link'
        camera_tf = transform(100, 0, 1)
        camera_tf.header.frame_id = self.node.target_frame
        q = Rotation.from_euler('x', 0.7).as_quat()
        camera_tf.transform.rotation = Quaternion(**dict(zip(('x', 'y', 'z', 'w'), q.tolist())))
        self.node.tf_buffer.clear()
        self.node.tf_buffer.set_transform(camera_tf, 'test')
        source = observation()
        self.node._pose_stamped_cb(source)
        filtered = self.node.pub.publish.call_args.args[0]
        expected = Rotation.from_quat(q).as_matrix() @ rotation(source.pose)
        np.testing.assert_allclose(rotation(filtered.pose)[:, 0], expected[:, 0], atol=1e-12)
        self.assert_z_closest_to_up(rotation(filtered.pose))
        self.assertEqual(filtered.header.frame_id, self.node.target_frame)

    def test_position_step_response_decays_exponentially(self):
        self.node._pose_stamped_cb(observation())
        changed = observation()
        changed.pose.position.x += 1.0
        for step in range(1, 5):
            self.node._pose_stamped_cb(changed)
            filtered = self.node.pub.publish.call_args.args[0]
            self.assertAlmostEqual(filtered.pose.position.x, 2.5 - 0.8 ** step)

    def assert_z_closest_to_up(self, matrix):
        normal = matrix[:, 0]
        projected_up = np.array([0.0, 0.0, 1.0]) - normal[2] * normal
        np.testing.assert_allclose(matrix[:, 2], projected_up / np.linalg.norm(projected_up), atol=1e-12)

    def test_orientation_ema_preserves_normal_tilt(self):
        initial = observation()
        initial.pose.orientation = Quaternion(w=1.0)
        self.node._pose_stamped_cb(initial)
        changed = observation()
        self.node._pose_stamped_cb(changed)
        filtered = self.node.pub.publish.call_args.args[0]
        target_q = align_surface_z(Rotation.from_matrix(rotation(changed.pose)).as_quat())
        expected = Rotation.from_rotvec(0.2 * Rotation.from_quat(target_q).as_rotvec()).as_matrix()
        np.testing.assert_allclose(rotation(filtered.pose)[:, 0], expected[:, 0], atol=1e-12)
        self.assert_z_closest_to_up(rotation(filtered.pose))
        self.assertAlmostEqual(np.linalg.norm(self.node.filtered_orientation), 1.0)

    def test_quaternion_sign_flip_does_not_move_the_estimate(self):
        source = observation()
        self.node._pose_stamped_cb(source)
        initial = rotation(self.node.pub.publish.call_args.args[0].pose)
        for field in ('x', 'y', 'z', 'w'):
            setattr(source.pose.orientation, field, -getattr(source.pose.orientation, field))
        self.node._pose_stamped_cb(source)
        filtered = self.node.pub.publish.call_args.args[0]
        np.testing.assert_allclose(rotation(filtered.pose), initial, atol=1e-12)
        self.assertEqual(self.node.pub.publish.call_count, 2)

    def test_arbitrary_roll_does_not_move_the_estimate(self):
        source = observation()
        self.node._pose_stamped_cb(source)
        initial = rotation(self.node.pub.publish.call_args.args[0].pose)
        for roll in np.linspace(-np.pi, np.pi, 13):
            changed = observation()
            q = (Rotation.from_matrix(rotation(source.pose)) * Rotation.from_euler('x', roll)).as_quat()
            changed.pose.orientation = Quaternion(**dict(zip(('x', 'y', 'z', 'w'), q.tolist())))
            self.node._pose_stamped_cb(changed)
            np.testing.assert_allclose(rotation(self.node.pub.publish.call_args.args[0].pose), initial, atol=1e-12)
        self.assertEqual(self.node.pub.publish.call_count, 14)

    def test_alignment_preserves_normals_including_vertical(self):
        orientations = list(Rotation.random(100, random_state=7).as_quat())
        orientations.extend(Rotation.from_euler('y', [[np.pi / 2], [-np.pi / 2]]).as_quat())
        for q in orientations:
            actual = Rotation.from_quat(align_surface_z(q)).as_matrix()
            expected = Rotation.from_quat(q).as_matrix()
            np.testing.assert_allclose(actual[:, 0], expected[:, 0], atol=1e-12)
            self.assertAlmostEqual(actual[2, 2], np.hypot(expected[0, 0], expected[1, 0]))

    def test_pose2d_takes_shortest_path_across_angle_wrap(self):
        self.node.is_3d = False
        self.node.set_parameters([Parameter('alpha', value=0.5)])
        self.node._pose2d_cb(Pose2D(x=0.0, y=0.0, theta=float(np.deg2rad(179))))
        self.node._pose2d_cb(Pose2D(x=2.0, y=4.0, theta=float(np.deg2rad(-179))))
        filtered = self.node.pub.publish.call_args.args[0]
        self.assertAlmostEqual(filtered.x, 1.0)
        self.assertAlmostEqual(filtered.y, 2.0)
        self.assertAlmostEqual(abs(filtered.theta), np.pi)

    def test_timeout_reinitializes_from_the_next_observation(self):
        self.node._pose_stamped_cb(observation())
        self.node.last_input_time -= 6_000_000_000
        changed = observation()
        changed.pose.position.x = 20.0
        self.node._pose_stamped_cb(changed)
        self.assertAlmostEqual(self.node.pub.publish.call_args.args[0].pose.position.x, 21.0)

    def test_stamped_filter_does_not_read_the_execution_clock(self):
        self.node.get_clock = Mock(side_effect=AssertionError('Filtering must use observation time'))
        self.node._pose_stamped_cb(observation())
        self.node._pose_stamped_cb(observation(100, 100_000_000))
        self.assertEqual(self.node.pub.publish.call_count, 2)
        self.assertEqual(self.node.last_input_time, 100_100_000_000)

    def test_backward_observation_time_starts_a_new_sequence(self):
        self.node._pose_stamped_cb(observation(101))
        self.node._pose_stamped_cb(observation(100))
        filtered = self.node.pub.publish.call_args.args[0]
        self.assertAlmostEqual(filtered.pose.position.x, 1.5)
        self.assertEqual(self.node.last_input_time, 100_000_000_000)

    def test_alpha_updates_are_validated(self):
        for value in (0.0, -0.1, 1.1, float('nan')):
            result = self.node.set_parameters([Parameter('alpha', value=value)])[0]
            self.assertFalse(result.successful)
        self.assertEqual(self.node.get_parameter('alpha').value, 0.2)
        self.node._pose_stamped_cb(observation())
        self.assertTrue(self.node.set_parameters([Parameter('alpha', value=1.0)])[0].successful)
        changed = observation()
        changed.pose.position.x = 5.0
        self.node._pose_stamped_cb(changed)
        self.assertAlmostEqual(self.node.pub.publish.call_args.args[0].pose.position.x, 6.0)

    def setup_delayed_tf(self):
        self.node.tf_buffer.clear()
        self.node.tf_buffer.set_transform(transform(100, 0, 1), 'test')
        self.node.set_parameters([Parameter('tf_timeout', value=1.0)])
        waiting, published = threading.Event(), threading.Event()
        original = self.node.tf_buffer.transform
        self.node.tf_buffer.transform = lambda *args, **kwargs: (waiting.set(), original(*args, **kwargs))[1]
        self.node.approach_pub.publish.side_effect = lambda _: published.set()
        source = rclpy.create_node('test_tf_source')
        publisher = source.create_publisher(PoseStamped, '/pose_in', 5)
        tf_publisher = source.create_publisher(TFMessage, '/tf', 5)
        return publisher, tf_publisher, source, waiting, published

    def test_tf_callbacks_run_while_pose_waits(self):
        publisher, tf_publisher, source, waiting, published = self.setup_delayed_tf()
        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        thread = threading.Thread(target=executor.spin)
        thread.start()
        try:
            deadline = time.monotonic() + 2
            while publisher.get_subscription_count() == 0 or tf_publisher.get_subscription_count() == 0:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            publisher.publish(observation(100, 500_000_000))
            self.assertTrue(waiting.wait(2))
            tf_publisher.publish(TFMessage(transforms=[transform(101, np.pi / 2, 9)]))
            self.assertTrue(published.wait(2))
            result = self.node.approach_pub.publish.call_args.args[0]
            expected = Rotation.from_euler('z', np.pi / 4).as_matrix() @ rotation(observation().pose)
            np.testing.assert_allclose(rotation(result.pose)[:, 0], -expected[:, 0], atol=1e-12)
            self.assert_z_closest_to_up(rotation(result.pose))
            self.assertEqual(result.header.stamp.nanosec, 500_000_000)
        finally:
            executor.shutdown()
            thread.join(timeout=2)
            source.destroy_node()


if __name__ == '__main__':
    unittest.main()
