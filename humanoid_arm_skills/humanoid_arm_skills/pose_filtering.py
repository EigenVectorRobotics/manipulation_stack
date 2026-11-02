import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import Pose2D, PoseStamped, Quaternion
from std_msgs.msg import Bool

from humanoid_arm_skills.target_kf import TargetKF
from humanoid_arm_skills.utils import (
    align_surface_z,
    approach_pose,
    normal_quaternion,
    normalize_angle,
    quaternion_ema,
    quaternion_matrix,
    surface_normal,
    upright_pose,
)

import tf2_ros
import tf2_geometry_msgs

FILTER_TYPES = ('ema', 'kalman')
KF_PARAMETERS = {'kf_q_pos': 3e-3, 'kf_sig_lat': 4e-3, 'kf_sig_depth': 1.5e-2, 'kf_sig_kin': 2e-3,
                 'kf_q_ang_deg': 0.5, 'kf_sig_ang_deg': 3.0, 'kf_gate': 16.27}


class PoseFilterNode(Node):
    """Exponential or Kalman pose filter with timestamped frame transforms."""

    def __init__(self, tf_buffer: tf2_ros.Buffer | None = None):
        # (1) Configure topics and smoothing.
        super().__init__('pose_filter')
        input_topic = self.declare_parameter('input_topic', '/pose_in').value
        output_topic = self.declare_parameter('output_topic', '/pose_out').value
        msg_type = self.declare_parameter('msg_type', 'PoseStamped').value
        self.target_frame = self.declare_parameter('target_frame_arm', 'pelvis').value
        filter_parameters = [self.declare_parameter('alpha', 0.2), self.declare_parameter('filter_type', 'ema')]
        filter_parameters += [self.declare_parameter(name, value) for name, value in KF_PARAMETERS.items()]
        validation = self._validate_parameters(filter_parameters)
        if not validation.successful:
            raise ValueError(validation.reason)
        self.add_on_set_parameters_callback(self._validate_parameters)
        self.declare_parameter('tf_timeout', 0.5)
        self.declare_parameter('lost_timeout', 5.0)
        reset_filter_topic = self.declare_parameter('reset_filter_topic', 'reset_filter').value

        # (2) Initialize the estimate and TF listener.
        self.filtered_position = None
        self.filtered_orientation = None
        self.filter_frame = None
        self.last_input_time = None
        self.target_kf = None
        self.is_3d = (msg_type == 'PoseStamped')
        self.tf_buffer = tf_buffer
        self.tf_listener = None
        if self.is_3d and self.target_frame:
            if self.tf_buffer is None:
                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = self._create_tf_listener()

        # (3) Connect pose inputs and outputs.
        if self.is_3d:
            self.create_subscription(PoseStamped, input_topic, self._pose_stamped_cb, 5)
            self.pub = self.create_publisher(PoseStamped, output_topic, 5)
            self.upright_pub = self.create_publisher(PoseStamped, f'{output_topic}_upright', 5)
            self.approach_pub = self.create_publisher(PoseStamped, f'{output_topic}_approach', 5)
        else:
            self.create_subscription(Pose2D, input_topic, self._pose2d_cb, 5)
            self.pub = self.create_publisher(Pose2D, output_topic, 5)
        self.create_subscription(
            Bool, reset_filter_topic, self._reset_filter_cb,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))

        # (4) Report configuration.
        self.get_logger().debug(
            f'Pose EMA: {input_topic} ({msg_type}) -> {output_topic}'
            + (f' [tf -> {self.target_frame}]' if self.target_frame else ''))
        self.get_logger().info('pose_filtering: ready')

    def _create_tf_listener(self):
        tf_node = Node(
            f'{self.get_name()}_tf', context=self.context, namespace=self.get_namespace(),
            use_global_arguments=False, enable_rosout=False, start_parameter_services=False,
            cli_args=['--ros-args', '-r', f'/tf:={self.resolve_topic_name("/tf")}',
                      '-r', f'/tf_static:={self.resolve_topic_name("/tf_static")}'])
        return tf2_ros.TransformListener(self.tf_buffer, tf_node, spin_thread=True)

    def destroy_node(self):
        if self.tf_listener is not None:
            self.tf_listener.executor.shutdown()
            self.tf_listener.dedicated_listener_thread.join()
            self.tf_listener.unregister()
            self.tf_listener.node.destroy_node()
            self.tf_listener = None
        return super().destroy_node()

    def _validate_parameters(self, parameters):
        for parameter in parameters:
            if parameter.name == 'alpha':
                value = parameter.value
                if not isinstance(value, float) or not np.isfinite(value) or not 0.0 < value <= 1.0:
                    return SetParametersResult(successful=False, reason='alpha must be a float in (0, 1]')
            elif parameter.name == 'filter_type' and parameter.value not in FILTER_TYPES:
                return SetParametersResult(successful=False, reason=f'filter_type must be one of {FILTER_TYPES}')
            elif parameter.name in KF_PARAMETERS:
                value = parameter.value
                if not isinstance(value, float) or not np.isfinite(value) or value <= 0.0:
                    return SetParametersResult(successful=False, reason=f'{parameter.name} must be a positive float')
        return SetParametersResult(successful=True)

    def _reset_filter_cb(self, msg):
        if msg.data:
            self._reset_filter()

    def _reset_filter(self):
        self.filtered_position = None
        self.filtered_orientation = None
        self.filter_frame = None
        self.last_input_time = None
        self.target_kf = None

    def _update_filter(self, position, orientation, stamp_ns):
        alpha = self.get_parameter('alpha').value
        if self.is_3d:
            orientation = align_surface_z(orientation)
        if self.filtered_position is None:
            self.filtered_position = position.copy()
            self.filtered_orientation = orientation
        else:
            self.filtered_position += alpha * (position - self.filtered_position)
            if self.is_3d:
                self.filtered_orientation = align_surface_z(
                    quaternion_ema(self.filtered_orientation, orientation, alpha))
            else:
                difference = normalize_angle(orientation - self.filtered_orientation)
                self.filtered_orientation = normalize_angle(self.filtered_orientation + alpha * difference)
        self.last_input_time = stamp_ns
        self.target_kf = None

    def _update_kalman(self, position, orientation, R_frame_measurement, stamp_ns):
        if self.target_kf is None:
            p = {name: self.get_parameter(name).value for name in KF_PARAMETERS}
            self.target_kf = TargetKF(
                q_pos=p['kf_q_pos'], sig_lat=p['kf_sig_lat'], sig_depth=p['kf_sig_depth'], sig_kin=p['kf_sig_kin'],
                q_ang=np.radians(p['kf_q_ang_deg']), sig_ang=np.radians(p['kf_sig_ang_deg']), gate=p['kf_gate'])
        
        nis, accepted = self.target_kf.update(
            stamp_ns * 1e-9, position, surface_normal(orientation), R_frame_measurement)
        
        if not accepted:
            self.get_logger().warn(f'Rejected target outlier (NIS {nis:.1f})', throttle_duration_sec=2.0)
            return
        
        self.filtered_position = self.target_kf.p.copy()
        self.filtered_orientation = normal_quaternion(self.target_kf.n)
        self.last_input_time = stamp_ns

    def _transform_pose(self, msg):
        if not (self.tf_buffer and self.target_frame):
            return msg, np.eye(3)
        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            self.get_logger().warn('Skipping pose without an observation timestamp', throttle_duration_sec=2.0)
            return None, None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, msg.header.frame_id, msg.header.stamp,
                timeout=Duration(seconds=self.get_parameter('tf_timeout').value))
        except tf2_ros.TransformException as error:
            self.get_logger().warn(f'TF: {error}', throttle_duration_sec=2.0)
            return None, None
        r = transform.transform.rotation
        rotation = quaternion_matrix(np.array([r.x, r.y, r.z, r.w]))
        return tf2_geometry_msgs.do_transform_pose_stamped(msg, transform), rotation

    def _pose_stamped_cb(self, msg):
        msg, R_frame_measurement = self._transform_pose(msg)
        if msg is None:
            return
        position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = msg.pose.orientation
        orientation = np.array([q.x, q.y, q.z, q.w])
        norm = np.linalg.norm(orientation)
        if not np.all(np.isfinite(position)) or not np.isfinite(norm) or norm == 0.0:
            self.get_logger().warn('Skipping invalid pose', throttle_duration_sec=2.0)
            return
        stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self._check_timeout(stamp_ns)
        if self.filter_frame != msg.header.frame_id:
            self._reset_filter()
            self.filter_frame = msg.header.frame_id

        kalman = self.get_parameter('filter_type').value == 'kalman'
        if kalman:
            self._update_kalman(position, orientation / norm, R_frame_measurement, stamp_ns)
        else:
            self._update_filter(position, orientation / norm, stamp_ns)
        self._publish_pose(msg.header)

    def _publish_pose(self, header):
        out = PoseStamped()
        out.header = header
        out.pose.position.x, out.pose.position.y, out.pose.position.z = self.filtered_position.tolist()
        x, y, z, w = self.filtered_orientation.tolist()
        out.pose.orientation = Quaternion(x=x, y=y, z=z, w=w)
        self.pub.publish(out)
        approach_out = PoseStamped()
        approach_out.header = header
        approach_out.pose = approach_pose(out.pose)
        self.approach_pub.publish(approach_out)
        upright_out = PoseStamped()
        upright_out.header = header
        upright_out.pose = upright_pose(out.pose, flip=True)
        self.upright_pub.publish(upright_out)

    def _pose2d_cb(self, msg):
        position = np.array([msg.x, msg.y])
        if not np.all(np.isfinite(position)) or not np.isfinite(msg.theta):
            self.get_logger().warn('Skipping invalid pose', throttle_duration_sec=2.0)
            return
        stamp_ns = self.get_clock().now().nanoseconds
        self._check_timeout(stamp_ns)
        self._update_filter(position, normalize_angle(msg.theta), stamp_ns)
        out = Pose2D()
        out.x, out.y = self.filtered_position.tolist()
        out.theta = float(self.filtered_orientation)
        self.pub.publish(out)

    def _check_timeout(self, stamp_ns):
        if self.last_input_time is None:
            return
        elapsed = (stamp_ns - self.last_input_time) * 1e-9
        if elapsed < 0.0 or elapsed > self.get_parameter('lost_timeout').value:
            self._reset_filter()


def main(args=None):
    rclpy.init(args=args)
    node = PoseFilterNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
