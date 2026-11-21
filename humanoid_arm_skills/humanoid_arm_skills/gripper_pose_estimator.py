import collections
import os
import threading

import cv2
import numpy as np
import warp as wp

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image

import tf2_ros

from humanoid_arm_skills.gripper_fit import GripperFit, GripperFoundationPose
from humanoid_arm_skills.utils import matrix_quaternion, matrix_rpy, quaternion_matrix

COLOR_SYNC = 0.03


def stamp_seconds(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class GripperPoseEstimator(Node):
    def __init__(self):
        super().__init__('gripper_pose_estimator')
        self.declare_parameter('arms', ['left', 'right'])
        self.declare_parameter('method', 'foundationpose')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('color_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('color_scale', 0.5)
        self.declare_parameter('refine_iterations', 2)
        # FoundationPose renders a decimated mesh when this file exists in the package meshes, else the full one
        self.declare_parameter('foundationpose_mesh', '{side}_dex1_closed_6k.STL')
        # A TensorRT plan of the refiner, built on this machine; empty runs the PyTorch refiner
        self.declare_parameter('foundationpose_engine', '')
        self.declare_parameter('min_points', 500)
        self.declare_parameter('min_inliers', 0.4)
        self.declare_parameter('tf_timeout', 0.2)
        self.declare_parameter('lock_rotation', True)
        self.declare_parameter('max_rotation_deg', 8.0)
        self.declare_parameter('max_translation', 0.03)

        # Each arm is fitted on every frame; one out of view drops out before the fit
        meshes = os.path.join(get_package_share_directory('humanoid_arm_skills'), 'meshes')
        self.method = self.get_parameter('method').value
        self.arms = [{
            'base_frame': f'{side}_dex1_base_link',
            'estimate_frame': f'{side}_dex1_base_link_estimate',
            'tip_frame': f'{side}_ee',
            'tip': None,
            'logger': self.get_logger().get_child(side),
            'mesh_path': os.path.join(meshes, f'{side}_dex1_closed.STL'),
            'fp_mesh_path': os.path.join(meshes, self.get_parameter('foundationpose_mesh').value.format(side=side)),
            'correction': np.eye(4),
        } for side in self.get_parameter('arms').value]
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.k = None
        self.k_color = None
        self.colors = collections.deque(maxlen=10)
        self.depths = collections.deque(maxlen=10)
        self.last_depth = 0.0
        self.pending = None
        self.wake = threading.Condition()
        self.running = True

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.create_subscription(CameraInfo, self.get_parameter('camera_info_topic').value, self._camera_info_cb, 1)
        # A one-deep best-effort reader starves on large images under load; _depth_cb already keeps only the newest frame
        self.create_subscription(Image, self.get_parameter('depth_topic').value, self._depth_cb, qos_profile_sensor_data)
        if self.method == 'foundationpose':
            self.create_subscription(CameraInfo, self.get_parameter('color_info_topic').value, self._color_info_cb, 1)
            self.create_subscription(Image, self.get_parameter('color_topic').value, self._color_cb, qos_profile_sensor_data)
        # Fits run on their own thread, so every GPU call shares one CUDA context and the executor never waits on the GPU
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    # ========== Callbacks ==========
    def _camera_info_cb(self, msg):
        self.k = np.array(msg.k).reshape(3, 3)

    def _color_info_cb(self, msg):
        self.k_color = np.array(msg.k).reshape(3, 3)

    def _color_cb(self, msg):
        with self.wake:
            self.colors.append(msg)
            self.wake.notify()

    def _depth_cb(self, msg):
        with self.wake:
            self.pending = msg
            self.depths.append(msg)
            self.wake.notify()

    # ========== Fit ==========
    def _work(self):
        if self.method == 'foundationpose':
            for arm in self.arms:
                mesh = arm['fp_mesh_path'] if os.path.exists(arm['fp_mesh_path']) else arm['mesh_path']
                if mesh != arm['fp_mesh_path']:
                    arm['logger'].warning(f"{arm['fp_mesh_path']} not found, FoundationPose renders {mesh}")
                arm['fitter'] = GripperFoundationPose(mesh, self.get_parameter('refine_iterations').value,
                                                      self.get_parameter('foundationpose_engine').value)
        else:
            wp.init()
            for arm in self.arms:
                arm['fitter'] = GripperFit(arm['mesh_path'])
        frames = ', '.join(arm['base_frame'] for arm in self.arms)
        self.get_logger().info(f'gripper_pose_estimator: ready ({frames}, {self.method})')
        while self.running:
            with self.wake:
                while (msg := self._next_depth()) is None and self.running:
                    self.wake.wait(timeout=0.5)
            ready = self.k is not None and (self.method != 'foundationpose' or self.k_color is not None)
            if msg is not None and ready:
                for arm in self.arms:
                    self._fit(arm, msg)

    def _next_depth(self):
        # FoundationPose takes the newest depth frame whose colour image has arrived: colour reaches the node later
        if self.method != 'foundationpose':
            msg, self.pending = self.pending, None
            return msg
        for msg in reversed(self.depths):
            if stamp_seconds(msg) <= self.last_depth:
                return None
            if self._color_for(msg) is not None:
                self.last_depth = stamp_seconds(msg)
                return msg
        return None

    def _color_for(self, msg):
        stamp = stamp_seconds(msg)
        color = min(list(self.colors), key=lambda c: abs(stamp_seconds(c) - stamp), default=None)
        return color if color is not None and abs(stamp_seconds(color) - stamp) <= COLOR_SYNC else None

    def _fit(self, arm, msg):
        # (1) The method's fit at the image's own stamp: the pose of the gripper base in the camera it was fitted in
        depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
        if self.method == 'foundationpose':
            fit = self._fit_foundationpose(arm, msg, depth)
        else:
            fit = self._fit_warp(arm, msg, depth)
        if fit is None or fit['points'] < self.get_parameter('min_points').value:
            return
        if fit['inliers'] < self.get_parameter('min_inliers').value:
            return
        arm['correction'] = fit['correction']

        # (2) The fitted pose as the estimate frame
        estimate = TransformStamped()
        estimate.header.stamp = msg.header.stamp
        estimate.header.frame_id = fit['frame_id']
        estimate.child_frame_id = arm['estimate_frame']
        p, q = estimate.transform.translation, estimate.transform.rotation
        p.x, p.y, p.z = fit['pose'][:3, 3].tolist()
        q.x, q.y, q.z, q.w = matrix_quaternion(fit['pose'][:3, :3]).tolist()
        tip = self._tip(arm)
        if tip is not None:
            tip.header.stamp = msg.header.stamp
        self.tf_broadcaster.sendTransform([estimate] if tip is None else [estimate, tip])

        rpy = np.degrees(matrix_rpy(fit['correction'][:3, :3]))
        arm['logger'].info(f"offset {np.round(fit['correction'][:3, 3] * 1000, 1)} mm, rpy {np.round(rpy, 1)} deg, "
                           f"inliers {fit['inliers']:.2f}, points {fit['points']}", throttle_duration_sec=1.0)

    def _fit_warp(self, arm, msg, depth):
        # Position fit, unless locked also the wrist rotation, warm-started from the last accepted offset
        camera_from_base = self._lookup(msg.header.frame_id, arm['base_frame'], msg.header.stamp)
        if camera_from_base is None:
            return None
        fit = arm['fitter'].fit(depth, self.k, camera_from_base, arm['correction'],
                                lock_rotation=self.get_parameter('lock_rotation').value,
                                max_translation=self.get_parameter('max_translation').value,
                                max_rotation=np.radians(self.get_parameter('max_rotation_deg').value))
        return None if fit is None else dict(fit, frame_id=msg.header.frame_id)

    def _fit_foundationpose(self, arm, msg, depth):
        # RGB-D refine in the colour camera from the encoder pose every frame, on the colour image closest to the depth
        color = self._color_for(msg)
        if color is None:
            return None
        color_frame = color.header.frame_id
        color_from_base = self._lookup(color_frame, arm['base_frame'], msg.header.stamp)
        color_from_depth = self._lookup(color_frame, msg.header.frame_id, rclpy.time.Time().to_msg())
        if color_from_base is None or color_from_depth is None:
            return None
        rgb = np.frombuffer(color.data, np.uint8).reshape(color.height, color.step)[:, :color.width * 3]
        rgb = rgb.reshape(color.height, color.width, 3)
        if color.encoding == 'bgr8':
            rgb = rgb[:, :, ::-1]
        scale = self.get_parameter('color_scale').value
        rgb = np.ascontiguousarray(cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        k_color = self.k_color.copy()
        k_color[:2] *= scale
        fit = arm['fitter'].fit(rgb, depth, self.k, color_from_depth, k_color, color_from_base)
        return None if fit is None else dict(fit, frame_id=color_frame)

    def _lookup(self, target, source, stamp):
        try:
            transform = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time.from_msg(stamp),
                                                        timeout=Duration(seconds=self.get_parameter('tf_timeout').value))
        except tf2_ros.TransformException:
            return None
        q, p = transform.transform.rotation, transform.transform.translation
        matrix = np.eye(4)
        matrix[:3, :3] = quaternion_matrix(np.array([q.x, q.y, q.z, q.w]))
        matrix[:3, 3] = [p.x, p.y, p.z]
        return matrix

    def _tip(self, arm):
        # The tip under the estimate frame, sent with every fit: a once-latched static TF is missed by late subscribers
        if arm['tip'] is None:
            try:
                arm['tip'] = self.tf_buffer.lookup_transform(arm['base_frame'], arm['tip_frame'], rclpy.time.Time())
            except tf2_ros.TransformException:
                return None
            arm['tip'].header.frame_id = arm['estimate_frame']
            arm['tip'].child_frame_id = f"{arm['tip_frame']}_estimate"
        return arm['tip']


def main(args=None):
    rclpy.init(args=args)
    node = GripperPoseEstimator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.running = False
        node.worker.join(timeout=1.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
