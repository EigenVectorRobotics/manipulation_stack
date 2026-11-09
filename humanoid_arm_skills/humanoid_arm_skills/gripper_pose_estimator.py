import os
import threading

import numpy as np
import warp as wp

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image

import tf2_ros

from humanoid_arm_skills.gripper_fit import GripperFit
from humanoid_arm_skills.utils import quaternion_matrix


class GripperPoseEstimator(Node):
    def __init__(self):
        super().__init__('gripper_pose_estimator')
        self.declare_parameter('arms', ['left', 'right'])
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/depth/camera_info')
        self.declare_parameter('pose_topic', 'gripper_pose')
        self.declare_parameter('min_points', 500)
        self.declare_parameter('min_inliers', 0.4)
        self.declare_parameter('tf_timeout', 0.2)

        # Each arm is fitted on every frame; one out of view drops out before the fit
        meshes = os.path.join(get_package_share_directory('humanoid_arm_skills'), 'meshes')
        self.arms = [{
            'base_frame': f'{side}_dex1_base_link',
            'estimate_frame': f'{side}_dex1_base_link_estimate',
            'mesh_path': os.path.join(meshes, f'{side}_dex1_closed.STL'),
            'offset': np.zeros(3),
            'publisher': self.create_publisher(PoseStamped, f'{side}_{self.get_parameter("pose_topic").value}', 10),
        } for side in self.get_parameter('arms').value]
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.k = None
        self.pending = None
        self.wake = threading.Condition()
        self.running = True

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.create_subscription(CameraInfo, self.get_parameter('camera_info_topic').value, self._camera_info_cb, 1)
        # A one-deep best-effort reader starves on large images under load; _depth_cb already keeps only the newest frame
        self.create_subscription(Image, self.get_parameter('depth_topic').value, self._depth_cb, qos_profile_sensor_data)
        # Fits run on their own thread, so every Warp call shares one CUDA context and the executor never waits on the GPU
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    # ========== Callbacks ==========
    def _camera_info_cb(self, msg):
        self.k = np.array(msg.k).reshape(3, 3)

    def _depth_cb(self, msg):
        with self.wake:
            self.pending = msg
            self.wake.notify()

    # ========== Fit ==========
    def _work(self):
        wp.init()
        for arm in self.arms:
            arm['fitter'] = GripperFit(arm['mesh_path'])
        frames = ', '.join(arm['base_frame'] for arm in self.arms)
        self.get_logger().info(f'gripper_pose_estimator: ready ({frames}, {len(self.arms[0]["fitter"].plan)} passes per frame)')
        while self.running:
            with self.wake:
                while self.pending is None and self.running:
                    self.wake.wait(timeout=0.5)
                msg, self.pending = self.pending, None
            if msg is not None and self.k is not None:
                for arm in self.arms:
                    self._fit(arm, msg)

    def _fit(self, arm, msg):
        # (1) The encoder pose of the gripper in the depth camera at the image's own stamp; the arm TF lags the image slightly
        try:
            transform = self.tf_buffer.lookup_transform(
                msg.header.frame_id, arm['base_frame'], rclpy.time.Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=self.get_parameter('tf_timeout').value))
        except tf2_ros.TransformException:
            return
        q, p = transform.transform.rotation, transform.transform.translation
        camera_from_base = np.eye(4)
        camera_from_base[:3, :3] = quaternion_matrix(np.array([q.x, q.y, q.z, q.w]))
        camera_from_base[:3, 3] = [p.x, p.y, p.z]

        # (2) Translation-only fit, warm-started from the last accepted offset
        depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)
        fit = arm['fitter'].fit(depth, self.k, camera_from_base, arm['offset'])
        if fit is None or fit['points'] < self.get_parameter('min_points').value:
            return
        if fit['inliers'] < self.get_parameter('min_inliers').value:
            return
        arm['offset'] = fit['offset']

        # (3) The encoder orientation with the fitted position, in the camera frame, as a pose and as the estimate frame
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (camera_from_base[:3, 3] + arm['offset']).tolist()
        pose.pose.orientation = q
        arm['publisher'].publish(pose)
        estimate = TransformStamped()
        estimate.header = msg.header
        estimate.child_frame_id = arm['estimate_frame']
        estimate.transform.translation.x = pose.pose.position.x
        estimate.transform.translation.y = pose.pose.position.y
        estimate.transform.translation.z = pose.pose.position.z
        estimate.transform.rotation = q
        self.tf_broadcaster.sendTransform(estimate)


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
