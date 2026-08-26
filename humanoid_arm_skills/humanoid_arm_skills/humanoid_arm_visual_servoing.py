"""
Visual servoing action server for arm manipulation.

Accepts a text description of a target and a left/right arm selector,
and performs visual servoing to guide the arm toward the target.

Uses dynamic replanning: an initial trajectory plan is computed, then
replanning is triggered whenever the control feedback error exceeds a
configurable threshold for a sustained duration.
"""

import time
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Empty, String

from humanoid_manipulation_interfaces.action import VisualServoingArm as VisualServoing
from humanoid_manipulation_interfaces.srv import PlanJointTrajectory
from humanoid_manipulation_interfaces.action import ControlJointTrajectory

from humanoid_arm_skills.utils import pose_distance
from humanoid_arm_skills.replan_policies import (
    EECloseToGoal,
    EEPoseUnavailable,
    InPostGoalCooldown,
    InReplanCooldown,
    NoActiveGoal,
    ReplanContext,
    ReplanPolicyChain,
    TargetPoseUnavailable,
)


class VisualServoArmNode(Node):
    """Action-server visual servoing for arm manipulation."""

    def __init__(self):
        super().__init__('humanoid_arm_vs')

        cb_group = ReentrantCallbackGroup()
        self._cb_group = cb_group

        # ── Static parameters (captured once for sub/pub/client setup) ──
        self.filtered_pose_upright_topic = self.declare_parameter(
            'filtered_pose_upright_topic',
            '/track_3d/selected_normal_pose_filtered/arm_upright').value
        self.keyboard_input_topic = self.declare_parameter(
            'keyboard_input_topic', '/keyboard_input').value
        self.left_ee_pose_topic = self.declare_parameter(
            'left_ee_pose_topic', '/left_ee_pose').value
        self.right_ee_pose_topic = self.declare_parameter(
            'right_ee_pose_topic', '/right_ee_pose').value
        self.plan_trajectory_service = self.declare_parameter(
            'plan_trajectory_service', '/plan_trajectory').value
        self.control_trajectory_action = self.declare_parameter(
            'control_trajectory_action', '/trajectory_controller').value

        # ── Dynamic parameters (read via self.get_parameter(...).value) ──
        self.declare_parameter('lost_timeout', 5.0)
        self.declare_parameter('goal_rate', 2.0)
        self.declare_parameter('action_timeout', 600.0)
        self.declare_parameter('replan_cooldown', 5.0)
        self.declare_parameter('replan_goal_position_delta', 0.01)
        self.declare_parameter('replan_goal_orientation_delta_deg', 15.0)
        self.declare_parameter('replan_skip_distance', 0.12)
        self.declare_parameter('post_goal_cooldown', 1.0)
        self.declare_parameter('arm_safety_distance_x', -0.01)
        self.declare_parameter('arm_safety_distance_y', 0.0)
        self.declare_parameter('arm_safety_distance_z', 0.03)

        # ── Persistent state ──
        self.last_pose_time = None
        self.last_goal = None
        self.is_tracking = False
        self._active_goal_handle = None
        self._controller_succeeded = False
        self.left_ee_pose = None
        self.right_ee_pose = None
        self._left_arm = False
        self._goal_completed_time = 0.0  # monotonic time of last goal success

        # ── Thread-safe replan state ──
        self._replan_lock = threading.Lock()
        self._replan_in_progress = False
        self._last_replan_time = 0.0
        self._last_planned_goal = None      # PoseStamped used in last plan

        # ── Thread-safe control state ──
        self._control_lock = threading.Lock()
        self._control_goal_handle = None

        # ── Subscribers ──
        self.create_subscription(
            PoseStamped, self.filtered_pose_upright_topic,
            self._filtered_pose_cb, 5, callback_group=cb_group)
        self.create_subscription(
            Bool, '/track_3d/is_tracking',
            lambda msg: setattr(self, 'is_tracking', msg.data),
            5, callback_group=cb_group)
        self.create_subscription(
            PoseStamped, self.left_ee_pose_topic,
            lambda msg: setattr(self, 'left_ee_pose', msg),
            5, callback_group=cb_group)
        self.create_subscription(
            PoseStamped, self.right_ee_pose_topic,
            lambda msg: setattr(self, 'right_ee_pose', msg),
            5, callback_group=cb_group)

        # ── Publishers ──
        self.track3d_reset_pub = self.create_publisher(
            Empty, '/track_3d/reset', 5)
        self.keyboard_pub = self.create_publisher(
            String, self.keyboard_input_topic, 5)

        # ── Service client ──
        self._plan_client = self.create_client(
            PlanJointTrajectory, self.plan_trajectory_service,
            callback_group=cb_group)

        # ── Action client ──
        self._control_client = ActionClient(
            self, ControlJointTrajectory, self.control_trajectory_action,
            callback_group=cb_group)

        # ── Action server ──
        self._action_server = ActionServer(
            self,
            VisualServoing,
            'humanoid_arm_visual_servoing',
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=cb_group,
        )

        # \u2500\u2500 Policy chains \u2500\u2500
        self._replan_chain = ReplanPolicyChain([
            NoActiveGoal(),
            EEPoseUnavailable(),
            TargetPoseUnavailable(),
            EECloseToGoal(),
            InPostGoalCooldown(),
            InReplanCooldown(),
        ])
        # Acceptance runs before any target for this goal has arrived, so it
        # can only use time-based state. EECloseToGoal belongs to the replan
        # chain, where last_goal is current.
        self._goal_acceptance_chain = ReplanPolicyChain([
            InPostGoalCooldown(),
        ])

        self.get_logger().info(
            '\033[32m\033[1m\u2714\033[0m humanoid_arm_visual_servoing: ready '
            '(action server, dynamic replanning)')

    # ── Action callbacks ──

    def _goal_cb(self, goal_request):
        if self._active_goal_handle is not None:
            self.get_logger().warn('Rejecting goal — another is active')
            return GoalResponse.REJECT

        ctx = ReplanContext(node=self, now=time.monotonic(),
                            left_arm=goal_request.left_arm)
        if self._goal_acceptance_chain.should_skip(ctx, self.get_logger()):
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle):
        self.get_logger().info('Cancel requested')
        return CancelResponse.ACCEPT

    def _execute_cb(self, goal_handle):
        self._active_goal_handle = goal_handle
        target = goal_handle.request.target_description
        self._left_arm = goal_handle.request.left_arm
        self.get_logger().info(
            f'Goal received: "{target}" '
            f'({"left" if self._left_arm else "right"} arm)')

        # Reset state
        self.last_pose_time = None
        self.last_goal = None
        self.is_tracking = False
        self._controller_succeeded = False
        with self._replan_lock:
            self._replan_in_progress = False
            self._last_replan_time = 0.0
            self._last_planned_goal = None

        # Reset track3d then initialise with VLM query
        self._reset_track3d()
        time.sleep(0.3)
        self.keyboard_pub.publish(String(data=target))

        start_time = self.get_clock().now().nanoseconds
        result = VisualServoing.Result()
        feedback = VisualServoing.Feedback()

        initial_plan_sent = False
        rate = self.create_rate(self.get_parameter('goal_rate').value)

        try:
            while rclpy.ok():
                # ── Trigger initial plan once poses are available ──
                if not initial_plan_sent:
                    ee_pose = (self.left_ee_pose if self._left_arm
                               else self.right_ee_pose)
                    if self.last_goal is not None and ee_pose is not None:
                        self.get_logger().info('Triggering initial plan')
                        self._request_replan()
                        initial_plan_sent = True

                # ── Controller succeeded ──
                if self._controller_succeeded:
                    self.get_logger().info(
                        'Controller succeeded — completing goal')
                    self._goal_completed_time = time.monotonic()
                    result.success = True
                    result.message = 'Controller reached target'
                    self._fill_result(result, self._left_arm)
                    self._reset_track3d()
                    goal_handle.succeed()
                    return result

                # ── Cancellation ──
                if goal_handle.is_cancel_requested:
                    self.get_logger().info('Goal cancelled')
                    self._cancel_control_goal()
                    result.success = False
                    result.message = 'Cancelled'
                    goal_handle.canceled()
                    self._reset_track3d()
                    return result

                elapsed = (self.get_clock().now().nanoseconds
                           - start_time) * 1e-9

                # ── Timeout ──
                if elapsed > self.get_parameter('action_timeout').value:
                    self.get_logger().warn('Action timed out')
                    self._cancel_control_goal()
                    result.success = False
                    result.message = 'Action timed out'
                    self._fill_result(result, self._left_arm)
                    self._reset_track3d()
                    goal_handle.abort()
                    return result

                # ── Tracking lost check ──
                tracking_lost = False
                if self.last_pose_time is not None:
                    pose_elapsed = (self.get_clock().now().nanoseconds
                                    - self.last_pose_time) * 1e-9
                    if pose_elapsed > self.get_parameter('lost_timeout').value:
                        tracking_lost = True

                # ── Publish feedback ──
                dist, ori_err = self._compute_errors(self._left_arm)
                feedback.distance_to_goal = dist
                feedback.orientation_error_deg = ori_err
                feedback.tracking_lost = tracking_lost
                feedback.elapsed_time = elapsed
                if self.last_goal is not None:
                    feedback.current_goal = self.last_goal
                goal_handle.publish_feedback(feedback)

                rate.sleep()

        finally:
            self._cancel_control_goal()
            self._active_goal_handle = None
            self.last_goal = None
            self.last_pose_time = None
            self._last_planned_goal = None

    # ── Dynamic replanning ──

    def _request_replan(self):
        """Request a new trajectory plan. Thread-safe, non-blocking.

        Gates replanning on goal change: only replans if the current goal has
        moved by at least `replan_goal_position_delta` metres or rotated by
        `replan_goal_orientation_delta_deg` degrees since the last plan.
        Also skips if the arm is already within `replan_skip_distance` of the
        goal.
        """
        ctx = ReplanContext(node=self, now=time.monotonic())
        if self._replan_chain.should_skip(ctx, self.get_logger()):
            return

        ee_pose = (self.left_ee_pose
                   if self._left_arm else self.right_ee_pose)
        target_pose = self.last_goal

        # Skip if a plan is already being planned
        with self._replan_lock:
            if self._replan_in_progress:
                self.get_logger().info(
                    'Replan skipped — plan already in progress')
                return
            self._replan_in_progress = True
            
        # The in-flight controller goal is deliberately left running: the
        # controller preempts it atomically when the new goal arrives, so the
        # arm stays commanded for the whole planning latency. Cancelling here
        # would clear its trajectory and leave the arm uncommanded instead.
        #
        # _replan_in_progress is handed over to _plan_response_cb only once a
        # request is genuinely in flight; every other exit must clear it here,
        # or one transient failure disables replanning for the node's life.
        request_sent = False
        try:
            if not self._plan_client.service_is_ready():
                self.get_logger().warn(
                    'PlanJointTrajectory service not available')
                return

            self.get_logger().info('Planning trajectory ...')
            request = PlanJointTrajectory.Request()
            request.start_pose = ee_pose
            request.target_pose = target_pose
            request.left_arm = bool(self._left_arm)

            self._last_planned_goal = target_pose

            future = self._plan_client.call_async(request)
            future.add_done_callback(self._plan_response_cb)
            request_sent = True
        except Exception as exc:
            self.get_logger().warn(f'Replan request failed: {exc}')
        finally:
            if not request_sent:
                with self._replan_lock:
                    self._replan_in_progress = False

    def _plan_response_cb(self, future):
        """Handle PlanJointTrajectory service response."""
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warn(f'Plan service error: {e}')
            with self._replan_lock:
                self._replan_in_progress = False
            return

        with self._replan_lock:
            self._replan_in_progress = False
            self._last_replan_time = time.monotonic()

        if response.success:
            self.get_logger().info('Plan updated — sending control goal')
            self._send_control_goal(self._left_arm, response.trajectory)
        else:
            self.get_logger().warn('PlanJointTrajectory failed')

    # ── ControlJointTrajectory action client ──

    def _send_control_goal(self, left_arm, trajectory):
        """Send (or re-send) a ControlJointTrajectory goal."""
        # No cancel here: sending the new goal preempts the old one inside the
        # controller, which keeps the arm commanded across the handover.
        # This runs inside a callback-group thread, so it must not block.
        if not self._control_client.server_is_ready():
            self.get_logger().warn(
                'ControlJointTrajectory action server not available')
            with self._control_lock:
                self._control_goal_handle = None
            return

        goal = ControlJointTrajectory.Goal()
        goal.left_arm = left_arm
        goal.trajectory = trajectory

        send_future = self._control_client.send_goal_async(
            goal, feedback_callback=self._control_feedback_cb)
        send_future.add_done_callback(self._control_goal_response_cb)

    def _control_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('ControlJointTrajectory goal rejected')
            with self._control_lock:
                self._control_goal_handle = None
            return
        with self._control_lock:
            self._control_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._control_result_cb)

    def _control_feedback_cb(self, feedback_msg):
        """Log controller progress.

        This used to replan when ``current_error`` stayed high, which was
        meaningful while the controller advanced waypoints on tracking error.
        The controller now advances on elapsed time, so ``current_error`` is
        ordinary tracking lag and ``waypoints_remaining`` counts down on a
        clock — neither indicates a stuck arm. Replanning is driven by target
        motion instead (see ``_filtered_pose_cb``).
        """
        fb = feedback_msg.feedback
        dist, _ = self._compute_errors(self._left_arm)

        self.get_logger().debug(
            f'Control feedback: goal distance={dist:.4f}m '
            f'tracking lag={fb.current_error:.4f} '
            f'waypoints={fb.waypoints_remaining}/{fb.total_waypoints}')

    def _control_result_cb(self, future):
        """Handle result from ControlTrajectory action server."""
        try:
            result = future.result().result
            if result.success:
                self.get_logger().info(
                    f'Control trajectory succeeded '
                    f'(final_error={result.final_error:.4f})')
                self._controller_succeeded = True
            else:
                if result.final_error in (-1.0, -2.0):
                    self.get_logger().info(
                        'Control trajectory superseded (preempted or cancelled)')
                else:
                    self.get_logger().warn(
                        f'Control trajectory failed '
                        f'(final_error={result.final_error:.4f})')
        except Exception as e:
            self.get_logger().warn(f'Control result error: {e}')

    def _cancel_control_goal(self):
        """Cancel the current ControlTrajectory goal if one is active."""
        with self._control_lock:
            handle = self._control_goal_handle
            self._control_goal_handle = None
        if handle is not None:
            try:
                handle.cancel_goal_async()
            except Exception:
                pass

    # ── Helpers ──

    def _fill_result(self, result, left_arm):
        dist, ori_err = self._compute_errors(left_arm)
        result.final_distance = dist
        result.orientation_error_deg = ori_err

    def _compute_errors(self, left_arm):
        """Return (distance, orientation_error_deg) from EE to goal."""
        ee_pose = self.left_ee_pose if left_arm else self.right_ee_pose
        if self.last_goal is None or ee_pose is None:
            return float('inf'), float('inf')

        return pose_distance(self.last_goal.pose, ee_pose.pose)

    def _reset_track3d(self):
        try:
            self.track3d_reset_pub.publish(Empty())
            self.get_logger().info('track3d reset')
        except Exception:
            pass

    # ── Subscriber callbacks ──

    def _filtered_pose_cb(self, msg: PoseStamped):
        # Apply offset in the pose's local frame (along its local x-axis)
        q = msg.pose.orientation
        x, y, z, w = q.x, q.y, q.z, q.w
        rot = np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ])
        offset_local = np.array([
            self.get_parameter('arm_safety_distance_x').value,
            self.get_parameter('arm_safety_distance_y').value,
            self.get_parameter('arm_safety_distance_z').value,
        ])
        offset_world = rot @ offset_local
        msg.pose.position.x += float(offset_world[0])
        msg.pose.position.y += float(offset_world[1])
        msg.pose.position.z += float(offset_world[2])

        self.last_pose_time = self.get_clock().now().nanoseconds
        self.last_goal = msg

        if self._active_goal_handle is None:
            return

        # First plan: replan immediately without comparison logging.
        if self._last_planned_goal is None:
            return

        dp, d_ori_deg = pose_distance(
            msg.pose, self._last_planned_goal.pose)
        pos_delta = self.get_parameter(
            'replan_goal_position_delta').value
        ori_delta_deg = self.get_parameter(
            'replan_goal_orientation_delta_deg').value

        # Skip if the goal hasn't moved/rotated enough since the last plan.
        if dp < pos_delta and d_ori_deg < ori_delta_deg:
            self.get_logger().debug(
                f'Replan skipped — goal nearly unchanged: '
                f'dp={dp:.4f}m (< {pos_delta}m), '
                f'dθ={d_ori_deg:.2f}° (< {ori_delta_deg}°)',
                throttle_duration_sec=5.0)
            return

        triggers = []
        if dp >= pos_delta:
            triggers.append(f'dp={dp:.4f}m (>= {pos_delta}m)')
        if d_ori_deg >= ori_delta_deg:
            triggers.append(f'dθ={d_ori_deg:.2f}° (>= {ori_delta_deg}°)')

        self.get_logger().info(
            f'Goal updated — requesting replan ({", ".join(triggers)})')
        self._request_replan()


def main(args=None):
    rclpy.init(args=args)
    node = VisualServoArmNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._reset_track3d()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
