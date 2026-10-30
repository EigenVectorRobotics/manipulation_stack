"""
Visual servoing action server for arm manipulation.

Accepts a text description of a target and a left/right arm selector,
and performs visual servoing to guide the arm toward the target.

Sends one move_to_pose goal and streams updated target poses to that action.
The move_to_pose server replans while the goal runs.
"""

import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Empty, String
from std_srvs.srv import Trigger

from humanoid_manipulation_interfaces.action import VisualServoingArm as VisualServoing
from humanoid_manipulation_interfaces.action import MoveToPose

from humanoid_arm_skills.utils import pose_distance
from humanoid_arm_skills.replan_policies import (
    InPostGoalCooldown,
    ReplanContext,
    ReplanPolicyChain,
)


class VisualServoArmNode(Node):
    """Action-server visual servoing for arm manipulation."""

    def __init__(self):
        super().__init__('humanoid_arm_vs')

        cb_group = ReentrantCallbackGroup()
        self._cb_group = cb_group

        # ── Static parameters (captured once for sub/pub/client setup) ──
        self.filtered_pose_topic = self.declare_parameter(
            'filtered_pose_topic',
            '/track_3d/selected_normal_pose_filtered/arm_approach').value
        self.keyboard_input_topic = self.declare_parameter(
            'keyboard_input_topic', '/keyboard_input').value
        self.left_ee_pose_topic = self.declare_parameter(
            'left_ee_pose_topic', '/left_ee_pose').value
        self.right_ee_pose_topic = self.declare_parameter(
            'right_ee_pose_topic', '/right_ee_pose').value
        self.move_to_pose_action = self.declare_parameter(
            'move_to_pose_action', '/move_to_pose').value
        self.move_to_pose_cancel_service = self.declare_parameter(
            'move_to_pose_cancel_service',
            self.move_to_pose_action + '/cancel').value
        # Only move toward a pose tracked from this goal's own seed.
        self.require_seed = self.declare_parameter('require_seed', True).value

        # ── Dynamic parameters (read via self.get_parameter(...).value) ──
        self.declare_parameter('lost_timeout', 5.0)
        self.declare_parameter('goal_rate', 2.0)
        self.declare_parameter('action_timeout', 600.0)
        self.declare_parameter('post_goal_cooldown', 1.0)
        # Ask for the target again when no seed arrives, then give up. seed_bbox
        # retries up to 3 frames itself, ~2 s each and up to ~9 s when Cosmos
        # runs to its token limit, so one request can take most of 30 s.
        self.declare_parameter('seed_timeout', 30.0)
        self.declare_parameter('seed_max_requests', 2)

        # ── Persistent state ──
        self.last_pose_time = None
        self.last_goal = None
        # Where this goal's tracking stands: 'reset' until track_3d reports it
        # stopped, 'seeding' until it reports tracking again, then 'tracking'.
        self._seed_state = 'tracking'
        self.is_tracking = False
        self._active_goal_handle = None
        self._controller_succeeded = False
        self.left_ee_pose = None
        self.right_ee_pose = None
        self._left_arm = False
        self._goal_completed_time = 0.0  # monotonic time of last goal success

        # ── Thread-safe motion state ──
        self._motion_lock = threading.Lock()
        self._motion_goal_handle = None
        self._motion_failure = None
        self._motion_generation = 0

        # ── Subscribers ──
        self.create_subscription(
            PoseStamped, self.filtered_pose_topic,
            self._filtered_pose_cb, 5, callback_group=cb_group)
        self.create_subscription(
            Bool, '/track_3d/is_tracking', self._is_tracking_cb,
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

        # ── Action client ──
        self._move_client = ActionClient(
            self, MoveToPose, self.move_to_pose_action,
            callback_group=cb_group)

        # Stopping the arm must not depend on this node holding the right goal
        # handle. The service clears the trajectory whatever is running, so it
        # is the reliable stop; cancelling the handle is the tidy one.
        self._move_cancel_client = self.create_client(
            Trigger, self.move_to_pose_cancel_service, callback_group=cb_group)

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

        self._goal_acceptance_chain = ReplanPolicyChain([
            InPostGoalCooldown(),
        ])

        self.get_logger().info(
            '\033[32m\033[1m\u2714\033[0m humanoid_arm_visual_servoing: ready '
            '(action server, target streaming)')

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
        with self._motion_lock:
            # Retires any motion still in flight from a previous VS goal.
            self._motion_generation += 1
            self._motion_goal_handle = None
            self._motion_failure = None

        # Reset track3d then initialise with VLM query
        self._request_target(target)
        seed_requests = 1
        seed_requested_at = time.monotonic()

        start_time = self.get_clock().now().nanoseconds
        result = VisualServoing.Result()
        feedback = VisualServoing.Feedback()

        motion_goal_sent = False
        rate = self.create_rate(self.get_parameter('goal_rate').value)

        try:
            while rclpy.ok():
                # ── Start motion once poses are available ──
                if not motion_goal_sent:
                    ee_pose = (self.left_ee_pose if self._left_arm
                               else self.right_ee_pose)
                    if self.last_goal is not None and ee_pose is not None:
                        self.get_logger().info('Sending initial motion goal')
                        motion_goal_sent = self._send_motion_goal()

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
                    result.success = False
                    result.message = 'Cancelled'
                    goal_handle.canceled()
                    self._active_goal_handle = None
                    self._cancel_motion_goal()
                    self.last_goal = None
                    self.last_pose_time = None
                    self._reset_track3d()
                    return result

                # ── No seed yet: ask again, then give up ──
                if (self.require_seed and self._seed_state != 'tracking'
                        and time.monotonic() - seed_requested_at
                        > self.get_parameter('seed_timeout').value):
                    max_requests = self.get_parameter('seed_max_requests').value
                    if seed_requests >= max_requests:
                        self.get_logger().warn(
                            f'No seed for "{target}" after {seed_requests} '
                            'request(s) — aborting')
                        result.success = False
                        result.message = (
                            f'No target seed after {seed_requests} request(s)')
                        self._fill_result(result, self._left_arm)
                        self._reset_track3d()
                        goal_handle.abort()
                        return result
                    seed_requests += 1
                    self.get_logger().warn(
                        f'No seed for "{target}" yet — asking again '
                        f'({seed_requests}/{max_requests})')
                    self._request_target(target)
                    seed_requested_at = time.monotonic()

                with self._motion_lock:
                    motion_failure = self._motion_failure
                if motion_failure is not None:
                    result.success = False
                    result.message = motion_failure
                    self._fill_result(result, self._left_arm)
                    goal_handle.abort()
                    return result

                elapsed = (self.get_clock().now().nanoseconds
                           - start_time) * 1e-9

                # ── Timeout ──
                if elapsed > self.get_parameter('action_timeout').value:
                    self.get_logger().warn('Action timed out')
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
            self._active_goal_handle = None
            self._cancel_motion_goal()
            self.last_goal = None
            self.last_pose_time = None

    # ── Initial motion ──

    def _send_motion_goal(self):
        target_pose = self.last_goal
        if not self._move_client.server_is_ready():
            self.get_logger().warn('move_to_pose action server not available')
            return False

        with self._motion_lock:
            generation = self._motion_generation

        goal = MoveToPose.Goal()
        goal.target_pose = target_pose
        goal.left_arm = bool(self._left_arm)
        goal.target_topic = self.filtered_pose_topic

        self.get_logger().info('Sending move_to_pose goal ...')
        try:
            send_future = self._move_client.send_goal_async(
                goal, feedback_callback=self._move_feedback_cb)
            send_future.add_done_callback(
                lambda fut: self._move_goal_response_cb(fut, generation))
        except Exception as exc:
            with self._motion_lock:
                if generation == self._motion_generation:
                    self._motion_failure = f'move_to_pose send failed: {exc}'
        return True

    # ── MoveToPose action client ──

    def _move_goal_response_cb(self, future, generation):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'move_to_pose send failed: {exc}')
            with self._motion_lock:
                if generation == self._motion_generation:
                    self._motion_failure = f'move_to_pose send failed: {exc}'
            return

        if not goal_handle.accepted:
            self.get_logger().warn('move_to_pose goal rejected')
            with self._motion_lock:
                if generation == self._motion_generation:
                    self._motion_failure = 'move_to_pose goal rejected'
            return

        # The VS goal this motion belonged to may have finished while the
        # acceptance was in flight. Cancel rather than leave the arm commanded.
        with self._motion_lock:
            stale = (generation != self._motion_generation
                     or self._active_goal_handle is None)
            if not stale:
                self._motion_goal_handle = goal_handle

        if stale:
            self.get_logger().info(
                'move_to_pose accepted after its VS goal ended — cancelling')
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda fut: self._move_result_cb(fut, generation))

    def _move_feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        dist, _ = self._compute_errors(self._left_arm)

        self.get_logger().debug(
            f'move_to_pose feedback: goal distance={dist:.4f}m '
            f'tracking={fb.tracking} lag={fb.current_error:.4f} '
            f'waypoints={fb.waypoints_remaining}/{fb.total_waypoints}')

    def _move_result_cb(self, future, generation):
        """Handle the terminal result of a move_to_pose goal."""
        try:
            result = future.result().result
        except Exception as exc:
            self.get_logger().warn(f'move_to_pose result error: {exc}')
            with self._motion_lock:
                if generation == self._motion_generation:
                    self._motion_failure = f'move_to_pose result error: {exc}'
            return

        with self._motion_lock:
            if generation != self._motion_generation:
                return
            if result.success:
                self._controller_succeeded = True
            else:
                self._motion_failure = result.message or 'move_to_pose failed'

        if result.success:
            self.get_logger().info(
                f'move_to_pose succeeded (final_error={result.final_error:.4f})')
        else:
            self.get_logger().info(f'move_to_pose ended: {result.message}')

    def _cancel_motion_goal(self):
        """Stop the arm and retire any motion this node started.

        Two steps, deliberately. The cancel service clears the trajectory in
        move_to_pose no matter which goal is running, so the arm stops within a
        control period even if a goal we never saw accepted is the one moving
        it. Cancelling the handle then terminates our own goal cleanly. The
        generation is retired too, so an acceptance still in flight is cancelled
        when it lands.
        """
        with self._motion_lock:
            self._motion_generation += 1
            handle = self._motion_goal_handle
            self._motion_goal_handle = None

        if self._move_cancel_client.service_is_ready():
            try:
                self._move_cancel_client.call_async(Trigger.Request())
            except Exception as exc:
                self.get_logger().warn(f'move_to_pose cancel service failed: {exc}')
        else:
            self.get_logger().warn('move_to_pose cancel service not available')

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

    def _request_target(self, target):
        """Reset track3d, then ask for the target by name."""
        self.last_goal = None
        self.last_pose_time = None
        self._seed_state = 'reset'
        self._reset_track3d()
        time.sleep(0.3)
        self.keyboard_pub.publish(String(data=target))

    # ── Subscriber callbacks ──

    def _is_tracking_cb(self, msg: Bool):
        # track_3d reports this every frame. After a request, False means the
        # reset has landed and True after that means the new seed is tracked.
        self.is_tracking = msg.data
        if self._seed_state == 'reset' and not msg.data:
            self._seed_state = 'seeding'
        elif self._seed_state == 'seeding' and msg.data:
            self._seed_state = 'tracking'
            if self._active_goal_handle is not None:
                self.get_logger().info('Tracking the new seed; accepting its poses')

    def _filtered_pose_cb(self, msg: PoseStamped):
        # track_3d keeps publishing the previous target until the reset lands,
        # so on 2026-10-26 a goal whose grounding found nothing moved to the
        # last target and reported success. Poses count only once track_3d has
        # stopped and started again from this goal's request.
        if self.require_seed and self._seed_state != 'tracking':
            return
        self.last_pose_time = self.get_clock().now().nanoseconds
        self.last_goal = msg


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
