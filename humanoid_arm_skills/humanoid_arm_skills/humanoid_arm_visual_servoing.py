"""
Visual servoing action server for arm manipulation.

Accepts a text description of a target and a left/right arm selector,
and performs visual servoing to guide the arm toward the target.

Uses dynamic replanning: an initial motion goal is sent, then a newer goal is
sent whenever the tracked target moves far enough. Planning and trajectory
tracking both live behind the move_to_pose action, so a newer goal preempts the
old one inside that server and the arm stays commanded across planning latency.
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
from std_srvs.srv import Trigger

from humanoid_manipulation_interfaces.action import VisualServoingArm as VisualServoing
from humanoid_manipulation_interfaces.action import MoveToPose

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
        self.move_to_pose_action = self.declare_parameter(
            'move_to_pose_action', '/move_to_pose').value
        self.move_to_pose_cancel_service = self.declare_parameter(
            'move_to_pose_cancel_service',
            self.move_to_pose_action + '/cancel').value
        self.target_update_topic = self.declare_parameter(
            'target_update_topic', '~/target_pose').value

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
        self._last_replan_time = 0.0
        self._last_planned_goal = None      # PoseStamped sent in the last goal

        # ── Thread-safe motion state ──
        self._motion_lock = threading.Lock()
        self._motion_goal_handle = None
        # Motions are numbered so that acceptances arriving out of order cannot
        # leave _motion_goal_handle pointing at an already-superseded goal —
        # cancelling that one would target a dead goal and let the live one run.
        self._motion_seq = 0
        self._motion_accepted_seq = -1
        # Bumped on every VS goal. A MoveToPose acceptance that lands after its
        # VS goal ended is cancelled on arrival rather than left commanding the
        # arm, which is the failure the old two-client split could not close:
        # a plan service call already in flight could not be cancelled.
        self._motion_generation = 0

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
        # The goal pose move_to_pose should follow as the target moves. This is
        # the offset-corrected pose, not the raw filtered one: the arm safety
        # offset is what makes it a reachable goal rather than the object's
        # surface, so the raw topic would be the wrong thing to track.
        self.target_pose_pub = self.create_publisher(
            PoseStamped, self.target_update_topic, 5)

        self.track3d_reset_pub = self.create_publisher(
            Empty, '/track_3d/reset', 5)
        self.keyboard_pub = self.create_publisher(
            String, self.keyboard_input_topic, 5)

        # ── Action client ──
        # move_to_pose owns the plan -> track handover and its preemption, so
        # replanning here is just "send a newer goal".
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
            self._last_replan_time = 0.0
            self._last_planned_goal = None
        with self._motion_lock:
            # Retires any motion still in flight from a previous VS goal.
            self._motion_generation += 1
            self._motion_accepted_seq = -1
            self._motion_goal_handle = None

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
            # Order matters. Clearing the handle first makes the NoActiveGoal
            # policy reject any further replan, and marks a goal acceptance
            # still in flight as stale. Cancelling first left a window in which
            # the pose callback could fire one more motion goal on the way out,
            # and the arm kept moving after the action had returned.
            self._active_goal_handle = None
            self._cancel_motion_goal()
            self.last_goal = None
            self.last_pose_time = None
            self._last_planned_goal = None

    # ── Dynamic replanning ──

    def _request_replan(self):
        """Send a new move_to_pose goal. Thread-safe, non-blocking.

        Gated on goal change: only sends if the current goal has moved by at
        least `replan_goal_position_delta` metres or rotated by
        `replan_goal_orientation_delta_deg` degrees since the last one, and
        skipped if the arm is already within `replan_skip_distance`.

        No "already in progress" gate: a newer goal preempts the older one
        inside move_to_pose, which is strictly better than dropping it. The old
        trajectory keeps executing there until the new plan is ready, so the arm
        is never left uncommanded while planning.
        """
        ctx = ReplanContext(node=self, now=time.monotonic())
        if self._replan_chain.should_skip(ctx, self.get_logger()):
            return

        target_pose = self.last_goal

        if not self._move_client.server_is_ready():
            self.get_logger().warn('move_to_pose action server not available')
            return

        with self._replan_lock:
            self._last_planned_goal = target_pose
            # Timed from the request, not the response: a lost or slow response
            # must not leave the cooldown permanently unarmed.
            self._last_replan_time = time.monotonic()
        with self._motion_lock:
            generation = self._motion_generation
            self._motion_seq += 1
            seq = self._motion_seq

        goal = MoveToPose.Goal()
        goal.target_pose = target_pose
        goal.left_arm = bool(self._left_arm)
        # Where move_to_pose can follow this target as it moves. Stored there
        # for now; replanning still happens by sending a newer goal from here.
        goal.target_topic = self.target_pose_pub.topic_name

        self.get_logger().info('Sending move_to_pose goal ...')
        send_future = self._move_client.send_goal_async(
            goal, feedback_callback=self._move_feedback_cb)
        send_future.add_done_callback(
            lambda fut: self._move_goal_response_cb(fut, generation, seq))

    # ── MoveToPose action client ──

    def _move_goal_response_cb(self, future, generation, seq):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().warn(f'move_to_pose send failed: {exc}')
            return

        if not goal_handle.accepted:
            self.get_logger().warn('move_to_pose goal rejected')
            return

        # The VS goal this motion belonged to may have finished while the
        # acceptance was in flight. Cancel rather than leave the arm commanded.
        with self._motion_lock:
            stale = (generation != self._motion_generation
                     or self._active_goal_handle is None)
            if not stale and seq > self._motion_accepted_seq:
                self._motion_accepted_seq = seq
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
        """Log motion progress.

        Purely diagnostic: replanning is driven by target motion, not by this
        error (see ``_filtered_pose_cb``). The controller advances waypoints on
        elapsed time, so ``current_error`` is tracking lag rather than evidence
        of a stuck arm.
        """
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
            return

        with self._motion_lock:
            if generation != self._motion_generation:
                return

        if result.success:
            self.get_logger().info(
                f'move_to_pose succeeded (final_error={result.final_error:.4f})')
            self._controller_succeeded = True
        else:
            # Preemption and cancellation are routine here: every replan
            # supersedes the goal before it.
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
            self._motion_accepted_seq = -1
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
        self.target_pose_pub.publish(msg)

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
