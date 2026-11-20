#include "server.h"
#include "../helper_funcs.h"

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace g1_pilot {

namespace {

bool valid_pose(const geometry_msgs::msg::Pose & pose)
{
    const auto & p = pose.position;
    const auto & q = pose.orientation;
    const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && std::abs(norm - 1.0) < 0.1;
}

} // namespace

void MoveToPoseServer::worker_loop_()
{
    while (rclcpp::ok() && !stopping_.load()) {
        std::shared_ptr<GoalHandle> goal;
        {
            std::unique_lock<std::mutex> lock(goal_mutex_);
            cv_.wait(lock, [this] { return pending_ || stopping_.load(); });
            if (stopping_.load()) return;
            goal = pending_;
            pending_ = nullptr;
        }
        // An exception must not take down the node while the arm is moving
        try {
            run_(goal);
        } catch (const std::exception & e) {
            fail_(goal, std::string("Internal error: ") + e.what());
        } catch (...) {
            fail_(goal, "Internal error");
        }
    }
}

void MoveToPoseServer::run_(const std::shared_ptr<GoalHandle> & goal)
{
    // (1) Take a start state at rest and transform the requested pose into the core frame.
    last_replan_ = {};
    plan_failures_ = 0;
    off_track_since_ = {};
    const auto request = goal->get_goal();
    if (!valid_pose(request->target_pose.pose)) {
        finish_(goal, false, 0.0, "Invalid target pose: non-finite position or non-unit quaternion");
        return;
    }
    JointState start;
    const auto start_failure = take_start_(goal, start);
    if (!start_failure.empty()) {
        finish_(goal, false, 0.0, start_failure);
        return;
    }
    const auto target = to_pelvis_frame_(request->target_pose);
    if (target.size() == 0) {
        finish_(goal, false, 0.0, "Target pose transform failed");
        return;
    }

    // (2) Plan without holding the control or goal mutex.
    RCLCPP_INFO(get_logger(), "Planning for %s arm ...", request->left_arm ? "left" : "right");
    Plan plan;
    try {
        plan = plan_(start, target, request->left_arm, false);
        check_reach_(plan, target);
    } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(), "Planning failed: %s", e.what());
        finish_(goal, false, 0.0, std::string("Planning failed: ") + e.what());
        return;
    }
    RCLCPP_INFO(get_logger(), "Planned %zu waypoints", plan.trajectory.points.size());

    // (3) Install only while this goal owns the arm, then monitor execution.
    {
        std::lock_guard<std::mutex> lock(ctrl_mutex_);
        Eigen::Isometry3d source;
        tf2::fromMsg(request->target_pose.pose, source);
        planned_source_ = source.matrix();
        planned_source_frame_ = request->target_pose.header.frame_id;
        final_leg_refreshed_ = false;
        hold_between_steps_ = false;
    }
    stepping_ = false;
    leg_step_ = 0;
    const auto status = install_(goal, plan.trajectory, target, plan.lock_time);
    if (status != GoalStatus::Active) {
        finish_(goal, false, 0.0, status == GoalStatus::Cancelled
            ? "Cancelled while planning" : "Preempted by a newer goal");
        return;
    }
    publish_target_(target);
    publish_path_(plan.path);
    RCLCPP_INFO(get_logger(), "Tracking trajectory");
    monitor_(goal);
}

std::string MoveToPoseServer::take_start_(const std::shared_ptr<GoalHandle> & goal, JointState & start)
{
    // (1) Give up when the goal ends or feedback is missing or stale.
    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(get_parameter("feedback_period").as_double()));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (rclcpp::ok() && !stopping_.load()) {
        {
            std::scoped_lock lock(goal_mutex_, ctrl_mutex_);
            if (active_ != goal) return "Preempted by a newer goal";
            if (goal->is_canceling()) return "Cancelled while planning";
            if (state_.name.empty()) return "No joint feedback received yet";
            const double age = feedback_age_locked_();
            if (age > get_parameter("feedback_timeout").as_double()) {
                char message[96];
                std::snprintf(message, sizeof(message), "Joint feedback is stale (%.2f s old)", age);
                return message;
            }

            // (2) Motion left by an earlier goal is braked first, so the plan starts at rest where the arm is.
            if (mode_ == ControlMode::Tracking) brake_locked_();
            if (mode_ != ControlMode::Braking) {
                start = mode_ == ControlMode::Holding ? held_state_locked_() : state_;
                return "";
            }
        }
        if (std::chrono::steady_clock::now() > deadline) return "Timed out braking the previous motion";
        std::unique_lock<std::mutex> lock(goal_mutex_);
        cv_.wait_for(lock, period);
    }
    return "Node shutting down";
}

MoveToPoseServer::Plan MoveToPoseServer::plan_(
    const JointState & start, const Eigen::MatrixXd & target, bool left, bool replan)
{
    planner_->setLeftArm(left);
    const double dt = get_parameter("plan_time_step").as_double();
    const double velocity = get_parameter("max_joint_vel").as_double();
    const double acceleration = get_parameter("max_joint_acc").as_double();
    const int steps = static_cast<int>(get_parameter("final_leg_steps").as_int());
    return to_plan_(replan
        ? planner_->replanTrajectory(start, target, steps, false, dt, velocity, acceleration)
        : planner_->planTrajectory(start, target, steps, false, dt, velocity, acceleration), left);
}

MoveToPoseServer::Plan MoveToPoseServer::to_plan_(const HumanoidPilot::JointTrajectory & planned, bool left)
{
    const auto configs = toJointConfigs(planned);
    if (configs.empty()) {
        const auto reason = planner_->plan_failure();
        throw std::runtime_error(reason.empty() ? "empty trajectory" : reason);
    }
    Plan result;
    result.path.reserve(configs.size());
    for (const auto & q : configs) result.path.push_back(planner_->forward_kinematics(q, left));
    result.trajectory = convertToTrajectory(planned, frame_, get_clock()->now());
    result.lock_time = planned.lock_time;
    if (result.trajectory.points.empty()) throw std::runtime_error("no waypoints");
    for (const auto & point : result.trajectory.points) {
        for (const double v : point.positions) {
            if (!std::isfinite(v)) throw std::runtime_error("non-finite joint position in the plan");
        }
        for (const double v : point.velocities) {
            if (!std::isfinite(v)) throw std::runtime_error("non-finite joint velocity in the plan");
        }
    }
    return result;
}

void MoveToPoseServer::try_replan_(const std::shared_ptr<GoalHandle> & goal)
{
    const auto target = replan_target_();
    if (target.size() == 0) return;
    JointState start;
    if (!stop_for_replan_(goal, start)) return;
    last_replan_ = std::chrono::steady_clock::now();
    Plan plan;
    try {
        plan = plan_(start, target, goal->get_goal()->left_arm, true);
        check_reach_(plan, target);
    } catch (const std::exception & e) {
        record_plan_failure_(goal, std::string("Replan failed: ") + e.what());
        return;
    }
    if (install_(goal, plan.trajectory, target, plan.lock_time) != GoalStatus::Active) return;
    plan_failures_ = 0;
    publish_path_(plan.path);
    publish_target_(target);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
        "Replanned to [%.3f %.3f %.3f], %zu waypoints",
        target(0, 3), target(1, 3), target(2, 3), plan.trajectory.points.size());
}

Eigen::MatrixXd MoveToPoseServer::replan_target_()
{
    // (1) Read the latest target when the replanning interval allows it.
    PoseStamped update;
    if (!read_target_(update)) return {};
    if (!valid_pose(update.pose)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Ignoring an invalid target update");
        return {};
    }
    if (last_replan_.time_since_epoch().count() != 0) {
        const double since = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - last_replan_).count();
        if (since < get_parameter("replan_min_interval").as_double()) return {};
    }
    const auto target = to_pelvis_frame_(update);
    if (target.size() == 0) return {};

    // (2) Finish braking before planning; a held endpoint always needs a replacement.
    std::lock_guard<std::mutex> lock(ctrl_mutex_);
    if (mode_ == ControlMode::Braking || mode_ == ControlMode::Complete) return {};
    Eigen::Isometry3d source_iso;
    tf2::fromMsg(update.pose, source_iso);
    const Eigen::MatrixXd source = source_iso.matrix();
    if (mode_ == ControlMode::Tracking && final_leg_refreshed_) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000, "Target fixed for the final leg");
        return {};
    }
    if (mode_ == ControlMode::Tracking && lock_time_ >= 0.0 &&
        (get_clock()->now() - traj_start_).seconds() >= lock_time_) {
        // The body shifts while the arm reaches out, so the final leg is replanned once against the latest target
        final_leg_refreshed_ = true;
        planned_source_ = source;
        planned_source_frame_ = update.header.frame_id;
        if (get_parameter("final_leg_stepping").as_bool()) {
            start_stepping_(target);
            return {};
        }
        RCLCPP_INFO(get_logger(), "Refreshing the final leg against the latest target");
        return target;
    }
    if (mode_ == ControlMode::Holding) {
        planned_source_ = source;
        planned_source_frame_ = update.header.frame_id;
        return target;
    }
    if (planned_for_.size() == 0) return {};
    const bool same_frame = planned_source_.size() != 0 && planned_source_frame_ == update.header.frame_id;
    const Eigen::MatrixXd & before = same_frame ? planned_source_ : planned_for_;
    const Eigen::MatrixXd & now = same_frame ? source : target;
    const double dp = (now.block<3, 1>(0, 3) - before.block<3, 1>(0, 3)).norm();
    const Eigen::Matrix3d dR = before.block<3, 3>(0, 0).transpose() * now.block<3, 3>(0, 0);
    if (dp >= get_parameter("replan_position_delta").as_double() ||
        std::abs(Eigen::AngleAxisd(dR).angle()) >= get_parameter("replan_orientation_delta").as_double()) {
        planned_source_ = source;
        planned_source_frame_ = update.header.frame_id;
        return target;
    }
    return {};
}

void MoveToPoseServer::start_stepping_(const Eigen::MatrixXd & target)
{
    // The goal is frozen in the anchor frame, which stays put while the pelvis leans with the reaching arm.
    try {
        goal_in_anchor_ = pelvis_from_anchor_().inverse() * Eigen::Matrix4d(target);
        goal_in_anchor_valid_ = true;
    } catch (const tf2::TransformException & ex) {
        goal_in_anchor_valid_ = false;
        RCLCPP_ERROR(get_logger(), "No final-leg anchor frame in TF: %s", ex.what());
    }
    stepping_offset_ = safety_offset_();
    leg_steps_ = static_cast<int>(get_parameter("final_leg_steps").as_int());
    hold_since_ = get_clock()->now();
    stepping_ = true;
    leg_step_ = 1;
    RCLCPP_INFO(get_logger(), "Stepping the final leg against the target fixed in %s",
                get_parameter("final_leg_anchor_frame").as_string().c_str());
}

void MoveToPoseServer::step_final_leg_(const std::shared_ptr<GoalHandle> & goal)
{
    // (1) Plan the next step only once the previous one is held at rest and the body has had time to settle.
    if (leg_step_ == 0) return;
    {
        std::lock_guard<std::mutex> lock(ctrl_mutex_);
        if (leg_step_ > 1 && mode_ != ControlMode::Holding) return;
        if (leg_step_ > 1 && (get_clock()->now() - hold_since_).seconds()
                < get_parameter("final_leg_settle_time").as_double()) return;
    }
    JointState start;
    if (!stop_for_replan_(goal, start)) return;

    // (2) Re-map the frozen goal into the current pelvis; a last pass repeats the goal step, then holds and retreats.
    // The frozen goal carries the safety offset of the stepping start; an offset set since then moves it in the target frame
    const std::size_t n = static_cast<std::size_t>(leg_steps_);
    const bool last = leg_step_ >= n;
    const std::size_t step = std::min(leg_step_, n - 1);
    Eigen::Matrix4d target;
    Plan plan;
    try {
        if (!goal_in_anchor_valid_) throw std::runtime_error("the anchor frame was not in TF when the leg started");
        target = pelvis_from_anchor_() * goal_in_anchor_;
        target.block<3, 1>(0, 3) += target.block<3, 3>(0, 0) * (safety_offset_() - stepping_offset_);
        target.block<3, 1>(0, 3) -= gripper_estimate_error_(goal->get_goal()->left_arm, target.block<3, 3>(0, 0));
        planner_->setLeftArm(goal->get_goal()->left_arm);
        const double dt = get_parameter("plan_time_step").as_double();
        const double velocity = get_parameter("max_joint_vel").as_double();
        const double acceleration = get_parameter("max_joint_acc").as_double();
        const int steps = static_cast<int>(get_parameter("final_leg_steps").as_int());
        plan = to_plan_(
            planner_->planFinalLegStep(start, target, step, leg_steps_, last, dt, velocity, acceleration),
            goal->get_goal()->left_arm);
        if (last) check_reach_(plan, target);
    } catch (const std::exception & e) {
        record_plan_failure_(goal, "Final-leg step " + std::to_string(leg_step_) + " failed: " + e.what());
        return;
    }

    // (3) Hold at the end of every step but the last, which completes the goal after its retreat.
    {
        std::lock_guard<std::mutex> lock(ctrl_mutex_);
        hold_between_steps_ = !last;
    }
    if (install_(goal, plan.trajectory, target, -1.0) != GoalStatus::Active) return;
    plan_failures_ = 0;
    publish_path_(plan.path);
    publish_target_(target);
    RCLCPP_INFO(get_logger(), "Final-leg step %zu/%zu to [%.3f %.3f %.3f]",
                leg_step_, n, target(0, 3), target(1, 3), target(2, 3));
    leg_step_ = last ? 0 : leg_step_ + 1;
}

void MoveToPoseServer::record_gripper_estimates_()
{
    // The error is taken at the fit's own stamp, so arm motion between the fit and its use does not bias it
    for (const bool left : {false, true}) {
        const std::string tip = std::string(left ? "left" : "right") + "_ee";
        geometry_msgs::msg::TransformStamped estimate, encoder;
        double stamp = 0.0;
        try {
            if (!tf_buffer_->canTransform("pelvis", tip + "_estimate", tf2::TimePointZero)) continue;
            estimate = tf_buffer_->lookupTransform("pelvis", tip + "_estimate", tf2::TimePointZero);
            stamp = rclcpp::Time(estimate.header.stamp).seconds();
            if (stamp <= gripper_estimate_seen_[left]) continue;
            if (!tf_buffer_->canTransform("pelvis", tip, rclcpp::Time(estimate.header.stamp))) continue;
            encoder = tf_buffer_->lookupTransform("pelvis", tip, rclcpp::Time(estimate.header.stamp));
        } catch (const tf2::TransformException &) {
            continue;
        }
        gripper_estimate_seen_[left] = stamp;
        const auto & e = estimate.transform.translation;
        const auto & k = encoder.transform.translation;
        const Eigen::Vector3d error(e.x - k.x, e.y - k.y, e.z - k.z);
        if (error.norm() > get_parameter("gripper_estimate_max_error").as_double()) {
            RCLCPP_WARN(get_logger(), "Ignoring %s gripper estimate: error %.1f mm",
                        left ? "left" : "right", error.norm() * 1e3);
            continue;
        }
        std::lock_guard<std::mutex> lock(estimate_mutex_);
        gripper_errors_[left].add(stamp, error);
    }
}

Eigen::Vector3d MoveToPoseServer::gripper_estimate_error_(bool left, const Eigen::Matrix3d & target_rotation)
{
    // Depth-fitted minus encoder gripper position in the pelvis; zero when disabled or no fit is recent enough
    constexpr double LEAD_TOWARD_GOAL = -0.008;  // m along the target's x: the planner sees the gripper this much short of the goal, so the press goes this much deeper
    if (!get_parameter("use_gripper_estimate").as_bool()) return Eigen::Vector3d::Zero();
    GripperErrorHistory::Correction c;
    {
        std::lock_guard<std::mutex> lock(estimate_mutex_);
        c = gripper_errors_[left].correction(get_clock()->now().seconds(),
                                            get_parameter("gripper_estimate_max_age").as_double(),
                                            get_parameter("gripper_estimate_hold_time").as_double());
    }
    if (c.fits == 0) {
        RCLCPP_WARN(get_logger(), "No gripper estimate (newest fit %.1f s old); stepping uncorrected", c.newest_age);
        return Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d error = c.error + target_rotation.col(0) * LEAD_TOWARD_GOAL;
    RCLCPP_INFO(get_logger(), "Gripper estimate error [%.1f %.1f %.1f] mm in pelvis from %s %zu fit(s), newest %.1f s old",
                error(0) * 1e3, error(1) * 1e3, error(2) * 1e3,
                c.newest_age <= get_parameter("gripper_estimate_max_age").as_double() ? "fresh" : "held median of",
                c.fits, c.newest_age);
    return error;
}

Eigen::Vector3d MoveToPoseServer::safety_offset_()
{
    return {get_parameter("arm_safety_distance_x").as_double(),
            get_parameter("arm_safety_distance_y").as_double(),
            get_parameter("arm_safety_distance_z").as_double()};
}

Eigen::Matrix4d MoveToPoseServer::pelvis_from_anchor_()
{
    // From TF, so the anchor can be the body EKF's foot_world; leg kinematics drift as the body leans
    const auto transform = tf_buffer_->lookupTransform(
        "pelvis", get_parameter("final_leg_anchor_frame").as_string(), tf2::TimePointZero,
        tf2::durationFromSec(get_parameter("tf_timeout").as_double()));
    return tf2::transformToEigen(transform).matrix();
}

void MoveToPoseServer::monitor_(const std::shared_ptr<GoalHandle> & goal)
{
    // (1) Monitor ownership, replanning, and control progress.
    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(get_parameter("feedback_period").as_double()));
    while (rclcpp::ok() && !stopping_.load()) {
        const auto status = goal_status_(goal);
        if (status != GoalStatus::Active) {
            if (status == GoalStatus::Cancelled) clear_trajectory_();
            finish_(goal, false, 0.0, status == GoalStatus::Cancelled
                ? "Cancelled" : "Preempted by a newer goal");
            return;
        }
        const auto failure = execution_failure_(goal->get_goal()->left_arm);
        if (!failure.empty()) {
            fail_(goal, failure);
            return;
        }
        if (stepping_) step_final_leg_(goal);
        else try_replan_(goal);
        if (goal_status_(goal) != GoalStatus::Active) continue;
        if (report_progress_(goal)) return;
        std::unique_lock<std::mutex> lock(goal_mutex_);
        cv_.wait_for(lock, period);
    }

    // (2) Terminate the action when the worker shuts down.
    finish_(goal, false, 0.0, "Node shutting down");
}

bool MoveToPoseServer::report_progress_(const std::shared_ptr<GoalHandle> & goal)
{
    // (1) Snapshot progress under the control mutex.
    auto feedback = std::make_shared<MoveToPoseFeedback>();
    ControlMode mode;
    double max_error = 0.0;
    JointState state;
    Eigen::MatrixXd target;
    {
        std::lock_guard<std::mutex> lock(ctrl_mutex_);
        mode = mode_;
        if (mode == ControlMode::Holding) update_errors_locked_(hold_command_.position);
        feedback->current_error = error_;
        max_error = max_error_;
        state = state_;
        target = planned_for_;
        feedback->tracking = mode == ControlMode::Tracking || mode == ControlMode::Braking;
        if (feedback->tracking) {
            feedback->total_waypoints = static_cast<uint32_t>(traj_points_.size());
            feedback->waypoints_remaining = static_cast<uint32_t>(traj_points_.size() - idx_);
        }
    }

    // (2) Publish or terminate after releasing the control mutex.
    if (mode == ControlMode::Idle) {
        finish_(goal, false, 0.0, "Trajectory cleared");
        return true;
    }
    if (mode == ControlMode::Complete) {
        RCLCPP_INFO(get_logger(), "Trajectory complete (error=%f)", feedback->current_error);
        if (max_error > get_parameter("max_tracking_error").as_double() &&
            !near_target_(state, target, goal->get_goal()->left_arm)) {
            char message[96];
            std::snprintf(message, sizeof(message),
                "Arm stopped with a joint %.3f rad from its commanded position", max_error);
            finish_(goal, false, feedback->current_error, message);
        } else {
            finish_(goal, true, feedback->current_error, "Target reached");
        }
        return true;
    }
    goal->publish_feedback(feedback);
    return false;
}

std::string MoveToPoseServer::execution_failure_(bool left)
{
    // (1) Old feedback ends the goal at once.
    char message[96];
    JointState state;
    Eigen::MatrixXd target;
    double max_error = 0.0;
    bool tracking = false;
    {
        std::lock_guard<std::mutex> lock(ctrl_mutex_);
        const double age = feedback_age_locked_();
        if (age > get_parameter("feedback_timeout").as_double()) {
            std::snprintf(message, sizeof(message), "Joint feedback is %.2f s old", age);
            return message;
        }
        tracking = mode_ == ControlMode::Tracking;
        max_error = max_error_;
        state = state_;
        target = planned_for_;
    }

    // (2) A joint error that stays high while tracking ends it too, except near the target, where pressing looks the same.
    if (!tracking || max_error <= get_parameter("max_tracking_error").as_double()) {
        off_track_since_ = {};
        return "";
    }
    if (near_target_(state, target, left)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "A joint is %.3f rad off near the target; continuing", max_error);
        off_track_since_ = {};
        return "";
    }
    const auto now = std::chrono::steady_clock::now();
    if (off_track_since_.time_since_epoch().count() == 0) off_track_since_ = now;
    if (std::chrono::duration<double>(now - off_track_since_).count()
            < get_parameter("tracking_error_timeout").as_double()) return "";
    std::snprintf(message, sizeof(message), "Arm is not following the trajectory: a joint is %.3f rad off", max_error);
    return message;
}

bool MoveToPoseServer::near_target_(const JointState & state, const Eigen::MatrixXd & target, bool left)
{
    // Measured gripper position, from the planner's model, against the installed target
    if (target.size() == 0) return false;
    const auto names = planner_->get_joint_names();
    Eigen::VectorXd q = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(names.size()));
    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto it = std::find(state.name.begin(), state.name.end(), names[i]);
        if (it != state.name.end()) q(static_cast<Eigen::Index>(i)) = state.position[std::distance(state.name.begin(), it)];
    }
    const Eigen::MatrixXd tip = planner_->forward_kinematics(q, left);
    return (tip.block<3, 1>(0, 3) - target.block<3, 1>(0, 3)).norm() <= get_parameter("near_target_distance").as_double();
}

void MoveToPoseServer::check_reach_(const Plan & plan, const Eigen::MatrixXd & target)
{
    // The goal configuration is the path sample nearest the target; any retreat after it only moves away
    double position = std::numeric_limits<double>::infinity();
    double angle = 0.0;
    for (const auto & pose : plan.path) {
        const double d = (pose.block<3, 1>(0, 3) - target.block<3, 1>(0, 3)).norm();
        if (d >= position) continue;
        position = d;
        const Eigen::Matrix3d dR = pose.block<3, 3>(0, 0).transpose() * target.block<3, 3>(0, 0);
        angle = std::abs(Eigen::AngleAxisd(dR).angle());
    }
    if (position <= get_parameter("goal_position_tolerance").as_double() &&
        angle <= get_parameter("goal_orientation_tolerance").as_double()) return;
    char message[128];
    std::snprintf(message, sizeof(message), "target out of reach: the arm ends %.1f mm and %.1f deg from it",
                  position * 1e3, angle * 180.0 / M_PI);
    throw std::runtime_error(message);
}

void MoveToPoseServer::record_plan_failure_(const std::shared_ptr<GoalHandle> & goal, const std::string & message)
{
    // A failed replan or step leaves the arm holding; repeated failures end the goal instead of holding forever
    const int limit = static_cast<int>(get_parameter("max_plan_failures").as_int());
    if (++plan_failures_ >= limit) {
        fail_(goal, message + " (" + std::to_string(plan_failures_) + " attempts)");
        return;
    }
    RCLCPP_WARN(get_logger(), "%s; holding position (attempt %d of %d)", message.c_str(), plan_failures_, limit);
}

Eigen::MatrixXd MoveToPoseServer::to_pelvis_frame_(const PoseStamped & pose)
{
    try {
        auto stamped = pose;
        stamped.header.stamp = rclcpp::Time(0);
        const auto out = tf_buffer_->transform(
            stamped, "pelvis",
            tf2::durationFromSec(get_parameter("tf_timeout").as_double()));
        Eigen::Isometry3d transformed;
        tf2::fromMsg(out.pose, transformed);
        transformed.translate(Eigen::Vector3d(
            get_parameter("arm_safety_distance_x").as_double(),
            get_parameter("arm_safety_distance_y").as_double(),
            get_parameter("arm_safety_distance_z").as_double()));
        return transformed.matrix();
    } catch (const tf2::TransformException & ex) {
        RCLCPP_ERROR(get_logger(), "TF lookup failed: %s", ex.what());
        return Eigen::MatrixXd();
    }
}

Eigen::MatrixXd MoveToPoseServer::pelvis_to_planning_frame_()
{
    if (frame_ == "pelvis") return Eigen::Matrix4d::Identity();
    try {
        const auto transform = tf_buffer_->lookupTransform(
            frame_, "pelvis", tf2::TimePointZero);
        return tf2::transformToEigen(transform).matrix();
    } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "TF lookup failed: %s", ex.what());
        return Eigen::MatrixXd();
    }
}

void MoveToPoseServer::publish_path_(const std::vector<Eigen::MatrixXd> & pelvis_path)
{
    const Eigen::MatrixXd transform = pelvis_to_planning_frame_();
    if (transform.size() == 0) return;
    std::vector<Eigen::MatrixXd> path;
    path.reserve(pelvis_path.size());
    for (const auto & pose : pelvis_path) path.push_back(transform * pose);
    path_pub_->publish(convertToPath(path, frame_, get_clock()->now()));
}

void MoveToPoseServer::publish_target_(const Eigen::MatrixXd & pelvis_target)
{
    // The target exactly as planned: in pelvis, after the safety offset and the gripper estimate correction
    const Eigen::Quaterniond turn(Eigen::Matrix3d(pelvis_target.block<3, 3>(0, 0)));
    PoseStamped message;
    message.header.stamp = get_clock()->now();
    message.header.frame_id = "pelvis";
    message.pose.position.x = pelvis_target(0, 3);
    message.pose.position.y = pelvis_target(1, 3);
    message.pose.position.z = pelvis_target(2, 3);
    message.pose.orientation.x = turn.x();
    message.pose.orientation.y = turn.y();
    message.pose.orientation.z = turn.z();
    message.pose.orientation.w = turn.w();
    target_pub_->publish(message);
}

} // namespace g1_pilot
