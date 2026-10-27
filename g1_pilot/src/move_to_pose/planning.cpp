#include "server.h"
#include "../helper_funcs.h"

#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <stdexcept>

namespace g1_pilot {

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
        run_(goal);
    }
}

void MoveToPoseServer::run_(const std::shared_ptr<GoalHandle> & goal)
{
    // (1) Snapshot feedback and transform the requested pose into the core frame.
    last_replan_ = {};
    JointState start;
    {
        std::lock_guard<std::mutex> lock(ctrl_mutex_);
        start = state_;
    }
    if (start.name.empty()) {
        finish_(goal, false, 0.0, "No joint feedback received yet");
        return;
    }
    const auto request = goal->get_goal();
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
    }
    const auto status = install_(goal, plan.trajectory, target, plan.lock_time);
    if (status != GoalStatus::Active) {
        finish_(goal, false, 0.0, status == GoalStatus::Cancelled
            ? "Cancelled while planning" : "Preempted by a newer goal");
        return;
    }
    publish_path_(plan.path);
    RCLCPP_INFO(get_logger(), "Tracking trajectory");
    monitor_(goal);
}

MoveToPoseServer::Plan MoveToPoseServer::plan_(
    const JointState & start, const Eigen::MatrixXd & target, bool left, bool replan)
{
    planner_->setLeftArm(left);
    const double dt = get_parameter("plan_time_step").as_double();
    const double velocity = get_parameter("max_joint_vel").as_double();
    const double acceleration = get_parameter("max_joint_acc").as_double();
    const auto planned = replan
        ? planner_->replanTrajectory(start, target, dt, velocity, acceleration)
        : planner_->planTrajectory(start, target, dt, velocity, acceleration);
    const auto configs = toJointConfigs(planned);
    if (configs.empty()) throw std::runtime_error("empty trajectory");
    Plan result;
    result.path.reserve(configs.size());
    for (const auto & q : configs) result.path.push_back(planner_->forward_kinematics(q, left));
    result.trajectory = convertToTrajectory(planned, frame_, get_clock()->now());
    result.lock_time = planned.lock_time;
    if (result.trajectory.points.empty()) throw std::runtime_error("no waypoints");
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
    } catch (const std::exception & e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "Replan failed: %s; holding position", e.what());
        return;
    }
    if (install_(goal, plan.trajectory, target, plan.lock_time) != GoalStatus::Active) return;
    publish_path_(plan.path);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
        "Replanned to [%.3f %.3f %.3f], %zu waypoints",
        target(0, 3), target(1, 3), target(2, 3), plan.trajectory.points.size());
}

Eigen::MatrixXd MoveToPoseServer::replan_target_()
{
    // (1) Read the latest target when the replanning interval allows it.
    PoseStamped update;
    if (!read_target_(update)) return {};
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
    if (mode_ == ControlMode::Tracking && lock_time_ >= 0.0 &&
        (get_clock()->now() - traj_start_).seconds() >= lock_time_) {
        // The body shifts while the arm reaches out, so the final leg is replanned once against the latest target
        if (!final_leg_refreshed_) {
            final_leg_refreshed_ = true;
            planned_source_ = source;
            planned_source_frame_ = update.header.frame_id;
            RCLCPP_INFO(get_logger(), "Refreshing the final leg against the latest target");
            return target;
        }
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000, "Target fixed for the final leg");
        return {};
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
        try_replan_(goal);
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
    {
        std::lock_guard<std::mutex> lock(ctrl_mutex_);
        mode = mode_;
        if (mode == ControlMode::Holding) error_ = joint_error_locked_(hold_command_.position);
        feedback->current_error = error_;
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
        finish_(goal, true, feedback->current_error, "Target reached");
        return true;
    }
    goal->publish_feedback(feedback);
    return false;
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

} // namespace g1_pilot
