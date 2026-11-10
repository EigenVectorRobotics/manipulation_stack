#include "server.h"
#include "../helper_funcs.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>

namespace g1_pilot {

void MoveToPoseServer::control_tick_()
{
    check_control_period_();
    std::lock_guard<std::mutex> lock(ctrl_mutex_);
    if (state_.name.empty()) return;
    if (mode_ == ControlMode::Holding) {
        hold_command_.header.stamp = get_clock()->now();
        command_pub_->publish(hold_command_);
    } else if (mode_ == ControlMode::Tracking || mode_ == ControlMode::Braking) {
        track_trajectory_locked_();
    }
}

MoveToPoseServer::GoalStatus MoveToPoseServer::install_(
    const std::shared_ptr<GoalHandle> & goal, const Trajectory & trajectory,
    const Eigen::MatrixXd & target, double lock_time)
{
    std::scoped_lock lock(goal_mutex_, ctrl_mutex_);
    if (active_ != goal) return GoalStatus::Preempted;
    if (goal->is_canceling()) {
        clear_trajectory_locked_();
        return GoalStatus::Cancelled;
    }
    set_trajectory_locked_(trajectory, ControlMode::Tracking);
    planned_for_ = target;
    lock_time_ = lock_time;
    return GoalStatus::Active;
}

bool MoveToPoseServer::stop_for_replan_(const std::shared_ptr<GoalHandle> & goal, JointState & start)
{
    // (1) Return the held endpoint only after braking completes.
    std::scoped_lock lock(goal_mutex_, ctrl_mutex_);
    if (active_ != goal || goal->is_canceling()) return false;
    if (mode_ == ControlMode::Braking || state_.name.empty()) return false;
    if (mode_ == ControlMode::Holding) {
        start = held_state_locked_();
        return true;
    }
    if (traj_points_.empty()) return false;

    // (2) Brake first; the replan starts from the held endpoint.
    brake_locked_();
    return false;
}

void MoveToPoseServer::brake_locked_()
{
    // Braking starts from the last commanded position and velocity, so the command stays continuous
    if (traj_points_.empty()) return;
    const auto & point = traj_points_[std::min(idx_, traj_points_.size() - 1)];
    const JointState brake_start{traj_names_, point.positions, point.velocities, {}};
    const auto brake = planner_->brake(
        brake_start, get_parameter("plan_time_step").as_double(),
        get_parameter("max_joint_vel").as_double(), get_parameter("max_joint_acc").as_double());
    set_trajectory_locked_(convertToTrajectory(brake, frame_, get_clock()->now()), ControlMode::Braking);
    Path empty_path;
    empty_path.header.frame_id = frame_;
    empty_path.header.stamp = traj_start_;
    path_pub_->publish(empty_path);
    RCLCPP_INFO(get_logger(), "Braking: %zu waypoints, %.3f s",
                traj_points_.size(), rclcpp::Duration(traj_points_.back().time_from_start).seconds());
}

double MoveToPoseServer::feedback_age_locked_()
{
    // An unstamped message counts as stale, so a publisher that leaves the header empty is caught at the first goal
    if (last_feedback_.nanoseconds() == 0) return std::numeric_limits<double>::infinity();
    return (get_clock()->now() - last_feedback_).seconds();
}

void MoveToPoseServer::clear_trajectory_()
{
    std::lock_guard<std::mutex> lock(ctrl_mutex_);
    clear_trajectory_locked_();
}

void MoveToPoseServer::clear_trajectory_locked_()
{
    traj_points_.clear();
    idx_ = 0;
    mode_ = ControlMode::Idle;
    hold_command_ = JointStateMsg();
}

void MoveToPoseServer::set_trajectory_locked_(Trajectory trajectory, ControlMode mode)
{
    traj_names_ = std::move(trajectory.joint_names);
    traj_points_ = std::move(trajectory.points);
    idx_ = 0;
    traj_start_ = get_clock()->now();
    mode_ = mode;
    hold_command_ = JointStateMsg();
    release_after_brake_ = false;
}

JointState MoveToPoseServer::held_state_locked_()
{
    JointState start = state_;
    start.velocity.assign(start.name.size(), 0.0);
    for (size_t i = 0; i < hold_command_.name.size(); ++i) {
        const auto it = std::find(start.name.begin(), start.name.end(), hold_command_.name[i]);
        if (it != start.name.end())
            start.position[std::distance(start.name.begin(), it)] = hold_command_.position[i];
    }
    return start;
}

bool MoveToPoseServer::hold_endpoint_locked_()
{
    if (traj_points_.empty() || traj_names_.empty()) return false;
    const auto & point = traj_points_.back();
    if (point.positions.size() != traj_names_.size()) {
        RCLCPP_ERROR(get_logger(), "Cannot hold: trajectory waypoint has missing joint positions");
        return false;
    }
    hold_command_ = JointStateMsg();
    hold_command_.name = traj_names_;
    hold_command_.position = point.positions;
    hold_command_.velocity.assign(traj_names_.size(), 0.0);
    auto stationary = point;
    stationary.velocities = hold_command_.velocity;
    stationary.accelerations.assign(traj_names_.size(), 0.0);
    hold_command_.effort = feedforward_torque_locked_(stationary);
    mode_ = ControlMode::Holding;
    traj_points_.clear();
    idx_ = 0;
    return true;
}

void MoveToPoseServer::track_trajectory_locked_()
{
    // (1) Select the current sample from elapsed trajectory time.
    if (traj_points_.empty() || idx_ >= traj_points_.size()) return;
    const double elapsed = (get_clock()->now() - traj_start_).seconds();
    const size_t last = traj_points_.size() - 1;
    while (idx_ < last && rclcpp::Duration(traj_points_[idx_ + 1].time_from_start).seconds() <= elapsed) {
        ++idx_;
    }
    const auto & point = traj_points_[idx_];
    update_errors_locked_(point.positions);

    // (2) Publish the command and handle the final sample.
    JointStateMsg cmd;
    cmd.header.stamp = get_clock()->now();
    cmd.name = traj_names_;
    cmd.position = point.positions;
    cmd.velocity = point.velocities;
    cmd.effort = feedforward_torque_locked_(point);
    command_pub_->publish(cmd);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Waypoint %zu/%zu, error %f",
                        idx_ + 1, traj_points_.size(), error_);
    if (idx_ == last && rclcpp::Duration(point.time_from_start).seconds() <= elapsed) {
        complete_trajectory_locked_();
    }
}

void MoveToPoseServer::complete_trajectory_locked_()
{
    if (mode_ == ControlMode::Braking && release_after_brake_) {
        clear_trajectory_locked_();
        RCLCPP_INFO(get_logger(), "Braking complete; released the arm");
    } else if (mode_ == ControlMode::Braking && hold_endpoint_locked_()) {
        RCLCPP_INFO(get_logger(), "Braking complete; holding endpoint for replanning");
    } else if (hold_between_steps_ && hold_endpoint_locked_()) {
        hold_since_ = get_clock()->now();
        RCLCPP_INFO(get_logger(), "Final-leg step reached; holding for the next step");
    } else {
        ++idx_;
        mode_ = ControlMode::Complete;
    }
    cv_.notify_all();
}

void MoveToPoseServer::check_control_period_()
{
    const auto tick = std::chrono::steady_clock::now();
    if (last_tick_.time_since_epoch().count() != 0) {
        const double dt = std::chrono::duration<double>(tick - last_tick_).count();
        if (dt > 5.0 * control_period_) {
            RCLCPP_WARN(get_logger(), "Control loop overran: %.1f ms since last tick", dt * 1e3);
        }
    }
    last_tick_ = tick;
}

std::vector<double> MoveToPoseServer::feedforward_torque_locked_(
    const TrajectoryPoint & point)
{
    // (1) Compute inverse dynamics for the commanded state.
    const size_t n = traj_names_.size();

    JointState query;
    query.name = traj_names_;
    query.position = point.positions;
    query.velocity = point.velocities;
    query.effort = point.effort;
    query.position.resize(n, 0.0);
    query.velocity.resize(n, 0.0);
    query.effort.resize(n, 0.0);

    std::vector<double> accel = point.accelerations;
    accel.resize(n, 0.0);

    const JointState ff = controller_model_->inverse_dynamics(query, accel);

    // (2) Return torque in trajectory joint order.
    std::unordered_map<std::string, double> by_name;
    for (size_t i = 0; i < ff.name.size() && i < ff.effort.size(); ++i) {
        by_name[ff.name[i]] = ff.effort[i];
    }

    std::vector<double> effort(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        auto it = by_name.find(traj_names_[i]);
        if (it != by_name.end()) effort[i] = it->second;
    }
    return effort;
}

void MoveToPoseServer::update_errors_locked_(const std::vector<double> & target)
{
    double sum_sq = 0.0;
    double largest = 0.0;
    size_t counted = 0;
    const size_t n = std::min(traj_names_.size(), target.size());
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < state_.name.size() && j < state_.position.size(); ++j) {
            if (traj_names_[i] == state_.name[j]) {
                const double d = target[i] - state_.position[j];
                sum_sq += d * d;
                largest = std::max(largest, std::abs(d));
                ++counted;
                break;
            }
        }
    }
    error_ = counted > 0 ? std::sqrt(sum_sq / counted) : 0.0;
    max_error_ = largest;
}

} // namespace g1_pilot
