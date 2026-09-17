#include "core/arm_mp/visual_servo_planner.h"
#include "core/interfaces.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace HumanoidPilot{

namespace {

// Below this start-to-goal distance the approach offset is skipped and the
// goal is planned to directly.
constexpr double DIRECT_APPROACH_DISTANCE = 0.05;
constexpr int GOAL_REFINEMENT_PASSES = 2;

constexpr double REPLAN_LOCAL_POS = 0.01;   // m
constexpr double REPLAN_LOCAL_ROT = 0.05;   // rad

// How far off the cached path the arm may be and still rejoin it. The arm is
// normally tracking that very path, so this is a tracking-error budget rather
// than a planning tolerance.
// The arm executes a spline through the cached waypoints, not the polyline
// joining them, so it is legitimately some distance from the segments even
// when tracking perfectly. The connector back onto the path is collision
// checked, so this is a budget rather than a safety margin.
constexpr double REPLAN_PATH_DRIFT = 0.25;  // rad, over the whole configuration

// Below this the arm is on a waypoint for all practical purposes, so there is
// no connector to check and nothing to prepend.
constexpr double REJOIN_EPS = 1e-4;         // rad

constexpr double PATH_SEGMENT_EPS = 1e-12;

double elapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

// Timing lines are built on their own stream: whatever formatting the caller
// left on std::cout would otherwise turn millisecond counts into scientific
// notation.
std::ostringstream timingLine() {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    return out;
}

} // namespace

VisualServoPlanner::VisualServoPlanner(
    pinocchio::Model& model,
    std::shared_ptr<HumanoidIK> ik_handle,
    bool left_arm,
    double approach_offset,
    int final_leg_steps,
    double step_size,
    double goal_bias,
    double rewire_factor,
    double solve_time_seconds,
    double goal_tolerance,
    double validity_resolution,
    double cost_threshold_factor
) :
    JointSpacePlanner(model, ik_handle, left_arm, step_size, goal_bias,
                      rewire_factor, solve_time_seconds, goal_tolerance,
                      validity_resolution, cost_threshold_factor)
{
    intermediate_pose_offset_ = Eigen::Matrix4d::Identity();
    intermediate_pose_offset_(0, 3) = -approach_offset;

    // Final-leg cartesian poses in goal-relative frame: intermediate -> goal.
    // At t=0 we're at intermediate (offset back from goal); at t=1 we're at goal.
    final_leg_ = linearInterpolate(intermediate_pose_offset_,
                                   Eigen::Matrix4d::Identity(),
                                   final_leg_steps);

    leg_q_ = Eigen::VectorXd::Zero(nq_);
}

VisualServoPlanner::~VisualServoPlanner() {}

std::vector<Eigen::Matrix4d> VisualServoPlanner::linearInterpolate(
    const Eigen::Matrix4d& T_from,
    const Eigen::Matrix4d& T_to,
    int steps
) {
    std::vector<Eigen::Matrix4d> out(steps);
    Eigen::Vector3d p_from = T_from.block<3,1>(0,3);
    Eigen::Vector3d p_to   = T_to.block<3,1>(0,3);
    Eigen::Quaterniond q_from(T_from.block<3,3>(0,0));
    Eigen::Quaterniond q_to(T_to.block<3,3>(0,0));

    for (int k = 0; k < steps; ++k) {
        double t = (steps <= 1) ? 0.0 : static_cast<double>(k) / (steps - 1);
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,3>(0,0) = q_from.slerp(t, q_to).toRotationMatrix();
        T.block<3,1>(0,3) = p_from + t * (p_to - p_from);
        out[k] = T;
    }
    return out;
}

Eigen::Matrix4d VisualServoPlanner::startPose(const JointState& start_state) {
    std::tie(leg_q_, std::ignore, std::ignore) =
        jointstate_to_vectors(start_state, model_);

    return configToPose(leg_q_);
}

void VisualServoPlanner::appendFinalLeg(
    const Eigen::MatrixXd& goal_pose,
    std::vector<Eigen::VectorXd>& joints
) {
    JointState seed;
    bool have_seed = false;
    if (!joints.empty()) {
        seed = vectors_to_jointstate(
            joints.back(),
            Eigen::VectorXd::Zero(model_.nv),
            Eigen::VectorXd::Zero(model_.nv),
            model_
        );
        have_seed = true;
    }

    for (std::size_t i = 0; i < final_leg_.size(); ++i) {
        Eigen::Matrix4d T_abs = goal_pose * final_leg_[i];
        const int passes =
            (i + 1 == final_leg_.size()) ? GOAL_REFINEMENT_PASSES : 1;
        JointState js = poseToJointConfig(T_abs, passes, have_seed ? &seed : nullptr, true);
        if (js.position.size() == 0) {
            std::cout << "[VisualServoPlanner] final-leg IK failed; truncating" << std::endl;
            break;
        }

        std::tie(leg_q_, std::ignore, std::ignore) =
            jointstate_to_vectors(js, model_);

        joints.push_back(leg_q_);
        seed = js;
        have_seed = true;
    }
}

std::pair<
std::vector<Eigen::VectorXd>,
std::vector<Eigen::MatrixXd>>
VisualServoPlanner::planPath(
    const JointState& start_state,
    const Eigen::MatrixXd& goal_pose
) {
    Eigen::Matrix4d T_start = startPose(start_state);

    double pos_dist = (goal_pose.block<3,1>(0,3) - T_start.block<3,1>(0,3)).norm();

    if (pos_dist < DIRECT_APPROACH_DISTANCE) {
        return JointSpacePlanner::planPath(start_state, goal_pose);
    }

    Eigen::MatrixXd intermediate_mat = goal_pose * intermediate_pose_offset_;

    auto [joints, carts] = JointSpacePlanner::planPath(start_state, intermediate_mat);
    if (joints.empty()) return {};

    appendFinalLeg(goal_pose, joints);

    return {joints, configsToPoses(joints)};
}

JointTrajectory
VisualServoPlanner::planTrajectory(
    const JointState& start_state,
    const Eigen::MatrixXd& goal_pose,
    double time_step,
    double max_joint_vel,
    double max_joint_acc
) {
    const auto plan_t0 = std::chrono::steady_clock::now();

    Eigen::Matrix4d T_start = startPose(start_state);

    double pos_dist = (goal_pose.block<3,1>(0,3) - T_start.block<3,1>(0,3)).norm();

    if (pos_dist < DIRECT_APPROACH_DISTANCE) {
        JointTrajectory direct = JointSpacePlanner::planTrajectory(
            start_state, goal_pose, time_step, max_joint_vel, max_joint_acc);
        auto msg = timingLine();
        msg << "[planTrajectory] direct, no approach leg: "
            << elapsedMs(plan_t0) << " ms";
        std::cout << msg.str() << std::endl;
        return direct;
    }

    Eigen::MatrixXd intermediate_mat = goal_pose * intermediate_pose_offset_;

    // Phase 1: RRT* to the intermediate pose. Call the base planPath explicitly
    // so we don't recurse into our own override.
    JointState intermediate_state = solveGoalConfig(intermediate_mat, start_state, 2);
    if (intermediate_state.position.size() == 0) return {};

    const double ik_ms = elapsedMs(plan_t0);

    (void) JointSpacePlanner::planPath(start_state, intermediate_state);
    if (waypoints_.empty()) return {};

    // Phase 2 is appended as extra waypoints so both phases share a single
    // time parameterisation instead of meeting at a discontinuity.
    std::vector<Eigen::VectorXd> joints = waypoints_;

    const double search_ms = elapsedMs(plan_t0);

    last_search_ = waypoints_;
    last_goal_pose_ = goal_pose;
    last_start_q_ = joints.front();
    last_plan_left_ = left_arm_;
    last_plan_valid_ = true;

    appendFinalLeg(goal_pose, joints);

    const double leg_ms = elapsedMs(plan_t0) - search_ms;

    JointTrajectory traj =
        timeParameterize(joints, time_step, max_joint_vel, max_joint_acc);

    auto msg = timingLine();
    msg << "[planTrajectory] full plan: " << elapsedMs(plan_t0)
        << " ms (goal IK " << ik_ms
        << " ms, RRT* " << (search_ms - ik_ms)
        << " ms, approach leg " << leg_ms << " ms)";
    std::cout << msg.str() << std::endl;

    return traj;
}

JointTrajectory
VisualServoPlanner::replanTrajectory(
    const JointState& start_state,
    const Eigen::MatrixXd& goal_pose,
    double time_step,
    double max_joint_vel,
    double max_joint_acc
) {
    const auto replan_t0 = std::chrono::steady_clock::now();
    const Eigen::Matrix4d goal4 = goal_pose;

    // Why the previous search could not be reused; null means it could. Kept
    // alongside the measured drifts so the thresholds can be tuned against
    // what actually happens on the robot.
    const char* reason = nullptr;
    double goal_drift = 0.0;
    double start_drift = 0.0;
    std::size_t rejoin_index = 0;
    Eigen::VectorXd rejoin_q;
    Eigen::VectorXd start_vel;

    if (!last_plan_valid_ || last_search_.empty()) {
        reason = "no plan to reuse";
    } else if (last_plan_left_ != left_arm_) {
        reason = "other arm";
    } else {
        goal_drift =
            (goal4.block<3,1>(0,3) - last_goal_pose_.block<3,1>(0,3)).norm();

        if (!poseNear(last_goal_pose_, goal4, REPLAN_LOCAL_POS, REPLAN_LOCAL_ROT)) {
            reason = "goal moved too far";
        } else {
            Eigen::VectorXd start_q;
            Eigen::VectorXd vel;
            std::tie(start_q, vel, std::ignore) =
                jointstate_to_vectors(start_state, model_);
            start_q = start_q.cwiseMax(model_.lowerPositionLimit)
                             .cwiseMin(model_.upperPositionLimit);

            if (last_start_q_.size() != start_q.size()) {
                reason = "configuration size changed";
            } else {
                // The arm is somewhere along the path it is executing, which
                // after shortcutting may be only two waypoints long — so the
                // distance that matters is to the nearest point on a segment,
                // not to the nearest waypoint.
                start_drift = (start_q - last_search_.front()).norm();
                rejoin_index = 0;

                for (std::size_t i = 0; i + 1 < last_search_.size(); ++i) {
                    const Eigen::VectorXd seg = last_search_[i + 1] - last_search_[i];
                    const double seg_sq = seg.squaredNorm();

                    double t = 0.0;
                    if (seg_sq > PATH_SEGMENT_EPS) {
                        t = (start_q - last_search_[i]).dot(seg) / seg_sq;
                        t = std::min(std::max(t, 0.0), 1.0);
                    }

                    const double d =
                        (start_q - (last_search_[i] + t * seg)).norm();
                    if (d < start_drift) {
                        start_drift = d;
                        // Everything past this segment is still ahead of the arm.
                        rejoin_index = i + 1;
                    }
                }

                if (start_drift > REPLAN_PATH_DRIFT) {
                    reason = "arm is not on the cached path";
                } else if (start_drift > REJOIN_EPS
                           && !motionValid(start_q, last_search_[rejoin_index])) {
                    // Only the short hop back onto the path needs checking.
                    // The path itself was validated when it was planned, and
                    // re-checking it would cost more than the replan saves.
                    reason = "rejoining the cached path is blocked";
                } else {
                    rejoin_q = start_q;
                    start_vel = vel;
                }
            }
        }
    }

    if (reason != nullptr) {
        JointTrajectory traj = planTrajectory(
            start_state, goal_pose, time_step, max_joint_vel, max_joint_acc);

        auto msg = timingLine();
        msg << "[replanTrajectory] NOT tier 0 (" << reason << "): "
            << elapsedMs(replan_t0) << " ms  [goal drift "
            << std::setprecision(4) << goal_drift << " m / limit "
            << REPLAN_LOCAL_POS << ", arm drift " << start_drift
            << " rad / limit " << REPLAN_PATH_DRIFT << "]";
        std::cout << msg.str() << std::endl;
        return traj;
    }

    // Everything from where the arm is to the end of the cached search. The
    // measured configuration is prepended only when the arm has actually
    // drifted off a waypoint, so an unmoved arm does not get a zero-length
    // first segment.
    const auto leg_t0 = std::chrono::steady_clock::now();

    std::vector<Eigen::VectorXd> joints;
    joints.reserve(last_search_.size() - rejoin_index + final_leg_.size() + 1);
    if (start_drift > REJOIN_EPS) joints.push_back(rejoin_q);
    for (std::size_t i = rejoin_index; i < last_search_.size(); ++i) {
        joints.push_back(last_search_[i]);
    }

    const std::size_t reused = joints.size();

    appendFinalLeg(goal4, joints);

    const double leg_ms = elapsedMs(leg_t0);

    JointTrajectory traj = timeParameterize(
        joints, time_step, max_joint_vel, max_joint_acc, &start_vel);

    auto msg = timingLine();
    msg << "[replanTrajectory] tier 0, search reused: "
        << elapsedMs(replan_t0) << " ms (approach leg " << leg_ms
        << " ms)  [goal drift " << std::setprecision(4) << goal_drift
        << " m, off path " << start_drift << " rad, rejoined "
        << rejoin_index << "/" << last_search_.size() << ", " << reused
        << " waypoints reused]";
    std::cout << msg.str() << std::endl;

    return traj;
}

} // namespace HumanoidPilot
