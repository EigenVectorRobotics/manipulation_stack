#include "core/arm_mp/visual_servo_planner.h"
#include "core/interfaces.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>

namespace HumanoidPilot{

namespace {

// Below this start-to-goal distance the approach offset is skipped and the
// goal is planned to directly.
constexpr double DIRECT_APPROACH_DISTANCE = 0.05;
constexpr int GOAL_REFINEMENT_PASSES = 2;

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
        JointState js = poseToJointConfig(T_abs, passes, have_seed ? &seed : nullptr);
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
    Eigen::Matrix4d T_start = startPose(start_state);

    double pos_dist = (goal_pose.block<3,1>(0,3) - T_start.block<3,1>(0,3)).norm();

    if (pos_dist < DIRECT_APPROACH_DISTANCE) {
        return JointSpacePlanner::planTrajectory(
            start_state, goal_pose, time_step, max_joint_vel, max_joint_acc);
    }

    Eigen::MatrixXd intermediate_mat = goal_pose * intermediate_pose_offset_;

    // Phase 1: RRT* to the intermediate pose. Call the base planPath explicitly
    // so we don't recurse into our own override.
    JointState intermediate_state = poseToJointConfig(intermediate_mat, 2, &start_state);
    if (intermediate_state.position.size() == 0) return {};

    (void) JointSpacePlanner::planPath(start_state, intermediate_state);
    if (waypoints_.empty()) return {};

    // Phase 2 is appended as extra waypoints so both phases share a single
    // time parameterisation instead of meeting at a discontinuity.
    std::vector<Eigen::VectorXd> joints = waypoints_;
    appendFinalLeg(goal_pose, joints);

    return timeParameterize(joints, time_step, max_joint_vel, max_joint_acc);
}

} // namespace HumanoidPilot
