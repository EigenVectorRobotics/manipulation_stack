#include "core/arm_mp/visual_servo_planner.h"
#include "core/interfaces.h"

#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <iterator>

namespace HumanoidPilot{

namespace {

// Below this start-to-goal distance the approach offset is skipped and the
// goal is planned to directly.
constexpr double DIRECT_APPROACH_DISTANCE = 0.05;
constexpr int PREGRASP_REFINEMENT_PASSES = 10;
constexpr int GOAL_REFINEMENT_PASSES = 5;

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
constexpr double PREGRASP_DELAY = 1.0;
constexpr double GOAL_DELAY = 1.0;

JointTrajectory concatenateTrajectories(
    const JointTrajectory& first,
    const JointTrajectory& second,
    double delay = 0.0
) {
    JointTrajectory out;
    out.name = first.name;

    if (!std::isfinite(delay) || delay < 0.0 ||
        (delay > 0.0 && (!std::isfinite(first.time_step) || first.time_step <= 0.0))) {
        throw std::invalid_argument("trajectory delay must be nonnegative with a positive time step");
    }
    const std::size_t delay_samples = delay <= 0.0 ? 0 :
        static_cast<std::size_t>(std::ceil(delay / first.time_step));

    auto append = [&out](std::size_t k, const JointTrajectory& traj){
        out.position[k].insert(out.position[k].end(), traj.position[k].begin(), traj.position[k].end());
        out.velocity[k].insert(out.velocity[k].end(), traj.velocity[k].begin(), traj.velocity[k].end());
        out.acceleration[k].insert(out.acceleration[k].end(), traj.acceleration[k].begin(), traj.acceleration[k].end());
    };

    for (std::size_t j = 0; j < first.name.size(); ++j) {
        append(j, first);
        out.position[j].insert(out.position[j].end(), delay_samples, second.position[j].front());
        out.velocity[j].insert(out.velocity[j].end(), delay_samples, 0.0);
        out.acceleration[j].insert(out.acceleration[j].end(), delay_samples, 0.0);
        append(j, second);
    }
    return out;
}

void linearInterpolate(
    const Eigen::Matrix4d& T_from,
    const Eigen::Matrix4d& T_to,
    int steps,
    std::vector<Eigen::Matrix4d>& out
) {

    if (static_cast<std::size_t>(steps) > out.capacity()) {
        throw std::invalid_argument("VisualServoPlanner::linearInterpolate: not enough space in output vector container."
            " Requested: " + std::to_string(steps) + ", but received " + std::to_string(out.capacity()));
    }
    out.resize(steps);

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
}

} // namespace

VisualServoPlanner::VisualServoPlanner(
    pinocchio::Model& model,
    std::shared_ptr<HumanoidIK> ik_handle,
    bool left_arm,
    double approach_offset,
    int max_final_leg_steps,
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
                      validity_resolution, cost_threshold_factor),
    approach_offset_(approach_offset), max_final_leg_steps_(max_final_leg_steps)
{
    intermediate_pose_offset_ = Eigen::Matrix4d::Identity();
    intermediate_pose_offset_(0, 3) = -approach_offset;

    // Final-leg cartesian poses in goal-relative frame: intermediate -> goal.
    // At t=0 we're at intermediate (offset back from goal); at t=1 we're at goal.
    final_leg_.reserve(max_final_leg_steps);
    approach_leg_.reserve(max_final_leg_steps);
    retreat_leg_.reserve(max_final_leg_steps);

}

VisualServoPlanner::~VisualServoPlanner() {}

void VisualServoPlanner::computeApproachLeg(
    const Eigen::MatrixXd& goal_pose,
    std::size_t leg_size
) {
    // (1) Calculate relative appraoch path
    approach_leg_.clear();
    linearInterpolate(
        intermediate_pose_offset_, Eigen::Matrix4d::Identity(),
        leg_size, final_leg_
    );

    // (2) Geet seed from prior
    JointState seed = waypoints_.empty() ? JointState() : 
        vectors_to_jointstate(
            waypoints_.back(),
            Eigen::VectorXd::Zero(model_.nv),
            Eigen::VectorXd::Zero(model_.nv),
            model_
        );
    bool have_seed = seed.position.size()>0;

    // (3) Calculating approach leg
    Eigen::VectorXd q;
    for (std::size_t i = 0; i < leg_size; ++i) {

        JointState js = poseToJointConfig(
            goal_pose * final_leg_[i], // goal
            (i + 1 == leg_size) ? GOAL_REFINEMENT_PASSES : 1, // passes or iterations
            have_seed ? &seed : nullptr, // seed joint state
            true // DLS based ik
        );

        if (js.position.size() == 0) {
            std::clog << "[VisualServoPlanner] IK failed; truncating" << std::endl;
            break;
        }

        std::tie(q, std::ignore, std::ignore) =
            jointstate_to_vectors(js, model_);

        approach_leg_.push_back(q);
        seed = js;
        have_seed = true;
    }
}

std::pair<double, std::size_t> VisualServoPlanner::closestOnPath(const Eigen::VectorXd& q){

    auto distance_to_segemnt = [](Eigen::VectorXd x, Eigen::VectorXd a, Eigen::VectorXd b){
        if ((a-b).norm() < PATH_SEGMENT_EPS) return (x-a).norm();

        const double progress = (x-a).dot(b-a);
        return (a + (b-a) * std::clamp(progress, 0.0, 1.0)).norm();
    };

    double tracking_drift = distance_to_segemnt(q, waypoints_[0], waypoints_[1]);
    std::size_t rejoin_index = 0;

    for (std::size_t i = 1; i < waypoints_.size() - 1; ++i) {
        const double d = distance_to_segemnt(q, waypoints_[i], waypoints_[i+1]);
        if (d < tracking_drift) {
            tracking_drift = d;
            rejoin_index = i+1;
        }
    }
    return {tracking_drift, rejoin_index};
}

JointTrajectory
VisualServoPlanner::planFinalLegStep(
    const JointState& start_state,
    const Eigen::MatrixXd& goal_pose,
    std::size_t step,
    int final_leg_steps,
    bool retreat,
    double time_step,
    double max_joint_vel,
    double max_joint_acc
) {
    if (step == 0 || step >= final_leg_steps) {
        throw std::out_of_range("VisualServoPlanner::planFinalLegStep: step outside the final leg");
    }

    Eigen::VectorXd start_q;
    std::tie(start_q, std::ignore, std::ignore) = jointstate_to_vectors(start_state, model_);
    start_q = start_q.cwiseMax(model_.lowerPositionLimit).cwiseMin(model_.upperPositionLimit);

    const JointState js = poseToJointConfig(
        goal_pose * final_leg_[step], 
        step + 1 == final_leg_steps ? GOAL_REFINEMENT_PASSES : 1, 
        &start_state, 
        true
    );
    if (js.position.size() == 0) return {};

    Eigen::VectorXd step_q;
    std::tie(step_q, std::ignore, std::ignore) = jointstate_to_vectors(js, model_);

    JointTrajectory traj = timeParameterize({start_q, step_q}, time_step, max_joint_vel/2, max_joint_acc/4);

    if (retreat) {
        computeApproachLeg(goal_pose, final_leg_steps);
        traj = concatenateTrajectories(traj, 
            timeParameterize(
                {approach_leg_.rbegin(), approach_leg_.rend()}, 
                time_step, max_joint_vel/2, max_joint_acc/4
            ),  
            PREGRASP_DELAY
        );
    }
    return traj;
}

std::vector<Eigen::VectorXd> VisualServoPlanner::constructPath(bool retreat) {
    std::vector<Eigen::VectorXd> joints;
    joints.insert(joints.end(), waypoints_.begin(), waypoints_.end());
    joints.insert(joints.end(), approach_leg_.begin(), approach_leg_.end());
    
    if (retreat){
        joints.insert(
            joints.end(), 
            std::make_reverse_iterator(approach_leg_.end()-1), 
            std::make_reverse_iterator(approach_leg_.begin()) 
        );
    }
    return joints;
}

JointTrajectory
VisualServoPlanner::constructTrajectory(
    bool retreat, double time_step,
    double max_joint_vel, double max_joint_acc
){
    // (1) Parameterize the paths and build the trajectory
    JointTrajectory traj = concatenateTrajectories(
        timeParameterize(waypoints_, time_step, max_joint_vel, max_joint_acc), 
        timeParameterize(approach_leg_, time_step, max_joint_vel/2, max_joint_acc/4),  
        PREGRASP_DELAY
    );

    traj.lock_time = traj.time_step * traj.position.size();
    
    if (!retreat){
        return traj;
    }

    // (2) Build retreat leg
    traj = concatenateTrajectories(traj, 
        timeParameterize(
            {approach_leg_.rbegin(), approach_leg_.rend()}, 
            time_step, max_joint_vel/2, max_joint_acc/4
        ),  
        PREGRASP_DELAY
    );
    
    return traj;
}

std::pair<
std::vector<Eigen::VectorXd>,
std::vector<Eigen::MatrixXd>>
VisualServoPlanner::planPath(
    const JointState& start_state,
    const Eigen::MatrixXd& goal_pose,
    int final_leg_steps,
    bool retreat
) {
    if (final_leg_steps > max_final_leg_steps_){
        throw std::invalid_argument("Asked for more number of final steps than the max steps set at initialization");
    }

    // (1) Compute first leg of the path
    setWall(goal_pose);
    (void) JointSpacePlanner::planPath(
        start_state, goal_pose * intermediate_pose_offset_
    );
    if (waypoints_.empty()) return {};

    // (2) Compute second leg of the path
    computeApproachLeg(goal_pose, final_leg_steps);

    //(3) Construct the output
    std::vector<Eigen::VectorXd> joints = constructPath(retreat);

    return {joints, configsToPoses(joints)};
}

JointTrajectory
VisualServoPlanner::planTrajectory(
    const JointState& start_state,
    const Eigen::MatrixXd& goal_pose,
    int final_leg_steps, bool retreat,
    double time_step, double max_joint_vel, double max_joint_acc
) {

    // Calculate the paths
    (void) planPath(start_state, goal_pose, final_leg_steps);
    if (waypoints_.empty()) return {};

    // update state
    last_goal_pose_ = goal_pose;
    last_plan_left_ = left_arm_;
    last_plan_valid_ = true;    

    // Construct output and return
    return constructTrajectory(retreat, time_step, max_joint_vel, max_joint_acc);
    
}

JointTrajectory
VisualServoPlanner::replanTrajectory(
    const JointState& start_state,
    const Eigen::MatrixXd& goal_pose,
    int final_leg_steps,
    bool retreat,
    double time_step,
    double max_joint_vel,
    double max_joint_acc
) {

    if (!last_plan_valid_
        || waypoints_.empty()           // First plan
        || last_plan_left_ != left_arm_ // Arm changed
        || !poseNear(last_goal_pose_, goal_pose, REPLAN_LOCAL_POS, REPLAN_LOCAL_ROT) // Goal moved a lot
        // || !poseNear(last_goal_pose_, goal_pose, approach_offset_, 3.14) // Away from pregrasp
    ) {
        return planTrajectory(
            start_state, goal_pose, final_leg_steps, retreat,
            time_step, max_joint_vel, max_joint_acc
        );
    }

    // Find closest point to exisitng path
    Eigen::VectorXd start_q;
    std::tie(start_q, std::ignore, std::ignore) = jointstate_to_vectors(start_state, model_);
    start_q = start_q.cwiseMax(model_.lowerPositionLimit).cwiseMin(model_.upperPositionLimit);

    auto [tracking_drift, rejoin_index] = closestOnPath(start_q); // _|_ perpendicular on path

    if (rejoin_index == waypoints_.size()-1){
        // TODO: Do something better close to the goal
        waypoints_.clear();
        computeApproachLeg(goal_pose, final_leg_steps);
        return constructTrajectory(retreat, time_step, max_joint_vel, max_joint_acc);
    }

    if (tracking_drift > REPLAN_PATH_DRIFT // Drifted too much from last path
        || tracking_drift > REJOIN_EPS && !motionValid(start_q, waypoints_[rejoin_index]) // No free way back
    ) {
        return planTrajectory(
            start_state, goal_pose, final_leg_steps, retreat,
            time_step, max_joint_vel, max_joint_acc
        );
    }
    
    // Remove waypoints already crossed
    waypoints_.erase(waypoints_.begin(), waypoints_.begin() + rejoin_index);
    
    if ((start_q - waypoints_.front()).norm() > REJOIN_EPS) {
        waypoints_.insert(waypoints_.begin(), start_q);
    }

    // Only computing the last leg again and build the output
    computeApproachLeg(goal_pose, final_leg_steps);
    return constructTrajectory(retreat, time_step, max_joint_vel, max_joint_acc);
}

} // namespace HumanoidPilot
