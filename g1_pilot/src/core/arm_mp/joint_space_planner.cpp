#include "core/arm_mp/joint_space_planner.h"
#include "core/interfaces.h"

#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <ompl/base/ProblemDefinition.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/ScopedState.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace ob = ompl::base;
namespace og = ompl::geometric;

namespace HumanoidPilot {

namespace {

constexpr std::size_t MAX_EXPECTED_WAYPOINTS = 256;

constexpr double PATH_EPS = 1e-9;
constexpr int GRID_PER_WAYPOINT = 20;
constexpr int MIN_GRID_POINTS = 200;
constexpr double FIXED_JOINT_RANGE = 1e-3;

constexpr double DLS_FALLBACK_POS_ERR = 1e-3;   // m
constexpr double DLS_FALLBACK_ROT_ERR = 1e-2;   // rad

constexpr double SEED_REUSE_POS = 0.15;   // m
constexpr double SEED_REUSE_ROT = 0.5;    // rad

/**
 * @brief Natural cubic spline through the waypoints, parameterised by
 * cumulative chord length in joint space.
 *
 * C2 continuity means q''(u) is continuous along the path, so the joint
 * acceleration only breaks where the time parameterisation itself switches
 * between accelerating and decelerating.
 */
class PathSpline {
    public:
        bool fit(const std::vector<Eigen::VectorXd>& waypoints);

        double length() const { return u_.empty() ? 0.0 : u_.back(); }

        void eval(
            double u,
            Eigen::VectorXd& q,
            Eigen::VectorXd& dq,
            Eigen::VectorXd& ddq
        ) const;

    private:
        std::vector<double> u_;
        std::vector<Eigen::VectorXd> q_;
        std::vector<Eigen::VectorXd> m_;
};

bool PathSpline::fit(const std::vector<Eigen::VectorXd>& waypoints) {
    u_.clear();
    q_.clear();
    m_.clear();
    if (waypoints.empty()) return false;

    u_.push_back(0.0);
    q_.push_back(waypoints.front());
    for (std::size_t i = 1; i < waypoints.size(); ++i) {
        const double d = (waypoints[i] - q_.back()).norm();
        if (d < PATH_EPS) continue;
        u_.push_back(u_.back() + d);
        q_.push_back(waypoints[i]);
    }

    const int n = static_cast<int>(q_.size());
    m_.assign(n, Eigen::VectorXd::Zero(q_.front().size()));
    if (n < 3) return true;

    // The tridiagonal system only depends on knot spacing, so every joint is
    // solved simultaneously with a vector right-hand side.
    std::vector<double> h(n - 1);
    for (int k = 0; k + 1 < n; ++k) h[k] = u_[k + 1] - u_[k];

    std::vector<double> c(n, 0.0);
    std::vector<Eigen::VectorXd> d(n, Eigen::VectorXd::Zero(q_.front().size()));

    for (int k = 1; k + 1 < n; ++k) {
        const Eigen::VectorXd rhs =
            6.0 * ((q_[k + 1] - q_[k]) / h[k] - (q_[k] - q_[k - 1]) / h[k - 1]);
        const double a = h[k - 1];
        const double b = 2.0 * (h[k - 1] + h[k]);
        const double denom = b - a * c[k - 1];
        if (std::abs(denom) < PATH_EPS) return false;
        c[k] = h[k] / denom;
        d[k] = (rhs - a * d[k - 1]) / denom;
    }

    for (int k = n - 2; k >= 1; --k) {
        m_[k] = d[k] - c[k] * m_[k + 1];
    }
    return true;
}

void PathSpline::eval(
    double u,
    Eigen::VectorXd& q,
    Eigen::VectorXd& dq,
    Eigen::VectorXd& ddq
) const {
    const int n = static_cast<int>(q_.size());
    if (n == 1) {
        q = q_[0];
        dq.setZero();
        ddq.setZero();
        return;
    }

    u = std::min(std::max(u, 0.0), u_.back());

    int k = static_cast<int>(
        std::upper_bound(u_.begin(), u_.end(), u) - u_.begin()) - 1;
    k = std::min(std::max(k, 0), n - 2);

    const double h = u_[k + 1] - u_[k];
    const double A = (u_[k + 1] - u) / h;
    const double B = (u - u_[k]) / h;

    q = A * q_[k] + B * q_[k + 1]
        + ((A * A * A - A) * m_[k] + (B * B * B - B) * m_[k + 1]) * (h * h / 6.0);
    dq = (q_[k + 1] - q_[k]) / h
        + (-(3.0 * A * A - 1.0) * m_[k] + (3.0 * B * B - 1.0) * m_[k + 1]) * (h / 6.0);
    ddq = A * m_[k] + B * m_[k + 1];
}

/**
 * @brief Feasible range of path acceleration at one grid point.
 *
 * Joint acceleration is q''(u) * sdot^2 + q'(u) * sddot, so each joint bounds
 * sddot from both sides; the intersection is what the profile may use.
 */
bool pathAccelBounds(
    const Eigen::VectorXd& dq,
    const Eigen::VectorXd& ddq,
    double sdot,
    double max_joint_acc,
    double& lo,
    double& hi
) {
    lo = -std::numeric_limits<double>::infinity();
    hi = std::numeric_limits<double>::infinity();

    const double s2 = sdot * sdot;
    for (int i = 0; i < dq.size(); ++i) {
        const double centripetal = ddq(i) * s2;
        if (std::abs(dq(i)) < PATH_EPS) {
            if (std::abs(centripetal) > max_joint_acc + PATH_EPS) return false;
            continue;
        }
        double b1 = (-max_joint_acc - centripetal) / dq(i);
        double b2 = (max_joint_acc - centripetal) / dq(i);
        if (b1 > b2) std::swap(b1, b2);
        lo = std::max(lo, b1);
        hi = std::min(hi, b2);
    }
    return lo <= hi;
}

} // namespace

struct JointSpacePlanner::RRTData {
    std::shared_ptr<ob::RealVectorStateSpace> space;
    ob::RealVectorBounds bounds;
    std::shared_ptr<ob::SpaceInformation> si;
    std::shared_ptr<ob::ProblemDefinition> pdef;
    std::shared_ptr<og::RRTstar> planner;
    std::shared_ptr<ob::PathLengthOptimizationObjective> objective;
    std::shared_ptr<og::PathSimplifier> simplifier;

    ob::ScopedState<ob::RealVectorStateSpace> start;
    ob::ScopedState<ob::RealVectorStateSpace> goal;

    Eigen::VectorXd q_check;

    double goal_tolerance;

    RRTData(
        int nq,
        const Eigen::VectorXd& lower_limit,
        const Eigen::VectorXd& upper_limit,
        double step_size,
        double goal_bias,
        double rewire_factor,
        double goal_tolerance_in,
        double validity_resolution
    ) :
        space(std::make_shared<ob::RealVectorStateSpace>(nq)),
        bounds(nq),
        si(std::make_shared<ob::SpaceInformation>(space)),
        pdef(std::make_shared<ob::ProblemDefinition>(si)),
        planner(std::make_shared<og::RRTstar>(si)),
        objective(std::make_shared<ob::PathLengthOptimizationObjective>(si)),
        simplifier(std::make_shared<og::PathSimplifier>(si)),
        start(space),
        goal(space),
        q_check(Eigen::VectorXd::Zero(nq)),
        goal_tolerance(goal_tolerance_in)
    {
        for (int i = 0; i < nq; ++i) {
            bounds.setLow(i, lower_limit(i));
            bounds.setHigh(i, upper_limit(i));
        }
        space->setBounds(bounds);

        si->setStateValidityCheckingResolution(validity_resolution);

        pdef->setOptimizationObjective(objective);

        planner->setRange(step_size);
        planner->setGoalBias(goal_bias);
        planner->setRewireFactor(rewire_factor);
        planner->setProblemDefinition(pdef);
    }

    void setBounds(const Eigen::VectorXd& lower, const Eigen::VectorXd& upper) {
        for (int i = 0; i < lower.size(); ++i) {
            bounds.setLow(i, lower(i));
            bounds.setHigh(i, upper(i));
        }
        space->setBounds(bounds);
        space->setup();
    }

    void setQuery(const Eigen::VectorXd& start_q, const Eigen::VectorXd& goal_q) {
        for (int i = 0; i < start_q.size(); ++i) {
            start[i] = start_q(i);
            goal[i] = goal_q(i);
        }
    }

    void reset() {
        pdef->clearSolutionPaths();
        pdef->clearStartStates();
        pdef->clearGoal();
        pdef->setStartAndGoalStates(start.get(), goal.get(), goal_tolerance);

        planner->clear();
        planner->setProblemDefinition(pdef);
    }
};

JointSpacePlanner::JointSpacePlanner(
    pinocchio::Model& model,
    std::shared_ptr<HumanoidIK> ik_handle,
    bool left_arm,
    double step_size,
    double goal_bias,
    double rewire_factor,
    double solve_time_seconds,
    double goal_tolerance,
    double validity_resolution,
    double cost_threshold_factor
) :
    model_(model),
    data_(model_),
    ik_handle_(ik_handle),
    left_arm_(left_arm),
    nq_(model.nq),
    left_ee_id_(model_.getFrameId("L_ee")),
    right_ee_id_(model_.getFrameId("R_ee")),
    step_size_(step_size),
    goal_bias_(goal_bias),
    rewire_factor_(rewire_factor),
    solve_time_seconds_(solve_time_seconds),
    goal_tolerance_(goal_tolerance),
    validity_resolution_(validity_resolution),
    cost_threshold_factor_(cost_threshold_factor)
{
    if (ik_handle_ == nullptr) {
        throw std::runtime_error("JointSpacePlanner: ik_handle is null");
    }

    start_q_ = Eigen::VectorXd::Zero(nq_);
    goal_q_ = Eigen::VectorXd::Zero(nq_);
    waypoints_.reserve(MAX_EXPECTED_WAYPOINTS);

    for (int joint_id = 1; joint_id < model_.njoints; ++joint_id) {
        joint_names_.push_back(model_.names[joint_id]);
        joint_q_index_.push_back(model_.joints[joint_id].idx_q());
    }

    rrt_ = std::make_unique<RRTData>(
        nq_,
        model_.lowerPositionLimit,
        model_.upperPositionLimit,
        step_size_,
        goal_bias_,
        rewire_factor_,
        goal_tolerance_,
        validity_resolution_
    );

    rrt_->si->setStateValidityChecker([this](const ob::State* s) {
        const auto* rv = s->as<ob::RealVectorStateSpace::StateType>();
        for (int i = 0; i < nq_; ++i) rrt_->q_check(i) = rv->values[i];
        if (ik_handle_ == nullptr) return true;
        return !ik_handle_->check_collision(rrt_->q_check);
    });
    rrt_->si->setup();
}

JointSpacePlanner::~JointSpacePlanner() {}

JointState JointSpacePlanner::poseToJointConfig(
    const Eigen::MatrixXd& pose,
    const int refinement_passes,
    const JointState* reference_pose,
    const bool use_dls
) {
    if (pose.rows() != 4 || pose.cols() != 4) {
        throw std::invalid_argument("JointSpacePlanner: pose must be 4x4 SE(3)");
    }

    double pos_err = 0.0;
    double rot_err = 0.0;

    const auto solve = [&](const bool dls) {
        JointState out = ik_handle_->solve_ik(
            pose, left_arm_,
            reference_pose, nullptr,
            &pos_err, &rot_err,
            nullptr, dls
        );

        for (int pass = 1; pass < refinement_passes; ++pass) {
            out = ik_handle_->solve_ik(
                pose, left_arm_,
                nullptr, nullptr,
                &pos_err, &rot_err,
                nullptr, dls
            );
        }
        return out;
    };

    JointState js = solve(use_dls);

    if (use_dls
        && (pos_err > DLS_FALLBACK_POS_ERR || rot_err > DLS_FALLBACK_ROT_ERR)) {
        std::cout << "[poseToJointConfig] DLS left pos err (m): " << pos_err
                  << ", rot err (rad): " << rot_err
                  << " — falling back to the optimisation solve" << std::endl;
        js = solve(false);
    }

    std::cout << "[poseToJointConfig] IK pos err (m): " << pos_err
              << ", rot err (rad): " << rot_err << std::endl;

    return js;
}

bool JointSpacePlanner::poseNear(
    const Eigen::Matrix4d& a,
    const Eigen::Matrix4d& b,
    const double pos_tol,
    const double rot_tol
) {
    if ((a.block<3,1>(0,3) - b.block<3,1>(0,3)).norm() > pos_tol) return false;

    const Eigen::Matrix3d dR = a.block<3,3>(0,0).transpose() * b.block<3,3>(0,0);
    return Eigen::AngleAxisd(dR).angle() <= rot_tol;
}

JointState JointSpacePlanner::solveGoalConfig(
    const Eigen::MatrixXd& pose,
    const JointState& start_state,
    const int refinement_passes
) {
    const Eigen::Matrix4d pose4 = pose;

    const bool reuse = cached_goal_valid_
        && cached_goal_left_ == left_arm_
        && poseNear(cached_goal_pose_, pose4, SEED_REUSE_POS, SEED_REUSE_ROT);

    JointState js = reuse
        ? poseToJointConfig(pose4, refinement_passes, &cached_goal_state_, true)
        : poseToJointConfig(pose4, refinement_passes, &start_state, false);

    if (js.position.size() != 0) {
        cached_goal_state_ = js;
        cached_goal_pose_ = pose4;
        cached_goal_left_ = left_arm_;
        cached_goal_valid_ = true;
    }

    return js;
}

bool JointSpacePlanner::runRRTStar(
    const Eigen::VectorXd& start_q,
    const Eigen::VectorXd& goal_q,
    std::vector<Eigen::VectorXd>& path
) {
    rrt_->setQuery(start_q, goal_q);

    if (!rrt_->si->isValid(rrt_->start.get())) {
        std::cout << "[runRRTStar] start state is in collision" << std::endl;
        path.resize(0);
        return false;
    }
    if (!rrt_->si->isValid(rrt_->goal.get())) {
        std::cout << "[runRRTStar] goal state is in collision" << std::endl;
        path.resize(0);
        return false;
    }

    rrt_->reset();

    const double direct = rrt_->si->distance(rrt_->start.get(), rrt_->goal.get());
    rrt_->objective->setCostThreshold(
        ob::Cost(std::max(cost_threshold_factor_ * direct, PATH_EPS)));

    ob::PlannerStatus status = rrt_->planner->solve(
        ob::timedPlannerTerminationCondition(solve_time_seconds_));
    if (status != ob::PlannerStatus::EXACT_SOLUTION) {
        std::cout << "[runRRTStar] no exact solution (status="
                  << status.asString() << ")" << std::endl;
        path.resize(0);
        return false;
    }

    auto path_geom = std::dynamic_pointer_cast<og::PathGeometric>(rrt_->pdef->getSolutionPath());
    if (!path_geom) {
        path.resize(0);
        return false;
    }

    rrt_->simplifier->ropeShortcutPath(*path_geom);
    rrt_->simplifier->smoothBSpline(*path_geom);

    const std::size_t state_count = path_geom->getStateCount();
    if (path.size() < state_count) path.resize(state_count);
    for (std::size_t i = 0; i < state_count; ++i) {
        const auto* rv = path_geom->getState(i)->as<ob::RealVectorStateSpace::StateType>();
        if (path[i].size() != nq_) path[i].resize(nq_);
        for (int j = 0; j < nq_; ++j) path[i](j) = rv->values[j];
    }
    path.resize(state_count);
    return true;
}

Eigen::Matrix4d JointSpacePlanner::configToPose(const Eigen::VectorXd& q) {
    pinocchio::framesForwardKinematics(model_, data_, q);

    return left_arm_
    ? data_.oMf[left_ee_id_].toHomogeneousMatrix()
    : data_.oMf[right_ee_id_].toHomogeneousMatrix();
}

std::vector<Eigen::MatrixXd> JointSpacePlanner::configsToPoses(
    const std::vector<Eigen::VectorXd>& path
) {
    std::vector<Eigen::MatrixXd> cart_space;
    cart_space.reserve(path.size());
    for (const auto& q : path) cart_space.push_back(configToPose(q));
    return cart_space;
}

JointTrajectory JointSpacePlanner::timeParameterize(
    const std::vector<Eigen::VectorXd>& waypoints,
    double time_step,
    double max_joint_vel,
    double max_joint_acc
) {
    JointTrajectory traj;
    traj.name = joint_names_;
    traj.time_step = time_step;

    const std::size_t n_joints = joint_names_.size();
    traj.position.assign(n_joints, {});
    traj.velocity.assign(n_joints, {});
    traj.acceleration.assign(n_joints, {});

    if (waypoints.empty()) return traj;
    if (time_step < PATH_EPS || max_joint_vel <= 0.0 || max_joint_acc <= 0.0) {
        throw std::invalid_argument(
            "JointSpacePlanner::timeParameterize: time_step and limits must be positive");
    }

    PathSpline spline;
    if (!spline.fit(waypoints)) return traj;

    Eigen::VectorXd q(nq_);
    Eigen::VectorXd dq(nq_);
    Eigen::VectorXd ddq(nq_);

    const double path_length = spline.length();
    if (path_length < PATH_EPS) {
        spline.eval(0.0, q, dq, ddq);
        for (std::size_t j = 0; j < n_joints; ++j) {
            traj.position[j].push_back(q(joint_q_index_[j]));
            traj.velocity[j].push_back(0.0);
            traj.acceleration[j].push_back(0.0);
        }
        return traj;
    }

    const int N = std::max(MIN_GRID_POINTS,
        GRID_PER_WAYPOINT * static_cast<int>(waypoints.size()));
    const double du = path_length / (N - 1);

    std::vector<Eigen::VectorXd> dq_grid(N, Eigen::VectorXd::Zero(nq_));
    std::vector<Eigen::VectorXd> ddq_grid(N, Eigen::VectorXd::Zero(nq_));
    std::vector<double> sdot_limit(N, 0.0);

    for (int k = 0; k < N; ++k) {
        spline.eval(k * du, q, dq_grid[k], ddq_grid[k]);
        const double max_abs = dq_grid[k].cwiseAbs().maxCoeff();
        sdot_limit[k] = (max_abs > PATH_EPS)
            ? max_joint_vel / max_abs
            : std::numeric_limits<double>::infinity();
    }

    // Backward pass: the fastest we may travel at each grid point and still be
    // able to brake to rest at the goal.
    std::vector<double> sdot_back(N, 0.0);
    for (int k = N - 1; k > 0; --k) {
        double sd = std::min(sdot_back[k], sdot_limit[k]);
        double lo = 0.0;
        double hi = 0.0;
        while (!pathAccelBounds(dq_grid[k], ddq_grid[k], sd, max_joint_acc, lo, hi)
               && sd > PATH_EPS) {
            sd *= 0.5;
        }
        sdot_back[k] = sd;
        if (!std::isfinite(lo)) lo = -max_joint_acc;
        const double next = sd * sd - 2.0 * lo * du;
        sdot_back[k - 1] = std::min(std::sqrt(std::max(next, 0.0)), sdot_limit[k - 1]);
    }

    // Forward pass: accelerate as hard as the limits allow without exceeding
    // the backward profile.
    std::vector<double> sdot(N, 0.0);
    sdot[0] = 0.0;
    for (int k = 0; k + 1 < N; ++k) {
        double sd = std::min(sdot[k], std::min(sdot_limit[k], sdot_back[k]));
        double lo = 0.0;
        double hi = 0.0;
        while (!pathAccelBounds(dq_grid[k], ddq_grid[k], sd, max_joint_acc, lo, hi)
               && sd > PATH_EPS) {
            sd *= 0.5;
        }
        sdot[k] = sd;
        if (!std::isfinite(hi)) hi = max_joint_acc;
        const double next = sd * sd + 2.0 * hi * du;
        sdot[k + 1] = std::min(std::sqrt(std::max(next, 0.0)),
                               std::min(sdot_limit[k + 1], sdot_back[k + 1]));
    }
    sdot[N - 1] = 0.0;

    std::vector<double> t(N, 0.0);
    for (int k = 0; k + 1 < N; ++k) {
        const double avg = 0.5 * (sdot[k] + sdot[k + 1]);
        t[k + 1] = t[k] + ((avg > PATH_EPS) ? du / avg : 0.0);
    }

    const double duration = t.back();
    const int n_samples =
        std::max(2, static_cast<int>(std::ceil(duration / time_step)) + 1);

    for (std::size_t j = 0; j < n_joints; ++j) {
        traj.position[j].reserve(n_samples);
        traj.velocity[j].reserve(n_samples);
        traj.acceleration[j].reserve(n_samples);
    }

    for (int i = 0; i < n_samples; ++i) {
        const double ti = std::min(i * time_step, duration);

        int k = static_cast<int>(
            std::upper_bound(t.begin(), t.end(), ti) - t.begin()) - 1;
        k = std::min(std::max(k, 0), N - 2);

        const double span = t[k + 1] - t[k];
        const double w = (span > PATH_EPS) ? (ti - t[k]) / span : 0.0;

        const double sd = sdot[k] + w * (sdot[k + 1] - sdot[k]);
        const double sdd = (span > PATH_EPS) ? (sdot[k + 1] - sdot[k]) / span : 0.0;

        spline.eval((k + w) * du, q, dq, ddq);

        for (std::size_t j = 0; j < n_joints; ++j) {
            const int qi = joint_q_index_[j];
            traj.position[j].push_back(q(qi));
            traj.velocity[j].push_back(dq(qi) * sd);
            traj.acceleration[j].push_back(dq(qi) * sdd + ddq(qi) * sd * sd);
        }
    }

    return traj;
}

std::pair<
std::vector<Eigen::VectorXd>,
std::vector<Eigen::MatrixXd>>
JointSpacePlanner::planPath(
    const Eigen::MatrixXd& goal_pose,
    const Eigen::MatrixXd& start_pose
) {
    std::cout << "IK for start pose:" << std::endl;
    JointState start_state = poseToJointConfig(start_pose, 3);
    if (start_state.position.size() == 0)
        return {};

    return planPath(start_state, goal_pose);
}

std::pair<
std::vector<Eigen::VectorXd>,
std::vector<Eigen::MatrixXd>>
JointSpacePlanner::planPath(
    const JointState& start_state,
    const Eigen::MatrixXd& goal_pose
) {
    std::cout << "IK for goal pose:" << std::endl;
    JointState goal_state = solveGoalConfig(goal_pose, start_state, 2);
    if (goal_state.position.size() == 0)
        return {};

    (void) planPath(start_state, goal_state);

    return {waypoints_, configsToPoses(waypoints_)};
}

void JointSpacePlanner::updateActiveJoints() {
    joint_moves_.assign(model_.njoints, false);
    q_moves_.assign(nq_, false);

    const auto ee_id = left_arm_ ? left_ee_id_ : right_ee_id_;
    for (auto j = model_.frames[ee_id].parentJoint; j != 0; j = model_.parents[j]) {
        const int idx_q = model_.joints[j].idx_q();
        const int nq_j = model_.joints[j].nq();

        double range = 0.0;
        for (int k = 0; k < nq_j; ++k) {
            range = std::max(range,
                model_.upperPositionLimit(idx_q + k) - model_.lowerPositionLimit(idx_q + k));
        }
        if (range <= FIXED_JOINT_RANGE) continue;

        joint_moves_[j] = true;
        for (int k = 0; k < nq_j; ++k) q_moves_[idx_q + k] = true;
    }
}

std::vector<Eigen::VectorXd>
JointSpacePlanner::planPath(
    const JointState& start_state,
    const JointState& goal_state
) {
    if (goal_state.position.size() == 0) {
        throw std::invalid_argument("JointSpacePlanner::planPath: goal pose is null");
    }

    if (start_state.position.size() == 0) {
        throw std::invalid_argument("JointSpacePlanner::planPath: start pose is null");
    }

    std::tie(start_q_, std::ignore, std::ignore) =
        jointstate_to_vectors(start_state, model_);
    std::tie(goal_q_, std::ignore, std::ignore) =
        jointstate_to_vectors(goal_state, model_);

    start_q_ = start_q_.cwiseMax(model_.lowerPositionLimit).cwiseMin(model_.upperPositionLimit);
    goal_q_ = goal_q_.cwiseMax(model_.lowerPositionLimit).cwiseMin(model_.upperPositionLimit);

    updateActiveJoints();

    Eigen::VectorXd lower = model_.lowerPositionLimit;
    Eigen::VectorXd upper = model_.upperPositionLimit;
    for (int i = 0; i < nq_; ++i) {
        if (q_moves_[i]) continue;
        goal_q_(i) = start_q_(i);
        lower(i) = start_q_(i);
        upper(i) = start_q_(i);
    }
    rrt_->setBounds(lower, upper);

    ik_handle_->set_active_joints(joint_moves_);
    const bool solved = runRRTStar(start_q_, goal_q_, waypoints_);
    ik_handle_->reset_active_collision_pairs();
    (void) solved;

    return waypoints_;
}

JointTrajectory
JointSpacePlanner::planTrajectory(
    const Eigen::MatrixXd& goal_pose,
    const Eigen::MatrixXd& start_pose,
    double time_step,
    double max_joint_vel,
    double max_joint_acc
) {
    std::cout << "IK for start pose:" << std::endl;
    JointState start_state = poseToJointConfig(start_pose, 3);
    if (start_state.position.size() == 0)
        return {};

    return planTrajectory(
        start_state, goal_pose, time_step, max_joint_vel, max_joint_acc);
}

JointTrajectory
JointSpacePlanner::planTrajectory(
    const JointState& start_state,
    const Eigen::MatrixXd& goal_pose,
    double time_step,
    double max_joint_vel,
    double max_joint_acc
){
    std::cout << "IK for goal pose:" << std::endl;
    JointState goal_state = solveGoalConfig(goal_pose, start_state, 2);
    if (goal_state.position.size() == 0)
        return {};

    return planTrajectory(
        start_state, goal_state, time_step, max_joint_vel, max_joint_acc);
}

JointTrajectory
JointSpacePlanner::planTrajectory(
    const JointState& start_state,
    const JointState& goal_state,
    double time_step,
    double max_joint_vel,
    double max_joint_acc
){
    (void) planPath(start_state, goal_state);

    return timeParameterize(waypoints_, time_step, max_joint_vel, max_joint_acc);
}

} // namespace HumanoidPilot
