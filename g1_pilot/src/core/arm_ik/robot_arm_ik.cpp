#include "core/arm_ik/robot_arm_ik.h"

#include <pinocchio/autodiff/casadi.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/collision/collision.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <casadi/casadi.hpp>
#include <coal/BVH/BVH_model.h>

#include <memory>
#include <fstream>
#include <set>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

// ============================================================================
// HumanoidIK Implementation
// ============================================================================

namespace HumanoidPilot {

/**
 * @brief Convert Eigen::VectorXd to casadi::DM
 */
inline casadi::DM eigen_to_casadi(const Eigen::VectorXd& vec) {
    std::vector<double> data(vec.data(), vec.data() + vec.size());
    return casadi::DM(data);
}

/**
 * @brief Convert Eigen::Matrix4d to casadi::DM
 */
inline casadi::DM eigen_to_casadi(const Eigen::Matrix4d& mat) {
    std::vector<double> data;
    data.reserve(16);
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            data.push_back(mat(row, col));
        }
    }
    return casadi::DM::reshape(casadi::DM(data), 4, 4);
}

inline pinocchio::SE3 eigen_to_pinocchio(const Eigen::Matrix4d& mat) {
    return pinocchio::SE3(mat);
}

struct HumanoidIK::CasadiData{
    casadi::Opti opti_l_;
    casadi::Opti opti_r_;
    casadi::Opti opti_;

    casadi::MX b_var_q_;
    casadi::MX b_var_q_last_;
    casadi::MX b_param_tf_l_;
    casadi::MX b_param_tf_r_;

    casadi::MX l_var_q_;
    casadi::MX l_var_q_last_;
    casadi::MX l_param_tf_l_;

    casadi::MX r_var_q_;
    casadi::MX r_var_q_last_;
    casadi::MX r_param_tf_r_;
};

HumanoidIK::HumanoidIK(
    pinocchio::Model& model,
    pinocchio::GeometryModel& geom_model,
    double collision_margin
) : model_(model), geom_model_(geom_model), cd(std::make_unique<CasadiData>())
{
    // Get required joint IDs    
    L_hand_id_ = model_.getFrameId("L_ee");
    R_hand_id_ = model_.getFrameId("R_ee");
    // oMcamera = model_.getFrameId("d435_link", pinocchio::BODY);
    // oMLidar = model_.getFrameId("mid360_link", pinocchio::BODY);

    // Setup Collision
    geom_model_.addAllCollisionPairs();
    filter_adjacent_collision_pairs();
    convexify_collision_geometry();
    filter_rest_collision_pairs();


    // Initialize data structures
    data_ = pinocchio::Data(model_);
    geom_data_ = pinocchio::GeometryData(geom_model_);

    // coal rejects a negative security_margin on mesh (BVH) geometry, so the
    // tolerance is applied to the reported penetration depth instead; that
    // needs contact points, which are off by default.
    for (auto& request : geom_data_.collisionRequests) {
        request.enable_contact = true;
        request.num_max_contacts = 4;
    }
    collision_margin_ = collision_margin;
    init_data_ = Eigen::VectorXd::Zero(model_.nq);

    nq_ = model_.nq;
    nv_ = model_.nv;

    constexpr double FIXED_JOINT_RANGE = 1e-3;
    dls_dof_mask_ = Eigen::VectorXd::Ones(model_.nv);
    for (int j = 1; j < model_.njoints; ++j) {
        const int idx_q = model_.joints[j].idx_q();
        const int idx_v = model_.joints[j].idx_v();
        for (int k = 0; k < model_.joints[j].nv(); ++k) {
            const double range = model_.upperPositionLimit(idx_q + k)
                               - model_.lowerPositionLimit(idx_q + k);
            if (range <= FIXED_JOINT_RANGE) dls_dof_mask_(idx_v + k) = 0.0;
        }
    }

    // Setup IK problem
    setup_optimization();

}

HumanoidIK::~HumanoidIK() = default;

bool HumanoidIK::check_collision(const Eigen::VectorXd& q) {
    // Colliding means hulls deeper than collision_margin_; stopAtFirst stays false as the first contact may be shallow.
    const bool any_contact = pinocchio::computeCollisions(
        model_, data_, geom_model_, geom_data_, q, collision_margin_ >= 0.0
    );
    if (!any_contact || collision_margin_ >= 0.0) return any_contact;

    for (std::size_t k = 0; k < geom_model_.collisionPairs.size(); ++k) {
        if (!geom_data_.activeCollisionPairs[k]) continue;
        const auto& result = geom_data_.collisionResults[k];
        if (!result.isCollision()) continue;
        for (std::size_t c = 0; c < result.numContacts(); ++c) {
            if (result.getContact(c).penetration_depth < collision_margin_) {
                return true;
            }
        }
    }
    return false;
}

void HumanoidIK::filter_adjacent_collision_pairs() {
    // Get kinematic adjacency from the model
    std::set<std::pair<int, int>> adjacent_pairs;
    for (int i = 1; i < model_.njoints; ++i) {
        adjacent_pairs.insert({model_.parents[i], i});
        adjacent_pairs.insert({i, model_.parents[i]});
    }
    
    // Filter out neighboring links
    std::vector<pinocchio::CollisionPair> filtered_pairs;
    for (const auto& cp : geom_model_.collisionPairs) {
        int link1 = geom_model_.geometryObjects[cp.first].parentJoint;
        int link2 = geom_model_.geometryObjects[cp.second].parentJoint;
        
        if (adjacent_pairs.find({link1, link2}) == adjacent_pairs.end()) {
            filtered_pairs.push_back(cp);
        }
    }
    
    geom_model_.collisionPairs = filtered_pairs;
}

void HumanoidIK::convexify_collision_geometry() {
    // Collision meshes are stored convex hulls; wrapped as convexes, they report true penetration depth.
    for (auto& object : geom_model_.geometryObjects) {
        auto mesh = std::dynamic_pointer_cast<coal::BVHModelBase>(object.geometry);
        if (!mesh) continue;
        mesh->buildConvexRepresentation(false);
        object.geometry = mesh->convex;
        object.geometry->computeLocalAABB();
    }
}

void HumanoidIK::filter_rest_collision_pairs() {
    // Hulls already overlapping with the arms hanging at rest would reject every plan from the natural pose.
    const Eigen::VectorXd rest = Eigen::VectorXd::Zero(model_.nq)
        .cwiseMax(model_.lowerPositionLimit)
        .cwiseMin(model_.upperPositionLimit);
    pinocchio::Data data(model_);
    pinocchio::GeometryData geom_data(geom_model_);
    pinocchio::computeCollisions(model_, data, geom_model_, geom_data, rest, false);

    std::vector<pinocchio::CollisionPair> kept;
    for (std::size_t k = 0; k < geom_model_.collisionPairs.size(); ++k) {
        if (!geom_data.collisionResults[k].isCollision()) kept.push_back(geom_model_.collisionPairs[k]);
    }
    geom_model_.collisionPairs = kept;
}

void HumanoidIK::set_active_joints(const std::vector<bool>& joint_moves) {
    // Joints between a body and the kinematic root.
    const auto ancestors = [this](pinocchio::JointIndex j) {
        std::set<pinocchio::JointIndex> out;
        while (j != 0) { out.insert(j); j = model_.parents[j]; }
        return out;
    };

    std::vector<std::set<pinocchio::JointIndex>> chain(model_.njoints);
    for (int j = 0; j < model_.njoints; ++j) {
        chain[j] = ancestors(static_cast<pinocchio::JointIndex>(j));
    }

    for (std::size_t k = 0; k < geom_model_.collisionPairs.size(); ++k) {
        const auto& cp = geom_model_.collisionPairs[k];
        const auto j1 = geom_model_.geometryObjects[cp.first].parentJoint;
        const auto j2 = geom_model_.geometryObjects[cp.second].parentJoint;

        bool can_change = false;
        for (const auto j : chain[j1]) {
            if (chain[j2].count(j)) continue;
            if (j < joint_moves.size() && joint_moves[j]) { can_change = true; break; }
        }
        if (!can_change) {
            for (const auto j : chain[j2]) {
                if (chain[j1].count(j)) continue;
                if (j < joint_moves.size() && joint_moves[j]) { can_change = true; break; }
            }
        }

        if (can_change) geom_data_.activateCollisionPair(k);
        else            geom_data_.deactivateCollisionPair(k);
    }
}

void HumanoidIK::reset_active_collision_pairs() {
    for (std::size_t k = 0; k < geom_model_.collisionPairs.size(); ++k) {
        geom_data_.activateCollisionPair(k);
    }
}

std::size_t HumanoidIK::active_collision_pair_count() const {
    std::size_t n = 0;
    for (const bool active : geom_data_.activeCollisionPairs) n += active ? 1 : 0;
    return n;
}

void HumanoidIK::setup_optimization() {

    pinocchio::ModelTpl<casadi::SX> cmodel(model_.cast<casadi::SX>());
    pinocchio::DataTpl<casadi::SX> cdata(cmodel);

    casadi::SX cq = casadi::SX::sym("q", cmodel.nq);
    casadi::SX cq_last = casadi::SX::sym("q", cmodel.nq);
    casadi::SX cTf_l = casadi::SX::sym("tf_l", 4, 4);
    casadi::SX cTf_r = casadi::SX::sym("tf_r", 4, 4);

    Eigen::Matrix<casadi::SX, Eigen::Dynamic, 1> cq_eigen(cmodel.nq);
    Eigen::Matrix<casadi::SX,4,4>  Target_L_eig;
    Eigen::Matrix<casadi::SX,4,4>  Target_R_eig;

    pinocchio::casadi::copy(cq, cq_eigen);
    pinocchio::casadi::copy(cTf_l, Target_L_eig);
    pinocchio::casadi::copy(cTf_r, Target_R_eig);

    auto L_hand_id = cmodel.getFrameId("L_ee");
    auto R_hand_id = cmodel.getFrameId("R_ee");

    pinocchio::framesForwardKinematics(cmodel, cdata, cq_eigen);
    pinocchio::SE3Tpl<casadi::SX> T_L = cdata.oMf[L_hand_id];
    pinocchio::SE3Tpl<casadi::SX> T_R = cdata.oMf[R_hand_id];

    pinocchio::SE3Tpl<casadi::SX>  Target_L(Target_L_eig);
    pinocchio::SE3Tpl<casadi::SX>  Target_R(Target_R_eig);

    casadi::SX Err_L(casadi::Sparsity::dense(6,1));
    casadi::SX Err_R(casadi::Sparsity::dense(6,1));

    pinocchio::casadi::copy(
        pinocchio::log6(T_L.actInv(Target_L)).toVector(),
        Err_L
    );

    pinocchio::casadi::copy(
        pinocchio::log6(T_R.actInv(Target_R)).toVector(),
        Err_R
    );

    std::vector<casadi_int> waist_q_idx;
    for (pinocchio::JointIndex j = 1; j < static_cast<pinocchio::JointIndex>(cmodel.njoints); ++j) {
        if (cmodel.names[j].find("waist") == std::string::npos) continue;
        const int idx_q = cmodel.idx_qs[j];
        const int nq_j = cmodel.nqs[j];
        for (int k = 0; k < nq_j; ++k) {
            waist_q_idx.push_back(static_cast<casadi_int>(idx_q + k));
        }
    }

    casadi::SX waist_cost = casadi::SX::zeros(1, 1);
    if (!waist_q_idx.empty()) {
        waist_cost = 0.1 * casadi::SX::sumsqr(cq(waist_q_idx));
    }

    casadi::SX left_trans_err = Err_L(casadi::Slice(0,3),0);
    casadi::SX right_trans_err = Err_R(casadi::Slice(0,3),0);
    casadi::SX left_rot_err = Err_L(casadi::Slice(3,6),0);
    casadi::SX right_rot_err = Err_R(casadi::Slice(3,6),0);

    casadi::SX both_arm_cost = (
        50.0 * (
            casadi::SX::sumsqr(left_trans_err) +
            casadi::SX::sumsqr(right_trans_err)
        ) + 
        3.0 * (
            casadi::SX::sumsqr(left_rot_err) +
            casadi::SX::sumsqr(right_rot_err)
        ) + 
        // 0.005 * casadi::SX::sumsqr(cq) +
        0.1 * casadi::SX::sumsqr(cq - cq_last) +
        waist_cost
    );

    casadi::SX left_arm_cost = (
        50.0 * casadi::SX::sumsqr(left_trans_err) + 
        3.0 * casadi::SX::sumsqr(left_rot_err) + 
        // 0.005 * casadi::SX::sumsqr(cq) +
        0.1 * casadi::SX::sumsqr(cq - cq_last) +
        waist_cost
    );

    casadi::SX right_arm_cost = (
        50.0 * casadi::SX::sumsqr(right_trans_err) + 
        3.0 * casadi::SX::sumsqr(right_rot_err) + 
        // 0.005 * casadi::SX::sumsqr(cq) +
        0.1 * casadi::SX::sumsqr(cq - cq_last) +
        waist_cost
    );

    // Functions
    casadi::Function both_arm_cost_f = casadi::Function(
        "both_arm_cost",
        {cq, cq_last, cTf_l, cTf_r},
        {both_arm_cost}
    );
    casadi::Function left_arm_cost_f = casadi::Function(
        "left_arm_cost",
        {cq, cq_last, cTf_l},
        {left_arm_cost}
    );
    casadi::Function right_arm_cost_f = casadi::Function(
        "right_arm_cost",
        {cq, cq_last, cTf_r},
        {right_arm_cost}
    );

    cd->b_var_q_ = cd->opti_.variable(model_.nq, 1);
    cd->b_var_q_last_ = cd->opti_.parameter(model_.nq, 1);
    cd->b_param_tf_l_ = cd->opti_.parameter(4, 4);
    cd->b_param_tf_r_ = cd->opti_.parameter(4, 4);

    cd->l_var_q_ = cd->opti_l_.variable(model_.nq, 1);
    cd->l_var_q_last_ = cd->opti_l_.parameter(model_.nq, 1);
    cd->l_param_tf_l_ = cd->opti_l_.parameter(4, 4);

    cd->r_var_q_ = cd->opti_r_.variable(model_.nq, 1);
    cd->r_var_q_last_ = cd->opti_r_.parameter(model_.nq, 1);
    cd->r_param_tf_r_ = cd->opti_r_.parameter(4, 4);
            
    casadi::DM lower_limits = eigen_to_casadi(model_.lowerPositionLimit);
    casadi::DM upper_limits = eigen_to_casadi(model_.upperPositionLimit);

    cd->opti_.subject_to(cd->opti_.bounded(lower_limits, cd->b_var_q_, upper_limits));
    cd->opti_l_.subject_to(cd->opti_l_.bounded(lower_limits, cd->l_var_q_, upper_limits));
    cd->opti_r_.subject_to(cd->opti_r_.bounded(lower_limits, cd->r_var_q_, upper_limits));

    // Minimize
    cd->opti_.minimize(both_arm_cost_f({
        cd->b_var_q_, cd->b_var_q_last_, cd->b_param_tf_l_, cd->b_param_tf_r_
    })[0]);
    cd->opti_l_.minimize(left_arm_cost_f({
        cd->l_var_q_, cd->l_var_q_last_, cd->l_param_tf_l_
    })[0]);
    cd->opti_r_.minimize(right_arm_cost_f({
        cd->r_var_q_, cd->r_var_q_last_, cd->r_param_tf_r_
    })[0]);
    
    // Set optimization options
    casadi::Dict opts;
    opts["expand"] = true;
    opts["detect_simple_bounds"] = true;
    opts["calc_lam_p"] = false;
    opts["print_time"] = false;
    opts["ipopt.sb"] = "yes";
    opts["ipopt.print_level"] = 0;
    opts["ipopt.max_iter"] = 50;
    opts["ipopt.tol"] = 1e-5;
    opts["ipopt.acceptable_tol"] = 1e-4;
    opts["ipopt.acceptable_iter"] = 3;
    opts["ipopt.warm_start_init_point"] = "yes";
    opts["ipopt.warm_start_bound_push"] = 1e-9;
    opts["ipopt.warm_start_slack_bound_push"] = 1e-9;
    opts["ipopt.warm_start_mult_bound_push"] = 1e-9;
    opts["ipopt.mu_strategy"] = "adaptive";
    opts["ipopt.derivative_test"] = "none";
    opts["ipopt.jacobian_approximation"] = "exact";
    
    cd->opti_.solver("ipopt", opts);
    cd->opti_l_.solver("ipopt", opts);
    cd->opti_r_.solver("ipopt", opts);

}

void HumanoidIK::reset() {
    init_data_ = Eigen::VectorXd::Zero(model_.nq);
}

Eigen::VectorXd HumanoidIK::solve_ik_dls(
    const Eigen::Matrix4d& target,
    const bool left
) {
    constexpr int MAX_ITERATIONS = 20;
    constexpr double TOLERANCE = 1e-6;      // on the 6D error twist
    constexpr double DAMPING = 1e-4;        // lambda^2
    constexpr double MAX_STEP = 0.3;        // rad, per iteration
    constexpr int MAX_BACKTRACKS = 6;

    const auto frame_id = left ? L_hand_id_ : R_hand_id_;
    const pinocchio::SE3 oMdes(target);

    const auto twist_error = [&](const Eigen::VectorXd& q_test) {
        pinocchio::framesForwardKinematics(model_, data_, q_test);
        return pinocchio::log6(data_.oMf[frame_id].actInv(oMdes)).toVector().norm();
    };

    Eigen::VectorXd q = init_data_;
    pinocchio::Data::Matrix6x J(6, model_.nv);
    pinocchio::Data::Matrix6 Jlog;
    Eigen::Matrix<double, 6, 6> JJt;
    Eigen::Matrix<double, 6, 1> err;
    Eigen::VectorXd dq(model_.nv);
    Eigen::VectorXd q_try(model_.nq);

    double err_norm = twist_error(q);

    for (int iter = 0; iter < MAX_ITERATIONS && err_norm > TOLERANCE; ++iter) {

        pinocchio::framesForwardKinematics(model_, data_, q);
        const pinocchio::SE3 eMd = data_.oMf[frame_id].actInv(oMdes);
        err = pinocchio::log6(eMd).toVector();

        J.setZero();
        pinocchio::computeFrameJacobian(model_, data_, q, frame_id,
                                        pinocchio::LOCAL, J);
        pinocchio::Jlog6(eMd.inverse(), Jlog);
        J = (-Jlog * J).eval();
        J = J * dls_dof_mask_.asDiagonal();

        for (int j = 1; j < model_.njoints; ++j) {
            const int idx_q = model_.joints[j].idx_q();
            const int idx_v = model_.joints[j].idx_v();
            for (int k = 0; k < model_.joints[j].nv(); ++k) {
                const double slack_lo = q(idx_q + k) - model_.lowerPositionLimit(idx_q + k);
                const double slack_hi = model_.upperPositionLimit(idx_q + k) - q(idx_q + k);
                if (slack_lo <= 0.0 && slack_hi <= 0.0) J.col(idx_v + k).setZero();
            }
        }

        JJt.noalias() = J * J.transpose();
        JJt.diagonal().array() += DAMPING;
        dq.noalias() = -J.transpose() * JJt.ldlt().solve(err);

        // A joint the step would drive past its limit contributes nothing but
        // a clamp. Drop it and let the others take up the slack, which is the
        // difference between converging and stalling a few millimetres out.
        bool dropped = false;
        for (int j = 1; j < model_.njoints; ++j) {
            const int idx_q = model_.joints[j].idx_q();
            const int idx_v = model_.joints[j].idx_v();
            for (int k = 0; k < model_.joints[j].nv(); ++k) {
                const double next = q(idx_q + k) + dq(idx_v + k);
                if (next > model_.upperPositionLimit(idx_q + k)
                    || next < model_.lowerPositionLimit(idx_q + k)) {
                    J.col(idx_v + k).setZero();
                    dropped = true;
                }
            }
        }

        if (dropped) {
            JJt.noalias() = J * J.transpose();
            JJt.diagonal().array() += DAMPING;
            dq.noalias() = -J.transpose() * JJt.ldlt().solve(err);
        }

        const double dq_norm = dq.norm();
        if (dq_norm > MAX_STEP) dq *= MAX_STEP / dq_norm;

        bool improved = false;
        double alpha = 1.0;
        for (int bt = 0; bt < MAX_BACKTRACKS; ++bt) {
            q_try = pinocchio::integrate(model_, q, alpha * dq);
            q_try = q_try.cwiseMax(model_.lowerPositionLimit)
                         .cwiseMin(model_.upperPositionLimit);

            const double try_norm = twist_error(q_try);
            if (try_norm < err_norm) {
                q = q_try;
                err_norm = try_norm;
                improved = true;
                break;
            }
            alpha *= 0.5;
        }

        if (!improved) break;
    }

    return q;
}

std::pair<double, double> HumanoidIK::compute_error(Eigen::Matrix4d A, Eigen::Matrix4d B){
    return {
        (A.block<3,1>(0,3) - B.block<3,1>(0,3)).norm(),
        std::acos(std::clamp(
            0.5 * ((
                A.block<3, 3>(0, 0) * B.block<3, 3>(0, 0).transpose()
            ).trace() - 1),
            -1.0,1.0
        ))
    };
}

JointState HumanoidIK::solve_ik(
    const Eigen::Matrix4d& left_wrist,
    const Eigen::Matrix4d& right_wrist,
    const JointState* current_state,
    const Eigen::VectorXd* EE_efrc_L,
    const Eigen::VectorXd* EE_efrc_R,
    double* l_pos_err,
    double* l_rot_err,
    double* r_pos_err,
    double* r_rot_err,
    bool* collision
)
{
    // Update initial guess
    if (current_state != nullptr) {
        std::tie(
            current_joint_pos_,
            current_joint_vel_,
            current_joint_eff_
        ) = jointstate_to_vectors(*current_state, model_);
        init_data_ = current_joint_pos_;
    }

    // Set optimization initial guess and parameters
    cd->opti_.set_initial(cd->b_var_q_, eigen_to_casadi(init_data_));
    cd->opti_.set_value(cd->b_param_tf_l_, eigen_to_casadi(left_wrist));
    cd->opti_.set_value(cd->b_param_tf_r_, eigen_to_casadi(right_wrist));
    cd->opti_.set_value(cd->b_var_q_last_, eigen_to_casadi(init_data_));

    // Solve optimization problem
    std::vector<double> sol_q_vec;

    try {

        // Solve optimization problem
        casadi::OptiSol sol = cd->opti_.solve();

        // Extract solution
        sol_q_vec = static_cast<std::vector<double>>(sol.value(cd->b_var_q_));

    } catch (const std::exception& e) {
        std::cerr << "ERROR in convergence: " << e.what() << std::endl;
        
        // Get debug solution
        sol_q_vec = static_cast<std::vector<double>>(
            cd->opti_.debug().value(cd->b_var_q_)
        );

    }

    sol_q = Eigen::Map<Eigen::VectorXd>(sol_q_vec.data(), model_.nq);

    pinocchio::framesForwardKinematics(model_, data_, sol_q);

    if (l_pos_err != nullptr || l_rot_err != nullptr) {
        auto [pe, re] = compute_error(
            data_.oMf[L_hand_id_].toHomogeneousMatrix(), left_wrist
        );
        if (l_pos_err) *l_pos_err = pe;
        if (l_rot_err) *l_rot_err = re;
    }
    if (r_pos_err != nullptr || r_rot_err != nullptr) {
        auto [pe, re] = compute_error(
            data_.oMf[R_hand_id_].toHomogeneousMatrix(), right_wrist
        );
        if (r_pos_err) *r_pos_err = pe;
        if (r_rot_err) *r_rot_err = re;
    }

    if (collision != nullptr) {
        *collision = pinocchio::computeCollisions(
            model_, data_, geom_model_, geom_data_, sol_q, true
        );
    }

    // Compute velocity
    sol_v = Eigen::VectorXd::Zero(model_.nv);
    
    // Compute feedforward torques using RNEA
    if (current_state != nullptr){
        sol_t = pinocchio::rnea(
            model_, data_, sol_q, sol_v,
            Eigen::VectorXd::Zero(model_.nv)
        );
    } else {
        sol_t = pinocchio::rnea(
            model_, data_, sol_q, sol_v,
            Eigen::VectorXd::Zero(model_.nv)
        );
    }

    generalize_ext_wrenches(EE_efrc_L, EE_efrc_R);

    JointState result = vectors_to_jointstate(
        sol_q, sol_v, sol_t, model_
    );
    init_data_ = sol_q;
    
    return result;

}

JointState HumanoidIK::solve_ik(
    const Eigen::Matrix4d& wrist,
    const bool left,
    const JointState* current_state,
    const Eigen::VectorXd* EE_efrc,
    double* pos_err,
    double* rot_err,
    bool* collision,
    const bool use_dls
)
{
    // Update initial guess
    if (current_state != nullptr) {
        std::tie(
            current_joint_pos_,
            current_joint_vel_,
            current_joint_eff_
        ) = jointstate_to_vectors(*current_state, model_);
        init_data_ = current_joint_pos_;
    }

    if (use_dls) {
        sol_q = solve_ik_dls(wrist, left);
    } else {

    if (left){
        cd->opti_l_.set_initial(cd->l_var_q_, eigen_to_casadi(init_data_));
        cd->opti_l_.set_value(cd->l_param_tf_l_, eigen_to_casadi(wrist));
        cd->opti_l_.set_value(cd->l_var_q_last_, eigen_to_casadi(init_data_));
    } else {
        cd->opti_r_.set_initial(cd->r_var_q_, eigen_to_casadi(init_data_));
        cd->opti_r_.set_value(cd->r_param_tf_r_, eigen_to_casadi(wrist));
        cd->opti_r_.set_value(cd->r_var_q_last_, eigen_to_casadi(init_data_));
    }

    // Solve optimization problem
    std::vector<double> sol_q_vec;

    try {
        if (left){
            casadi::OptiSol sol = cd->opti_l_.solve();
            sol_q_vec = static_cast<std::vector<double>>(sol.value(cd->l_var_q_));
        } else {
            casadi::OptiSol sol = cd->opti_r_.solve();
            sol_q_vec = static_cast<std::vector<double>>(sol.value(cd->r_var_q_));
        }

    } catch (const std::exception& e) {
        std::cerr << "ERROR in convergence: " << e.what() << std::endl;
        
        // Get debug solution
        if (left){
            sol_q_vec = static_cast<std::vector<double>>(
                cd->opti_l_.debug().value(cd->l_var_q_)
            );
        } else {
            sol_q_vec = static_cast<std::vector<double>>(
                cd->opti_r_.debug().value(cd->r_var_q_)
            );
        }
        
    }

    sol_q = Eigen::Map<Eigen::VectorXd>(sol_q_vec.data(), model_.nq);

    } // !use_dls

    pinocchio::framesForwardKinematics(model_, data_, sol_q);

    if (pos_err != nullptr || rot_err != nullptr) {
        auto frame_id = left ? L_hand_id_ : R_hand_id_;
        auto [pe, re] = compute_error(
            data_.oMf[frame_id].toHomogeneousMatrix(), wrist
        );
        if (pos_err) *pos_err = pe;
        if (rot_err) *rot_err = re;
    }

    if (collision != nullptr) {
        *collision = pinocchio::computeCollisions(
            model_, data_, geom_model_, geom_data_, sol_q, true
        );
    }

    // Compute velocity
    sol_v = Eigen::VectorXd::Zero(model_.nv);
    
    // Compute feedforward torques using RNEA
    if (current_state != nullptr){
        sol_t = pinocchio::rnea(
            model_, data_, current_joint_pos_, sol_v,
            Eigen::VectorXd::Zero(model_.nv)
        );
    } else {
        sol_t = pinocchio::rnea(
            model_, data_, sol_q, sol_v,
            Eigen::VectorXd::Zero(model_.nv)
        );
    }

    if (left) generalize_ext_wrenches(EE_efrc, nullptr);
    else generalize_ext_wrenches(nullptr, EE_efrc);

    JointState result = vectors_to_jointstate(
        sol_q, sol_v, sol_t, model_
    );
    init_data_ = sol_q;
    
    return result;

}

void HumanoidIK::generalize_ext_wrenches(
    const Eigen::VectorXd* EE_efrc_L,
    const Eigen::VectorXd* EE_efrc_R
){
    if (EE_efrc_L == nullptr && EE_efrc_R == nullptr)
        return;

    if (EE_efrc_L != nullptr){

        J_L = pinocchio::Data::Matrix6x::Zero(6, nv_);

        pinocchio::computeFrameJacobian(
            model_, data_, sol_q, 
            L_hand_id_, pinocchio::LOCAL_WORLD_ALIGNED, 
            J_L
        );

        sol_t += J_L.transpose() * (*EE_efrc_L);
    }

    if (EE_efrc_R != nullptr){

        J_R = pinocchio::Data::Matrix6x::Zero(6, nv_);

        pinocchio::computeFrameJacobian(
            model_, data_, sol_q, 
            R_hand_id_, pinocchio::LOCAL_WORLD_ALIGNED, 
            J_R
        );

        sol_t += J_R.transpose() * (*EE_efrc_R);
    }

    

}

} // HumanoidPilot namespace
