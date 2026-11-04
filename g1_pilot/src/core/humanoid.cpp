#include "core/interfaces.h"

#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/frames.hpp>


#include "core/arm_control/arm_control.h"
#include "core/arm_ik/arm_ik.h"
#include "core/arm_mp/arm_mp.h"

namespace HumanoidPilot {

constexpr double COULOMB_VEL_EPS = 1e-3;

struct Humanoid::Impl {
    pinocchio::Model model_full;
    pinocchio::Model model;

    pinocchio::GeometryModel geom_full;
    pinocchio::GeometryModel geom;

    pinocchio::Data data;
    pinocchio::Data data_full;

    std::mutex data_mutex;

    Eigen::VectorXd q;
    Eigen::VectorXd v;
    Eigen::VectorXd e;
    
    Eigen::VectorXd grav_torques;
    Eigen::VectorXd a;
    Eigen::VectorXd ff_torques;

    Eigen::VectorXd armature;
    Eigen::VectorXd damping;
    Eigen::VectorXd friction;

    std::shared_ptr<HumanoidIK> ik;
    std::unique_ptr<ImpedanceController> controller;
    std::unique_ptr<VisualServoPlanner> motion_planner;

    double collision_margin = 0.0;

    void initialize(){
        data = pinocchio::Data(model);
        data_full = pinocchio::Data(model_full);
        ik = std::make_shared<HumanoidIK>(model, geom, collision_margin);
        // controller = std::make_unique<IKTrajTracker>(model, ik);
        controller = std::make_unique<ImpedanceController>(model);
        motion_planner = std::make_unique<VisualServoPlanner>(model, ik);
    }

    JointState grav_ff(JointState current_state){
        std::lock_guard<std::mutex> lock(data_mutex);

        std::tie(q,v,e) = jointstate_to_vectors(current_state, model);

        pinocchio::computeGeneralizedGravity(model, data, q);
        grav_torques = data.g;

        return vectors_to_jointstate(q,v,grav_torques, model);
    }

    JointState inverse_dynamics(
        const JointState& current_state,
        const std::vector<double>& acceleration
    ){
        std::lock_guard<std::mutex> lock(data_mutex);

        JointState padded = current_state;
        padded.position.resize(padded.name.size(), 0.0);
        padded.velocity.resize(padded.name.size(), 0.0);
        padded.effort.resize(padded.name.size(), 0.0);

        std::tie(q,v,e) = jointstate_to_vectors(padded, model);

        a = Eigen::VectorXd::Zero(model.nv);
        for (std::size_t i = 0; i < padded.name.size() && i < acceleration.size(); ++i) {
            const auto joint_id = model.getJointId(padded.name[i]);
            if (joint_id > 0 && joint_id < static_cast<std::size_t>(model.njoints)) {
                a(model.joints[joint_id].idx_v()) = acceleration[i];
            }
        }

        ff_torques = pinocchio::rnea(model, data, q, v, a);

        if (armature.size() == ff_torques.size()) {
            ff_torques.array() += armature.array() * a.array()
                                + damping.array() * v.array()
                                + friction.array() * (v.array() / COULOMB_VEL_EPS).tanh();
        }

        return vectors_to_jointstate(q, v, ff_torques, model);
    }
};

void reduce_model(
    std::vector<std::string> joint_names,
    pinocchio::Model &model_in,
    pinocchio::GeometryModel &gmodel_in,
    pinocchio::Model &model_out,
    pinocchio::GeometryModel &gmodel_out
){
    Eigen::VectorXd reference_config = Eigen::VectorXd::Zero(model_in.nq);
    std::vector<pinocchio::JointIndex> joints_idx;

    for (const auto& joint_name : joint_names) {
        if (model_in.existJointName(joint_name)) {
            joints_idx.push_back(
                model_in.getJointId(joint_name)
            );
        }
    }

    pinocchio::buildReducedModel(
        model_in, 
        gmodel_in,
        joints_idx, 
        reference_config, 
        model_out,
        gmodel_out
    );
}

Humanoid::Humanoid(
    const RobotConfig* robot_config
): impl_(std::make_unique<Impl>())
{
    robot_config_ = RobotConfig();
    if (robot_config != nullptr) {
        robot_config_ = *robot_config;
    }
    extract_config();

    pinocchio::urdf::buildModel(robot_config_.asset_file, impl_->model_full);
    pinocchio::urdf::buildGeom(
        impl_->model_full,
        robot_config_.asset_file,
        pinocchio::COLLISION,
        impl_->geom_full,
        robot_config_.asset_root
    );
    add_end_effector_frames();

    reduce_model(
        locked_joints_,
        impl_->model_full,
        impl_->geom_full,
        impl_->model,
        impl_->geom
    );

    apply_joint_limit_overrides();
    apply_joint_dynamics();
    impl_->collision_margin = collision_margin_;
    impl_->initialize();

}

Humanoid::~Humanoid() = default;

void Humanoid::apply_joint_limit_overrides() {
    for (const auto& [joint_name, lim] : joint_limit_overrides_) {
        if (!impl_->model.existJointName(joint_name)) {
            std::cerr
                << "[Humanoid] joint_limit_overrides references joint '"
                << joint_name << "' which is not in the reduced model "
                << "(probably locked) — skipping" << std::endl;
            continue;
        }
        int idx = impl_->model.joints[impl_->model.getJointId(joint_name)].idx_q();
        impl_->model.lowerPositionLimit(idx) = lim.lower;
        impl_->model.upperPositionLimit(idx) = lim.upper;
    }
}

void Humanoid::apply_joint_dynamics() {
    const auto& model = impl_->model;
    impl_->armature = Eigen::VectorXd::Zero(model.nv);
    impl_->damping  = Eigen::VectorXd::Zero(model.nv);
    impl_->friction = Eigen::VectorXd::Zero(model.nv);

    for (int joint_id = 1; joint_id < model.njoints; ++joint_id) {
        const auto it = joint_dynamics_.find(model.names[joint_id]);
        const JointDynamics& d =
            (it != joint_dynamics_.end()) ? it->second : joint_dynamics_default_;
        const int iv = model.joints[joint_id].idx_v();
        impl_->armature(iv) = d.armature;
        impl_->damping(iv)  = d.damping;
        impl_->friction(iv) = d.friction;
    }
}

void Humanoid::add_end_effector_frames() {
    auto add = [&](const std::string& frame_name, const EEFrame& ee) {
        auto parent_id = impl_->model_full.getJointId(ee.parent_joint);
        pinocchio::SE3 placement(Eigen::Matrix3d::Identity(), ee.offset);
        impl_->model_full.addFrame(pinocchio::Frame(
            frame_name, parent_id, placement, pinocchio::OP_FRAME
        ));
    };
    add("L_ee", left_ee_);
    add("R_ee", right_ee_);
}

JointState Humanoid::grav_ff(JointState current_state){
    return impl_->grav_ff(current_state);
}

JointState Humanoid::inverse_dynamics(
    const JointState& state,
    const std::vector<double>& acceleration
){
    return impl_->inverse_dynamics(state, acceleration);
}

std::vector<std::string> Humanoid::get_joint_names() {
    return std::vector<std::string>(
        impl_->model.names.begin() + 1, impl_->model.names.end());
}

JointState Humanoid::solve_ik(
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
){
    return impl_->ik->solve_ik(
        left_wrist, right_wrist,
        current_state,
        EE_efrc_L, EE_efrc_R,
        l_pos_err, l_rot_err,
        r_pos_err, r_rot_err,
        collision
    );
    // if (current_state==nullptr) return JointState();
    // return control_no_arms(*current_state);
}

JointState Humanoid::solve_ik(
    const Eigen::Matrix4d& wrist,
    const bool left,
    const JointState* current_state,
    const Eigen::VectorXd* EE_efrc,
    double* pos_err,
    double* rot_err,
    bool* collision,
    const bool use_dls
){
    return impl_->ik->solve_ik(
        wrist, left,
        current_state,
        EE_efrc,
        pos_err, rot_err,
        collision,
        use_dls
    );
}

void Humanoid::reset(){
    impl_->ik->reset();
}

bool Humanoid::check_collision(const Eigen::VectorXd& q){
    return impl_->ik->check_collision(q);
}

void Humanoid::update(JointState current_state){
    return impl_->controller->update(current_state);
}

Eigen::MatrixXd Humanoid::get_current_left_ee_pose(){
    return impl_->controller->get_current_left_ee_pose();
}
Eigen::MatrixXd Humanoid::get_current_right_ee_pose(){
    return impl_->controller->get_current_right_ee_pose();
}

Eigen::VectorXd Humanoid::get_current_left_ee_vel(){
    return impl_->controller->get_current_left_ee_vel();
}
Eigen::VectorXd Humanoid::get_current_right_ee_vel(){
    return impl_->controller->get_current_right_ee_vel();
}

double Humanoid::get_current_left_ee_error(){
    return impl_->controller->get_current_left_ee_error();
}
double Humanoid::get_current_right_ee_error(){
    return impl_->controller->get_current_right_ee_error();
}

JointState Humanoid::control_no_arms(JointState current_state){
    return impl_->controller->control_no_arms(current_state);
}

JointState Humanoid::control_both_arms(
    JointState current_state,
    Eigen::Matrix4d desired_l_ee_pose,
    Eigen::Matrix4d desired_r_ee_pose,
    Eigen::VectorXd desired_l_ee_vel,
    Eigen::VectorXd desired_r_ee_vel
){
    return impl_->controller->control_both_arms(
        current_state, 
        desired_l_ee_pose, desired_r_ee_pose,
        desired_l_ee_vel, desired_r_ee_vel
    );
}

JointState Humanoid::control_left_arm(
    JointState current_state,
    Eigen::Matrix4d desired_ee_pose,
    Eigen::VectorXd desired_ee_vel
){
    return impl_->controller->control_left_arm(
        current_state, desired_ee_pose,
        desired_ee_vel
    );
}

JointState Humanoid::control_right_arm(
    JointState current_state,
    Eigen::Matrix4d desired_ee_pose,
    Eigen::VectorXd desired_ee_vel
){
    return impl_->controller->control_right_arm(
        current_state, desired_ee_pose,
        desired_ee_vel
    );
}

std::pair<
std::vector<Eigen::VectorXd>,
std::vector<Eigen::MatrixXd>>
Humanoid::planPath(
    const Eigen::MatrixXd* goal_pose,
    const Eigen::MatrixXd* start_pose,
    const Eigen::VectorXd* /*start_vel*/,
    const Eigen::VectorXd* /*goal_vel*/,
    const Eigen::VectorXd* /*start_acc*/,
    const Eigen::VectorXd* /*goal_acc*/
){
    if (goal_pose == nullptr) {
        throw std::invalid_argument("Humanoid::planPath: goal_pose is null");
    }
    if (start_pose == nullptr) {
        throw std::invalid_argument("Humanoid::planPath: start_pose is null");
    }

    return impl_->motion_planner->planPath(*goal_pose, *start_pose);
}

JointTrajectory
Humanoid::planTrajectory(
    const Eigen::MatrixXd* goal_pose,
    const Eigen::MatrixXd* start_pose,
    double time_step,
    double max_joint_vel,
    double max_joint_acc
){
    if (goal_pose == nullptr) {
        throw std::invalid_argument("Humanoid::planTrajectory: goal_pose is null");
    }
    if (start_pose == nullptr) {
        throw std::invalid_argument("Humanoid::planTrajectory: start_pose is null");
    }

    return impl_->motion_planner->planTrajectory(
        *goal_pose, *start_pose,
        time_step, max_joint_vel, max_joint_acc
    );
}

JointTrajectory
Humanoid::planTrajectory(
    const JointState& start_state,
    const Eigen::MatrixXd& goal_pose,
    double time_step,
    double max_joint_vel,
    double max_joint_acc
){
    return impl_->motion_planner->planTrajectory(
        start_state, goal_pose,
        time_step, max_joint_vel, max_joint_acc
    ); 
}

JointTrajectory
Humanoid::replanTrajectory(
    const JointState& start_state,
    const Eigen::MatrixXd& goal_pose,
    double time_step,
    double max_joint_vel,
    double max_joint_acc
){
    return impl_->motion_planner->replanTrajectory(
        start_state, goal_pose,
        time_step, max_joint_vel, max_joint_acc
    );
}

JointTrajectory
Humanoid::planFinalLegStep(
    const JointState& start_state,
    const Eigen::MatrixXd& goal_pose,
    std::size_t step,
    bool retreat,
    double time_step,
    double max_joint_vel,
    double max_joint_acc
){
    return impl_->motion_planner->planFinalLegStep(
        start_state, goal_pose, step, retreat,
        time_step, max_joint_vel, max_joint_acc
    );
}

std::size_t Humanoid::finalLegSize() const {
    return impl_->motion_planner->finalLegSize();
}

JointTrajectory Humanoid::brake(
    const JointState& state,
    double time_step,
    double max_joint_vel,
    double max_joint_acc
){
    return impl_->motion_planner->brake(state, time_step, max_joint_vel, max_joint_acc);
}

Eigen::MatrixXd Humanoid::forward_kinematics(
    const Eigen::VectorXd& q,
    const bool left
){
    std::lock_guard<std::mutex> lock(impl_->data_mutex);

    pinocchio::framesForwardKinematics(impl_->model, impl_->data, q);

    return left
    ? impl_->data.oMf[impl_->model.getFrameId("L_ee")].toHomogeneousMatrix()
    : impl_->data.oMf[impl_->model.getFrameId("R_ee")].toHomogeneousMatrix();
}

Eigen::Matrix4d Humanoid::frame_transform(
    const JointState& state,
    const std::string& base_frame,
    const std::string& tip_frame
){
    std::lock_guard<std::mutex> lock(impl_->data_mutex);
    const auto& model = impl_->model_full;
    if (!model.existFrame(base_frame) || !model.existFrame(tip_frame)) {
        throw std::invalid_argument("Humanoid::frame_transform: unknown frame " + base_frame + " or " + tip_frame);
    }

    JointState padded = state;
    padded.velocity.resize(padded.name.size(), 0.0);
    padded.effort.resize(padded.name.size(), 0.0);
    const Eigen::VectorXd q = std::get<0>(jointstate_to_vectors(padded, model));

    pinocchio::framesForwardKinematics(model, impl_->data_full, q);
    const auto& oMb = impl_->data_full.oMf[model.getFrameId(base_frame)];
    const auto& oMt = impl_->data_full.oMf[model.getFrameId(tip_frame)];
    return oMb.actInv(oMt).toHomogeneousMatrix();
}

void Humanoid::setLeftArm(bool left_arm) { impl_->motion_planner->setLeftArm(left_arm); }

} // HumanoidPilot namespace
