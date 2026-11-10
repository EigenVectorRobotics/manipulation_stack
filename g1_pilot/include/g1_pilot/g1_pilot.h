#ifndef G1_PILOT_H
#define G1_PILOT_H

#include <Eigen/Dense>
#include <map>
#include <memory>

namespace HumanoidPilot {

struct JointState {
    std::vector<std::string> name;
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> effort;
};

/**
 * @brief A time-parameterised joint-space trajectory, sampled at a constant
 * time_step so no explicit time stamps are needed.
 *
 * position/velocity/acceleration are indexed [joint][sample]: the outer index
 * matches name, the inner index is the sample number, so each inner vector is
 * one joint's trajectory.
 */
struct JointTrajectory {
    std::vector<std::string> name;
    std::vector<std::vector<double>> position;
    std::vector<std::vector<double>> velocity;
    std::vector<std::vector<double>> acceleration;
    double time_step = 0.0;
    double lock_time = -1.0;  // seconds after which the target stays fixed; negative never fixes it
};

struct RobotConfig {
    std::string asset_file = "../assets/g1/g1_29dof_with_hand_rev_1_0.urdf";
    std::string asset_root = "../assets/g1/";
    std::string config_file = "../config/g1.yaml";
    int NUM_DOF = 29;
};

class Humanoid {

public:

    // A gripper frame fixed to the child link of parent_joint, offset in that link's frame.
    struct EEFrame {
        std::string parent_joint;
        std::string parent_link;
        Eigen::Vector3d offset = Eigen::Vector3d::Zero();
    };

    Humanoid(
        const RobotConfig* robot_config = nullptr
    );
    ~Humanoid();

    JointState grav_ff(JointState input);

    /**
     * @brief Full inverse-dynamics feedforward torque, tau = M(q)qdd + C(q,qd)qd + g(q).
     *
     * grav_ff covers only the g(q) term. Pass the trajectory's acceleration
     * sample to also feed forward the inertial and Coriolis terms. acceleration
     * is indexed like state.name; an empty vector reduces this to grav_ff.
     * The returned JointState lists every joint of the reduced model in
     * pinocchio order, which is not necessarily the caller's order.
     */
    JointState inverse_dynamics(
        const JointState& state,
        const std::vector<double>& acceleration = {}
    );

    std::vector<std::string> get_joint_names();

    JointState solve_ik(
        const Eigen::Matrix4d& left_wrist,
        const Eigen::Matrix4d& right_wrist,
        const JointState* current_state = nullptr,
        const Eigen::VectorXd* EE_efrc_L = nullptr,
        const Eigen::VectorXd* EE_efrc_R = nullptr,
        double* l_pos_err = nullptr,
        double* l_rot_err = nullptr,
        double* r_pos_err = nullptr,
        double* r_rot_err = nullptr,
        bool* collision = nullptr
    );

    JointState solve_ik(
        const Eigen::Matrix4d& wrist,
        const bool left = false,
        const JointState* current_state = nullptr,
        const Eigen::VectorXd* EE_efrc = nullptr,
        double* pos_err = nullptr,
        double* rot_err = nullptr,
        bool* collision = nullptr,
        const bool use_dls = false
    );

    bool check_collision(const Eigen::VectorXd& q);

    void reset();

    void update(JointState current_state);

    Eigen::VectorXd get_current_left_ee_vel();
    Eigen::VectorXd get_current_right_ee_vel();

    double get_current_left_ee_error();
    double get_current_right_ee_error();

    JointState control_no_arms(
        JointState current_state
    );

    JointState control_both_arms(
        JointState current_state,
        Eigen::Matrix4d desired_l_ee_pose,
        Eigen::Matrix4d desired_r_ee_pose,
        Eigen::VectorXd desired_l_ee_vel = Eigen::VectorXd::Zero(6),
        Eigen::VectorXd desired_r_ee_vel = Eigen::VectorXd::Zero(6)
    );

    JointState control_left_arm(
        JointState current_state,
        Eigen::Matrix4d desired_ee_pose,
        Eigen::VectorXd desired_ee_vel = Eigen::VectorXd::Zero(6)
    );

    JointState control_right_arm(
        JointState current_state,
        Eigen::Matrix4d desired_ee_pose,
        Eigen::VectorXd desired_ee_vel = Eigen::VectorXd::Zero(6)
    );

    // std::pair<
    // std::vector<Eigen::VectorXd>,
    // std::vector<Eigen::MatrixXd>>
    // planPath(
    //     const Eigen::MatrixXd* goal_pose,
    //     const Eigen::MatrixXd* start_pose = nullptr,
    //     const Eigen::VectorXd* start_vel = nullptr,
    //     const Eigen::VectorXd* goal_vel = nullptr,
    //     const Eigen::VectorXd* start_acc = nullptr,
    //     const Eigen::VectorXd* goal_acc = nullptr
    // );

    // JointTrajectory
    // planTrajectory(
    //     const Eigen::MatrixXd* goal_pose,
    //     const Eigen::MatrixXd* start_pose = nullptr,
    //     double time_step = 0.01,
    //     double max_joint_vel = 1.0,
    //     double max_joint_acc = 2.0
    // );

    JointTrajectory
    planTrajectory(
        const JointState& start_state,
        const Eigen::MatrixXd& goal_pose,
        int final_leg_steps = 10,
        bool retreat = false,
        double time_step = 0.01,
        double max_joint_vel = 1.0,
        double max_joint_acc = 2.0
    );

    JointTrajectory
    replanTrajectory(
        const JointState& start_state,
        const Eigen::MatrixXd& goal_pose,
        int final_leg_steps = 10,
        bool retreat = false,
        double time_step = 0.01,
        double max_joint_vel = 1.0,
        double max_joint_acc = 2.0
    );

    JointTrajectory
    planFinalLegStep(
        const JointState& start_state,
        const Eigen::MatrixXd& goal_pose,
        std::size_t step,
        int final_leg_steps,
        bool retreat = false,
        double time_step = 0.01,
        double max_joint_vel = 1.0,
        double max_joint_acc = 2.0
    );

    JointTrajectory brake(
        const JointState& state,
        double time_step = 0.01,
        double max_joint_vel = 1.0,
        double max_joint_acc = 2.0
    );

    Eigen::MatrixXd forward_kinematics(
        const Eigen::VectorXd& q,
        const bool left = false
    );

    /**
     * @brief base <- tip transform between two frames of the full model,
     * including the legs and waist that the reduced arm model locks.
     * Joints missing from state are held at zero.
     */
    Eigen::Matrix4d frame_transform(
        const JointState& state,
        const std::string& base_frame,
        const std::string& tip_frame
    );

    void setLeftArm(bool left_arm);

    // Why the last plan came back empty; empty when it succeeded.
    std::string plan_failure();

    const EEFrame& get_left_ee() const { return left_ee_; }
    const EEFrame& get_right_ee() const { return right_ee_; }

protected:

    struct Impl;
    std::unique_ptr<Impl> impl_;

    void extract_config();
    void add_end_effector_frames();
    void apply_joint_limit_overrides();
    void apply_joint_dynamics();

    RobotConfig robot_config_;

    EEFrame left_ee_;
    EEFrame right_ee_;

    std::vector<std::string> locked_joints_;

    /// Negative tolerates interpenetration; see collision_margin in g1.yaml.
    double collision_margin_ = 0.0;

    struct JointLimit {
        double lower;
        double upper;
    };
    std::map<std::string, JointLimit> joint_limit_overrides_;

    /// Motor-side dynamics absent from the URDF; see joint_dynamics in g1.yaml.
    struct JointDynamics {
        double armature = 0.0;
        double damping = 0.0;
        double friction = 0.0;
    };
    JointDynamics joint_dynamics_default_;
    std::map<std::string, JointDynamics> joint_dynamics_;

};


} // HumanoidPilot namespace


#endif // G1_PILOT_H
