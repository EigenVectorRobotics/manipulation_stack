#ifndef JOINT_SPACE_PLANNER_H
#define JOINT_SPACE_PLANNER_H

#include "core/arm_ik/arm_ik.h"

#include <memory>

namespace HumanoidPilot {

/**
 * @brief OMPL RRT*-based joint-space motion planner.
 *
 * Matches the PolynomialTrajectoryGenerator / VisualServoPlanner interface:
 * planTrajectory(...) and planPath(...) accept 4x4 SE(3) goal/start poses.
 * The provided HumanoidIK handle converts task-space poses into joint-space
 * configurations and provides collision checking; joint limits come from the
 * pinocchio model and are enforced by OMPL bounds and the IK solver.
 */
class JointSpacePlanner {
    public:
        JointSpacePlanner(
            pinocchio::Model& model,
            std::shared_ptr<HumanoidIK> ik_handle,
            bool left_arm = false,
            double step_size = 0.05,
            double goal_bias = 0.1,
            double rewire_factor = 1.1,
            double solve_time_seconds = 1.0,
            double goal_tolerance = 0.01,
            double validity_resolution = 0.005,
            double cost_threshold_factor = 2.0
        );
        virtual ~JointSpacePlanner();

        /**
         * @param goal_pose       4x4 SE(3) goal end-effector pose
         * @param start_pose      4x4 SE(3) start end-effector pose
         * @param time_step       output sampling period (s)
         * @param max_joint_vel   joint velocity bound (rad/s), all joints
         * @param max_joint_acc   joint acceleration bound (rad/s^2), all joints
         *
         * The RRT* path is fitted with a C2 cubic spline and time-optimally
         * parameterised against the two bounds, then sampled at time_step.
         */
        virtual JointTrajectory
        planTrajectory(
            const Eigen::MatrixXd& goal_pose,
            const Eigen::MatrixXd& start_pose,
            double time_step = 0.01,
            double max_joint_vel = 1.0,
            double max_joint_acc = 2.0
        );

        virtual JointTrajectory
        planTrajectory(
            const JointState& start_state,
            const Eigen::MatrixXd& goal_pose,
            double time_step = 0.01,
            double max_joint_vel = 1.0,
            double max_joint_acc = 2.0
        );

        virtual JointTrajectory
        planTrajectory(
            const JointState& start_state,
            const JointState& goal_state,
            double time_step = 0.01,
            double max_joint_vel = 1.0,
            double max_joint_acc = 2.0
        );

        virtual std::pair<
        std::vector<Eigen::VectorXd>,
        std::vector<Eigen::MatrixXd>>
        planPath(
            const Eigen::MatrixXd& goal_pose,
            const Eigen::MatrixXd& start_pose
        );

        virtual std::pair<
        std::vector<Eigen::VectorXd>,
        std::vector<Eigen::MatrixXd>>
        planPath(
            const JointState& start_state,
            const Eigen::MatrixXd& goal_pose
        );

        std::vector<Eigen::VectorXd>
        planPath(
            const JointState& start_state,
            const JointState& goal_state
        );

        void setLeftArm(bool left_arm) { left_arm_ = left_arm; }

    protected:
        bool runRRTStar(
            const Eigen::VectorXd& start_q,
            const Eigen::VectorXd& goal_q,
            std::vector<Eigen::VectorXd>& path
        );

        JointTrajectory timeParameterize(
            const std::vector<Eigen::VectorXd>& waypoints,
            double time_step,
            double max_joint_vel,
            double max_joint_acc
        );

        void updateActiveJoints();

        JointState solveGoalConfig(
            const Eigen::MatrixXd& pose,
            const JointState& start_state,
            const int refinement_passes
        );

        static bool poseNear(
            const Eigen::Matrix4d& a,
            const Eigen::Matrix4d& b,
            const double pos_tol,
            const double rot_tol
        );

        JointState poseToJointConfig(
            const Eigen::MatrixXd& pose,
            const int refinement_passes = 5,
            const JointState* reference_pose = nullptr,
            const bool use_dls = false
        );
        Eigen::Matrix4d configToPose(const Eigen::VectorXd& q);
        std::vector<Eigen::MatrixXd> configsToPoses(
            const std::vector<Eigen::VectorXd>& path
        );

        struct RRTData;
        std::unique_ptr<RRTData> rrt_;

        pinocchio::Model model_;
        pinocchio::Data data_;
        std::shared_ptr<HumanoidIK> ik_handle_;
        bool left_arm_;

        int nq_;
        pinocchio::FrameIndex left_ee_id_;
        pinocchio::FrameIndex right_ee_id_;

        double step_size_;
        double goal_bias_;
        double rewire_factor_;
        double solve_time_seconds_;
        double goal_tolerance_;
        double validity_resolution_;
        double cost_threshold_factor_;

        Eigen::VectorXd start_q_;
        Eigen::VectorXd goal_q_;
        std::vector<Eigen::VectorXd> waypoints_;

        std::vector<std::string> joint_names_;
        std::vector<int> joint_q_index_;

        std::vector<bool> joint_moves_;   // by pinocchio joint id
        std::vector<bool> q_moves_;       // by configuration index

        JointState cached_goal_state_;
        Eigen::Matrix4d cached_goal_pose_ = Eigen::Matrix4d::Identity();
        bool cached_goal_valid_ = false;
        bool cached_goal_left_ = false;
};

} // namespace HumanoidPilot

#endif // JOINT_SPACE_PLANNER_H
