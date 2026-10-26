#ifndef VISUAL_SERVO_PLANNER_H
#define VISUAL_SERVO_PLANNER_H

#include "core/arm_mp/joint_space_planner.h"

namespace HumanoidPilot{

class VisualServoPlanner : public JointSpacePlanner {
    public:
        VisualServoPlanner(
            pinocchio::Model& model,
            std::shared_ptr<HumanoidIK> ik_handle,
            bool left_arm = false,
            double approach_offset = 0.05,
            int final_leg_steps = 8,
            double step_size = 0.05,
            double goal_bias = 0.1,
            double rewire_factor = 1.1,
            double solve_time_seconds = 1.0,
            double goal_tolerance = 0.01,
            double validity_resolution = 0.005,
            double cost_threshold_factor = 2.0
        );
        virtual ~VisualServoPlanner();

        using JointSpacePlanner::planPath;
        using JointSpacePlanner::planTrajectory;

        std::pair<
        std::vector<Eigen::VectorXd>,
        std::vector<Eigen::MatrixXd>>
        planPath(
            const JointState& start_state,
            const Eigen::MatrixXd& goal_pose
        ) override;

        JointTrajectory
        planTrajectory(
            const JointState& start_state,
            const Eigen::MatrixXd& goal_pose,
            double time_step = 0.01,
            double max_joint_vel = 1.0,
            double max_joint_acc = 2.0
        ) override;

        JointTrajectory
        replanTrajectory(
            const JointState& start_state,
            const Eigen::MatrixXd& goal_pose,
            double time_step = 0.01,
            double max_joint_vel = 1.0,
            double max_joint_acc = 2.0
        );

    protected:
        std::vector<Eigen::Matrix4d> linearInterpolate(
            const Eigen::Matrix4d& T_from,
            const Eigen::Matrix4d& T_to,
            int steps
        );

        Eigen::Matrix4d startPose(const JointState& start_state);

        void appendFinalLeg(
            const Eigen::MatrixXd& goal_pose,
            std::vector<Eigen::VectorXd>& joints,
            bool retreat = false
        );

        JointTrajectory appendRetreat(
            JointTrajectory traj,
            const Eigen::VectorXd& goal_q,
            const Eigen::MatrixXd& goal_pose,
            double time_step,
            double max_joint_vel,
            double max_joint_acc
        );

        Eigen::Matrix4d intermediate_pose_offset_;
        std::vector<Eigen::Matrix4d> final_leg_;
        Eigen::VectorXd leg_q_;

        std::vector<Eigen::VectorXd> last_search_;
        Eigen::Matrix4d last_goal_pose_ = Eigen::Matrix4d::Identity();
        Eigen::VectorXd last_start_q_;
        bool last_plan_valid_ = false;
        bool last_plan_left_ = false;
};

} // namespace HumanoidPilot

#endif // VISUAL_SERVO_PLANNER_H
