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
            double approach_offset = 0.10,
            int max_final_leg_steps = 30,
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
            const Eigen::MatrixXd& goal_pose,
            int final_leg_step = 10,
            bool retreat = false
        );

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

    protected:

        // Joint space computation
        void computeApproachLeg(const Eigen::MatrixXd& goal_pose, std::size_t leg_size);

        // Find closest point on path given a current configuration
        std::pair<double, std::size_t> closestOnPath(const Eigen::VectorXd& q);

        // Uses results stored in class variables to construct outputs
        std::vector<Eigen::VectorXd> constructPath(bool retreat);

        // Uses results stored in class variables to construct outputs
        JointTrajectory constructTrajectory(bool retreat, double time_step,
                                    double max_joint_vel, double max_joint_acc);

        double approach_offset_;

        // Containers needed to store the approach retreat trajectories
        int max_final_leg_steps_;
        Eigen::Matrix4d intermediate_pose_offset_;
        std::vector<Eigen::Matrix4d> final_leg_; // Stores cartesian space relative pose matrices

        std::vector<Eigen::VectorXd> approach_leg_; // Store joint space calculations for the last legs
        std::vector<Eigen::VectorXd> retreat_leg_;
        Eigen::Matrix4d last_goal_pose_ = Eigen::Matrix4d::Identity();

        // Replanning
        bool last_plan_valid_ = false;
        bool last_plan_left_ = false;
        
};

} // namespace HumanoidPilot

#endif // VISUAL_SERVO_PLANNER_H
