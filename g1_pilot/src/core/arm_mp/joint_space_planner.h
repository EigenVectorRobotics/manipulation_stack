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

        // Brake supplied joints from a valid state within max_joint_vel using quadratic deceleration and a terminal hold.
        static JointTrajectory brake(
            const JointState& state,
            double time_step = 0.01,
            double max_joint_vel = 1.0,
            double max_joint_acc = 2.0
        );

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

        /**
         * @param start_velocity  measured joint velocity at the first
         *        waypoint, or null to start from rest. Only the component
         *        along the path can be honoured: the trajectory begins at the
         *        path speed that best matches it, so a replan issued while the
         *        arm is moving hands over without a stop. Clamped to whatever
         *        still leaves room to brake to rest at the goal.
         */
        JointTrajectory timeParameterize(
            const std::vector<Eigen::VectorXd>& waypoints,
            double time_step,
            double max_joint_vel,
            double max_joint_acc,
            const Eigen::VectorXd* start_velocity = nullptr
        );

        void updateActiveJoints();

        JointState solveGoalConfig(
            const Eigen::MatrixXd& pose,
            const JointState& start_state,
            const int refinement_passes
        );

        /**
         * @brief True when the straight line between two configurations is
         *        collision free at the planner's validity resolution.
         */
        bool motionValid(const Eigen::VectorXd& a, const Eigen::VectorXd& b);

        // States whose end effector lies past the target's vertical plane are invalid.
        void setWall(const Eigen::Matrix4d& target_pose);

        /**
         * @brief Make the roadmap match the joints this query may move.
         *
         * The roadmap is sampled inside the slice of configuration space the
         * active arm can reach, so it survives from one query to the next: a
         * replanned goal is a fresh graph search, not a fresh search tree.
         * It is thrown away when the active joint set changes (the other arm),
         * and merely revalidated when the joints held fixed have moved, since
         * only edge validity depends on them.
         */
        void ensureSearchSpace(const Eigen::VectorXd& reference_q);

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

        // What the standing roadmap was built for.
        std::vector<bool> roadmap_q_moves_;
        Eigen::VectorXd roadmap_fixed_q_;
        bool roadmap_seeded_ = false;


        JointState cached_goal_state_;
        Eigen::Matrix4d cached_goal_pose_ = Eigen::Matrix4d::Identity();
        bool cached_goal_valid_ = false;
        bool cached_goal_left_ = false;

        Eigen::Matrix4d wall_pose_ = Eigen::Matrix4d::Identity();
        Eigen::Vector3d wall_normal_ = Eigen::Vector3d::UnitX();
        bool wall_set_ = false;
};

} // namespace HumanoidPilot

#endif // JOINT_SPACE_PLANNER_H
