#ifndef G1_PILOT_MOVE_TO_POSE_SERVER_H
#define G1_PILOT_MOVE_TO_POSE_SERVER_H

#include <g1_pilot/g1_pilot.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <humanoid_manipulation_interfaces/action/move_to_pose.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace g1_pilot {

using HumanoidPilot::Humanoid;
using HumanoidPilot::JointState;
using JointStateMsg = sensor_msgs::msg::JointState;
using PoseStamped = geometry_msgs::msg::PoseStamped;
using Path = nav_msgs::msg::Path;
using Trajectory = trajectory_msgs::msg::JointTrajectory;
using TrajectoryPoint = trajectory_msgs::msg::JointTrajectoryPoint;
using MoveToPoseAction = humanoid_manipulation_interfaces::action::MoveToPose;
using MoveToPoseGoal = MoveToPoseAction::Goal;
using MoveToPoseFeedback = MoveToPoseAction::Feedback;
using MoveToPoseResult = MoveToPoseAction::Result;
using GoalHandle = rclcpp_action::ServerGoalHandle<MoveToPoseAction>;
using Trigger = std_srvs::srv::Trigger;

class MoveToPoseServer : public rclcpp::Node {
public:
    MoveToPoseServer();
    ~MoveToPoseServer() override;

private:
    enum class ControlMode { Idle, Tracking, Braking, Holding, Complete };
    enum class GoalStatus { Active, Preempted, Cancelled };
    struct Plan {
        Trajectory trajectory;
        std::vector<Eigen::MatrixXd> path;
        double lock_time = -1.0;
    };

    // Setup and goal lifecycle: server.cpp.
    void declare_parameters_();
    void create_models_();
    void create_publishers_();
    void create_action_server_();
    rclcpp_action::GoalResponse on_goal_(
        const rclcpp_action::GoalUUID &, std::shared_ptr<const MoveToPoseGoal> goal);
    rclcpp_action::CancelResponse on_cancel_(std::shared_ptr<GoalHandle> goal);
    void on_accepted_(std::shared_ptr<GoalHandle> goal);
    void on_cancel_service_(std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> response);
    GoalStatus goal_status_(const std::shared_ptr<GoalHandle> & goal);
    void finish_(const std::shared_ptr<GoalHandle> & goal,
                 bool success, double error, const std::string & message);
    void set_target_topic_(const std::string & topic);
    void clear_target_topic_();
    void on_live_target_(PoseStamped::SharedPtr msg);
    bool read_target_(PoseStamped & target);

    // Planning and frame conversion: planning.cpp.
    void worker_loop_();
    void run_(const std::shared_ptr<GoalHandle> & goal);
    void monitor_(const std::shared_ptr<GoalHandle> & goal);
    bool report_progress_(const std::shared_ptr<GoalHandle> & goal);
    Plan plan_(const JointState & start, const Eigen::MatrixXd & target, bool left, bool replan);
    Plan to_plan_(const HumanoidPilot::JointTrajectory & planned, bool left);
    void try_replan_(const std::shared_ptr<GoalHandle> & goal);
    Eigen::MatrixXd replan_target_();
    void start_stepping_(const JointState & state, const Eigen::MatrixXd & target);
    Eigen::Vector3d safety_offset_();
    Eigen::Vector3d gripper_estimate_error_(bool left, const Eigen::Matrix3d & target_rotation);
    void step_final_leg_(const std::shared_ptr<GoalHandle> & goal);
    Eigen::Matrix4d pelvis_from_anchor_(const JointState & state);
    Eigen::MatrixXd to_pelvis_frame_(const PoseStamped & pose);
    Eigen::MatrixXd pelvis_to_planning_frame_();
    void publish_path_(const std::vector<Eigen::MatrixXd> & pelvis_path);

    // Trajectory execution: control.cpp. Helpers ending in _locked_ require ctrl_mutex_.
    GoalStatus install_(const std::shared_ptr<GoalHandle> & goal,
                        const Trajectory & trajectory, const Eigen::MatrixXd & target, double lock_time);
    bool stop_for_replan_(const std::shared_ptr<GoalHandle> & goal, JointState & start);
    void clear_trajectory_();
    void clear_trajectory_locked_();
    void set_trajectory_locked_(Trajectory trajectory, ControlMode mode);
    JointState held_state_locked_();
    bool hold_endpoint_locked_();
    void control_tick_();
    void check_control_period_();
    void publish_poses_locked_(const Eigen::MatrixXd & transform);
    void track_trajectory_locked_();
    void complete_trajectory_locked_();
    std::vector<double> feedforward_torque_locked_(const TrajectoryPoint & point);
    double joint_error_locked_(const std::vector<double> & target);

    // The worker owns the planner; the control timer owns the controller model.
    std::unique_ptr<Humanoid> planner_;
    std::unique_ptr<Humanoid> controller_model_;
    std::string frame_;
    double control_period_ = 0.01;
    std::thread worker_;
    std::atomic<bool> stopping_{false};
    std::chrono::steady_clock::time_point last_tick_{};
    std::chrono::steady_clock::time_point last_replan_{};

    rclcpp::Publisher<JointStateMsg>::SharedPtr command_pub_;
    rclcpp::Publisher<PoseStamped>::SharedPtr left_ee_pub_;
    rclcpp::Publisher<PoseStamped>::SharedPtr right_ee_pub_;
    rclcpp::Publisher<Path>::SharedPtr path_pub_;
    rclcpp::Subscription<JointStateMsg>::SharedPtr feedback_sub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::CallbackGroup::SharedPtr cb_group_;
    rclcpp_action::Server<MoveToPoseAction>::SharedPtr action_server_;
    rclcpp::Service<Trigger>::SharedPtr cancel_service_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // ctrl_mutex_ protects feedback and every trajectory transition.
    std::mutex ctrl_mutex_;
    JointState state_;
    ControlMode mode_ = ControlMode::Idle;
    std::vector<std::string> traj_names_;
    std::vector<TrajectoryPoint> traj_points_;
    JointStateMsg hold_command_;
    size_t idx_ = 0;
    rclcpp::Time traj_start_;
    double error_ = 0.0;
    Eigen::MatrixXd planned_for_;
    // The planned target in its own frame, so body sway in the pelvis frame does not look like a moved target
    Eigen::MatrixXd planned_source_;
    std::string planned_source_frame_;
    double lock_time_ = -1.0;
    bool final_leg_refreshed_ = false;
    bool hold_between_steps_ = false;
    rclcpp::Time hold_since_;

    // Final-leg stepping (worker only): the goal frozen in the foot frame, re-mapped into the pelvis before every step
    bool stepping_ = false;
    std::size_t leg_step_ = 0;
    int leg_steps_ = 0;
    Eigen::Matrix4d goal_in_anchor_ = Eigen::Matrix4d::Identity();
    Eigen::Vector3d stepping_offset_ = Eigen::Vector3d::Zero();

    // target_mutex_ protects the target stream independently of control and goals.
    std::mutex target_mutex_;
    std::string target_topic_;
    rclcpp::Subscription<PoseStamped>::SharedPtr target_sub_;
    PoseStamped live_target_;
    bool have_live_target_ = false;
    uint64_t live_target_count_ = 0;

    // goal_mutex_ protects ownership; planning runs without either goal or control lock.
    std::mutex goal_mutex_;
    std::condition_variable cv_;
    std::shared_ptr<GoalHandle> active_;
    std::shared_ptr<GoalHandle> pending_;
    bool cancel_requested_ = false;
};

} // namespace g1_pilot

#endif // G1_PILOT_MOVE_TO_POSE_SERVER_H
