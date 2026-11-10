#include <g1_pilot/g1_pilot.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "helper_funcs.h"
#include <tf2_ros/static_transform_broadcaster.h>

#include "humanoid_manipulation_interfaces/action/control_joint_trajectory.hpp"

#include <chrono>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <condition_variable>

using namespace HumanoidPilot;
using namespace std::chrono_literals;

using TrajectoryControllerAction = humanoid_manipulation_interfaces::action::ControlJointTrajectory;
using GoalHandleControl = rclcpp_action::ServerGoalHandle<TrajectoryControllerAction>;

class TrajectoryControllerActionServer : public rclcpp::Node
{
public:
    ~TrajectoryControllerActionServer()
    {
        shutting_down_.store(true);
        goal_cv_.notify_all();
        if (execute_thread_.joinable()) {
            execute_thread_.join();
        }
    }

    TrajectoryControllerActionServer() : Node("trajectory_controller_action_server")
    {
        /* Initialize arm handle */
        std::string package_share_directory = ament_index_cpp::get_package_share_directory("g1_description");
        std::string g1_pilot_share_directory = ament_index_cpp::get_package_share_directory("g1_pilot");

        std::string default_asset_file = package_share_directory + "/assets/g1/g1_29dof_with_hand_rev_1_0.urdf";
        std::string default_asset_root = package_share_directory + "/assets/g1/";
        std::string default_config_file = g1_pilot_share_directory + "/config/g1.yaml";
        int default_n_dof = 29;

        this->declare_parameter<std::string>("asset_file", default_asset_file);
        this->declare_parameter<std::string>("asset_root", default_asset_root);
        this->declare_parameter<std::string>("config_file", default_config_file);
        this->declare_parameter<int>("num_dof", default_n_dof);

        RobotConfig config;
        config.asset_file = this->get_parameter("asset_file").as_string();
        config.asset_root = this->get_parameter("asset_root").as_string();
        config.config_file = this->get_parameter("config_file").as_string();
        config.NUM_DOF = this->get_parameter("num_dof").as_int();

        arm_handle_ = std::make_unique<Humanoid>(&config);

        /* Publishers */
        this->declare_parameter<std::string>("position_control_topic", "position_control");

        joint_states_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            this->get_parameter("position_control_topic").as_string(), 10);

        ee_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        ee_broadcaster_->sendTransform(ee_static_transforms(*arm_handle_, this->get_clock()->now()));

        /* Subscribers */
        this->declare_parameter<std::string>("feedback_topic", "feedback");
        feedback_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            this->get_parameter("feedback_topic").as_string(), 10,
            std::bind(&TrajectoryControllerActionServer::handle_new_feedback_, this, std::placeholders::_1));

        /* Action server */
        action_server_ = rclcpp_action::create_server<TrajectoryControllerAction>(
            this,
            "trajectory_controller",
            std::bind(&TrajectoryControllerActionServer::handle_goal_, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&TrajectoryControllerActionServer::handle_cancel_, this, std::placeholders::_1),
            std::bind(&TrajectoryControllerActionServer::handle_accepted_, this, std::placeholders::_1));

        /* Control loop timer */
        this->declare_parameter<double>("control_period", 0.01);
        const auto control_period = std::chrono::duration<double>(
            this->get_parameter("control_period").as_double());
        control_loop_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(control_period),
            std::bind(&TrajectoryControllerActionServer::controller_, this));

        RCLCPP_INFO(this->get_logger(), "Trajectory controller action server ready");
    }

private:
    // Publishers
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_states_pub_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> ee_broadcaster_;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr feedback_sub_;

    // Timers
    rclcpp::TimerBase::SharedPtr control_loop_;

    // Action server
    rclcpp_action::Server<TrajectoryControllerAction>::SharedPtr action_server_;

    // Arm handle
    std::unique_ptr<Humanoid> arm_handle_;

    // Trajectory and control state
    std::vector<std::string> trajectory_joint_names_;
    std::vector<trajectory_msgs::msg::JointTrajectoryPoint> trajectory_points_;
    trajectory_msgs::msg::JointTrajectoryPoint current_target_;
    size_t next_waypoint_idx_ = 0;
    double error_ = 0.0;
    rclcpp::Time traj_start_;
    JointState current_state_;

    // Mutexes
    std::mutex feedback_mutex_;
    std::mutex goal_mutex_;

    // Active goal tracking
    std::shared_ptr<GoalHandleControl> active_goal_handle_;
    uint32_t active_total_waypoints_ = 0;
    std::condition_variable goal_cv_;

    // Thread management
    std::thread execute_thread_;
    std::atomic<bool> shutting_down_{false};

    // --- Action server callbacks ---
    rclcpp_action::GoalResponse handle_goal_(
        const rclcpp_action::GoalUUID & /*uuid*/,
        std::shared_ptr<const TrajectoryControllerAction::Goal> goal)
    {
        RCLCPP_INFO(this->get_logger(), "Received trajectory with %zu points",
                    goal->trajectory.points.size());

        {
            std::lock_guard<std::mutex> lock(goal_mutex_);
            trajectory_joint_names_ = goal->trajectory.joint_names;
            trajectory_points_ = goal->trajectory.points;
            next_waypoint_idx_ = 0;
            active_total_waypoints_ = trajectory_points_.size();
            traj_start_ = this->get_clock()->now();

            if (active_goal_handle_ && active_goal_handle_->is_active()) {
                RCLCPP_INFO(this->get_logger(), "Preempting active goal");
                auto result = std::make_shared<TrajectoryControllerAction::Result>();
                result->success = false;
                result->final_error = -2.0;
                active_goal_handle_->abort(result);
                active_goal_handle_ = nullptr;
            }
        }
        goal_cv_.notify_all();

        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel_(
        const std::shared_ptr<GoalHandleControl> /*goal_handle*/)
    {
        RCLCPP_INFO(this->get_logger(), "Received cancel request");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_success_(const std::shared_ptr<GoalHandleControl> goal_handle)
    {
        auto result = std::make_shared<TrajectoryControllerAction::Result>();
        result->success = true;
        result->final_error = error_;
        goal_handle->succeed(result);
        active_goal_handle_ = nullptr;
        RCLCPP_INFO(this->get_logger(), "Goal succeeded with error: %f", result->final_error);
    }

    void handle_accepted_(const std::shared_ptr<GoalHandleControl> goal_handle)
    {
        if (execute_thread_.joinable()) {
            execute_thread_.join();
        }
        execute_thread_ = std::thread(
            &TrajectoryControllerActionServer::execute_goal_, this, goal_handle);
    }

    void execute_goal_(const std::shared_ptr<GoalHandleControl> goal_handle)
    {
        {
            std::lock_guard<std::mutex> lock(goal_mutex_);
            active_goal_handle_ = goal_handle;
        }

        while (rclcpp::ok() && !shutting_down_.load()) {
            std::unique_lock<std::mutex> lock(goal_mutex_);

            if (active_goal_handle_ != goal_handle) {
                RCLCPP_INFO(this->get_logger(), "Goal preempted, exiting execute thread");
                return;
            }

            if (goal_handle->is_canceling()) {
                trajectory_points_.clear();
                next_waypoint_idx_ = 0;
                auto result = std::make_shared<TrajectoryControllerAction::Result>();
                result->success = false;
                result->final_error = -1.0;
                goal_handle->canceled(result);
                active_goal_handle_ = nullptr;
                RCLCPP_INFO(this->get_logger(), "Goal canceled");
                return;
            }

            if (next_waypoint_idx_ >= trajectory_points_.size()) {
                handle_success_(goal_handle);
                return;
            }

            auto feedback = std::make_shared<TrajectoryControllerAction::Feedback>();
            feedback->current_error = error_;
            feedback->waypoints_remaining = active_total_waypoints_ - next_waypoint_idx_;
            feedback->total_waypoints = active_total_waypoints_;
            goal_handle->publish_feedback(feedback);

            goal_cv_.wait_for(lock, 100ms);
        }
    }

    void controller_()
    {
        std::scoped_lock lock(feedback_mutex_, goal_mutex_);

        if (current_state_.name.empty()) {
            return;
        }

        arm_handle_->update(current_state_);

        if (trajectory_points_.empty() || next_waypoint_idx_ >= trajectory_points_.size()) {
            return;
        }

        const double elapsed = (this->get_clock()->now() - traj_start_).seconds();
        const size_t last_idx = trajectory_points_.size() - 1;

        size_t idx = next_waypoint_idx_;
        while (idx < last_idx &&
               rclcpp::Duration(trajectory_points_[idx + 1].time_from_start).seconds() <= elapsed) {
            ++idx;
        }

        const bool advanced = idx != next_waypoint_idx_;
        next_waypoint_idx_ = idx;
        current_target_ = trajectory_points_[next_waypoint_idx_];
        error_ = compute_position_error_(current_target_.positions);

        if (advanced) {
            goal_cv_.notify_all();
        }

        const bool finished = next_waypoint_idx_ >= last_idx &&
            rclcpp::Duration(trajectory_points_[last_idx].time_from_start).seconds() <= elapsed;

        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "Waypoint %zu/%u, error: %f",
                             next_waypoint_idx_ + 1,
                             active_total_waypoints_, error_);

        auto cmd = sensor_msgs::msg::JointState();
        cmd.header.stamp = this->get_clock()->now();
        cmd.name = trajectory_joint_names_;
        cmd.position = current_target_.positions;
        cmd.velocity = current_target_.velocities;
        cmd.effort = feedforward_torque_(current_target_);
        joint_states_pub_->publish(cmd);

        if (finished) {
            ++next_waypoint_idx_;
            goal_cv_.notify_all();
        }
    }

    /**
     * @brief Inverse-dynamics feedforward for one trajectory point, aligned to
     * trajectory_joint_names_.
     *
     * Uses the point's accelerations so the inertial and Coriolis terms are fed
     * forward too, not just gravity. inverse_dynamics returns every joint of the
     * reduced model in pinocchio order, which is not the goal's joint order, so
     * the torques are matched back by name. velocities/accelerations/effort are
     * padded because a JointTrajectoryPoint may legally leave them empty.
     */
    std::vector<double> feedforward_torque_(
        const trajectory_msgs::msg::JointTrajectoryPoint& point)
    {
        const size_t n = trajectory_joint_names_.size();

        JointState query;
        query.name = trajectory_joint_names_;
        query.position = point.positions;
        query.velocity = point.velocities;
        query.effort = point.effort;
        query.position.resize(n, 0.0);
        query.velocity.resize(n, 0.0);
        query.effort.resize(n, 0.0);

        std::vector<double> accel = point.accelerations;
        accel.resize(n, 0.0);

        const JointState ff = arm_handle_->inverse_dynamics(query, accel);

        std::unordered_map<std::string, double> ff_by_name;
        for (size_t j = 0; j < ff.name.size() && j < ff.effort.size(); ++j) {
            ff_by_name[ff.name[j]] = ff.effort[j];
        }

        std::vector<double> effort(n, 0.0);
        for (size_t j = 0; j < n; ++j) {
            auto it = ff_by_name.find(trajectory_joint_names_[j]);
            if (it != ff_by_name.end()) effort[j] = it->second;
        }
        return effort;
    }

    double compute_position_error_(const std::vector<double>& positions)
    {
        double sum_sq = 0.0;
        size_t counted = 0;
        const size_t n = std::min(trajectory_joint_names_.size(), positions.size());
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < current_state_.name.size() && j < current_state_.position.size(); ++j) {
                if (trajectory_joint_names_[i] == current_state_.name[j]) {
                    const double d = positions[i] - current_state_.position[j];
                    sum_sq += d * d;
                    ++counted;
                    break;
                }
            }
        }
        return counted > 0 ? std::sqrt(sum_sq / counted) : 0.0;
    }

    void handle_new_feedback_(sensor_msgs::msg::JointState::UniquePtr msg)
    {
        std::lock_guard<std::mutex> lock(feedback_mutex_);
        current_state_.name = msg->name;
        current_state_.position = msg->position;
        current_state_.velocity = msg->velocity;
        current_state_.effort = msg->effort;
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrajectoryControllerActionServer>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
