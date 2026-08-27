#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "sensor_msgs/msg/joint_state.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <g1_pilot/g1_pilot.h>
#include "helper_funcs.h"

using namespace std::chrono_literals;
using namespace HumanoidPilot;

class JointTrajPublisher : public rclcpp::Node
{
public:
    JointTrajPublisher() : Node("joint_traj_publisher")
    {
        std::string description_share = ament_index_cpp::get_package_share_directory("g1_description");
        std::string g1_pilot_share = ament_index_cpp::get_package_share_directory("g1_pilot");

        std::string default_asset_file = description_share + "/assets/g1/g1_29dof_with_hand_rev_1_0.urdf";
        std::string default_asset_root = description_share + "/assets/g1/";
        std::string default_config_file = g1_pilot_share + "/config/g1.yaml";

        this->declare_parameter<std::string>("asset_file", default_asset_file);
        this->declare_parameter<std::string>("asset_root", default_asset_root);
        this->declare_parameter<std::string>("config_file", default_config_file);
        this->declare_parameter<int>("num_dof", 29);
        this->declare_parameter<bool>("left_arm", true);
        this->declare_parameter<double>("waypoint_error_margin", 0.05);

        this->declare_parameter<std::string>("goal_pose_topic", "target_pose");
        this->declare_parameter<std::string>("position_control_topic", "position_control");
        this->declare_parameter<std::string>("feedback_topic", "feedback");
        this->declare_parameter<std::string>("traj_topic", "traj");

        RobotConfig config;
        config.asset_file = this->get_parameter("asset_file").as_string();
        config.asset_root = this->get_parameter("asset_root").as_string();
        config.config_file = this->get_parameter("config_file").as_string();
        config.NUM_DOF = this->get_parameter("num_dof").as_int();

        arm_ = std::make_unique<Humanoid>(&config);
        left_arm_ = this->get_parameter("left_arm").as_bool();

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        position_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
            this->get_parameter("position_control_topic").as_string(), 10);
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            this->get_parameter("traj_topic").as_string(), 10);

        goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            this->get_parameter("goal_pose_topic").as_string(), 5,
            std::bind(&JointTrajPublisher::handle_goal_pose_, this, std::placeholders::_1));

        feedback_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            this->get_parameter("feedback_topic").as_string(), 10,
            std::bind(&JointTrajPublisher::handle_feedback_, this, std::placeholders::_1));

        this->declare_parameter<double>("control_period", 0.01);
        const auto control_period = std::chrono::duration<double>(
            this->get_parameter("control_period").as_double());
        control_loop_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(control_period),
            std::bind(&JointTrajPublisher::controller_, this));

        RCLCPP_INFO(this->get_logger(),
            "Joint Traj Publisher ready (arm: %s)", left_arm_ ? "left" : "right");
    }

private:
    std::unique_ptr<Humanoid> arm_;
    bool left_arm_{true};

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr position_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr feedback_sub_;

    rclcpp::TimerBase::SharedPtr control_loop_;

    JointTrajectory trajectory_;
    size_t next_waypoint_idx_ = 0;
    size_t total_waypoints_ = 0;
    double error_ = 0.0;
    rclcpp::Time traj_start_;

    JointState current_state_;
    std::unordered_map<std::string, size_t> joint_name_to_index_;

    std::mutex feedback_mutex_;
    std::mutex traj_mutex_;

    Eigen::MatrixXd transform_to_pelvis_(const geometry_msgs::msg::PoseStamped &pose_msg)
    {
        try {
            auto stamped = pose_msg;
            stamped.header.stamp = rclcpp::Time(0);
            auto transformed = tf_buffer_->transform(stamped, "pelvis", tf2::durationFromSec(0.5));
            Eigen::Isometry3d pose_in_pelvis;
            tf2::fromMsg(transformed.pose, pose_in_pelvis);
            return pose_in_pelvis.matrix();
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
            return Eigen::MatrixXd();
        }
    }

    void handle_goal_pose_(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(),
            "Received goal pose in frame: %s", msg->header.frame_id.c_str());

        JointState start_state;
        {
            std::lock_guard<std::mutex> lock(feedback_mutex_);
            if (current_state_.name.empty()) {
                RCLCPP_ERROR(this->get_logger(),
                    "No joint feedback received yet — cannot plan");
                return;
            }
            start_state = current_state_;
        }

        Eigen::MatrixXd goal_matrix = transform_to_pelvis_(*msg);
        if (goal_matrix.size() == 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to transform goal pose to pelvis frame");
            return;
        }

        arm_->setLeftArm(left_arm_);
        auto planned_traj = arm_->planTrajectory(start_state, goal_matrix);
        if (planned_traj.position.empty() || planned_traj.position[0].empty()) {
            RCLCPP_ERROR(this->get_logger(), "Trajectory planning failed: empty trajectory");
            return;
        }

        const size_t num_pts = planned_traj.position[0].size();
        const size_t num_joints = planned_traj.name.size();

        // Perform transpose: convert SOA columns into Cartesian frames for visualization
        std::vector<Eigen::MatrixXd> cart_traj;
        cart_traj.reserve(num_pts);
        Eigen::VectorXd q(num_joints);

        for (size_t step = 0; step < num_pts; ++step) {
            for (size_t j = 0; j < num_joints; ++j) {
                q[j] = planned_traj.position[j][step];
            }
            cart_traj.push_back(arm_->forward_kinematics(q, left_arm_));
        }

        nav_msgs::msg::Path path_msg = convertToPath(cart_traj, "pelvis", this->get_clock()->now());

        {
            std::lock_guard<std::mutex> lock(traj_mutex_);
            trajectory_ = std::move(planned_traj);
            next_waypoint_idx_ = 0;
            total_waypoints_ = num_pts;
            traj_start_ = this->get_clock()->now();
        }

        path_pub_->publish(path_msg);
        RCLCPP_INFO(this->get_logger(), "Planned trajectory with %zu waypoints", num_pts);
    }

    void handle_feedback_(sensor_msgs::msg::JointState::UniquePtr msg)
    {
        std::lock_guard<std::mutex> lock(feedback_mutex_);
        current_state_.name = msg->name;
        current_state_.position = msg->position;
        current_state_.velocity = msg->velocity;
        current_state_.effort = msg->effort;

        if (joint_name_to_index_.size() != msg->name.size()) {
            joint_name_to_index_.clear();
            for (size_t i = 0; i < msg->name.size(); ++i) {
                joint_name_to_index_[msg->name[i]] = i;
            }
        }
    }

    void controller_()
    {
        std::scoped_lock lock(feedback_mutex_, traj_mutex_);

        if (current_state_.name.empty() || total_waypoints_ == 0 || next_waypoint_idx_ >= total_waypoints_) {
            return;
        }

        arm_->update(current_state_);

        const double step = trajectory_.time_step > 0.0 ? trajectory_.time_step : 0.01;
        const double elapsed = (this->get_clock()->now() - traj_start_).seconds();
        const size_t last_idx = total_waypoints_ - 1;

        size_t idx = static_cast<size_t>(std::max(0.0, std::floor(elapsed / step)));
        const bool finished = idx >= last_idx;
        next_waypoint_idx_ = std::min(idx, last_idx);

        error_ = compute_position_error_(next_waypoint_idx_);

        RCLCPP_INFO(this->get_logger(), "Waypoint %zu/%zu, error: %f",
            next_waypoint_idx_ + 1, total_waypoints_, error_);

        sensor_msgs::msg::JointState cmd;
        cmd.header.stamp = this->get_clock()->now();
        cmd.name = trajectory_.name;

        const size_t num_joints = trajectory_.name.size();
        cmd.position.resize(num_joints);
        cmd.velocity.resize(num_joints);
        cmd.effort.assign(num_joints, 0.0);

        std::vector<double> accel(num_joints, 0.0);
        for (size_t j = 0; j < num_joints; ++j) {
            cmd.position[j] = trajectory_.position[j][next_waypoint_idx_];
            if (j < trajectory_.velocity.size() && next_waypoint_idx_ < trajectory_.velocity[j].size()) {
                cmd.velocity[j] = trajectory_.velocity[j][next_waypoint_idx_];
            }
            if (j < trajectory_.acceleration.size() && next_waypoint_idx_ < trajectory_.acceleration[j].size()) {
                accel[j] = trajectory_.acceleration[j][next_waypoint_idx_];
            }
        }

        cmd.effort = feedforward_torque_(cmd.name, cmd.position, cmd.velocity, accel);

        position_pub_->publish(cmd);

        if (finished) {
            RCLCPP_INFO(this->get_logger(), "Trajectory complete");
            next_waypoint_idx_ = total_waypoints_;
        }
    }

    /**
     * @brief Inverse-dynamics feedforward for one trajectory sample, reordered
     * to match the commanded joint list.
     *
     * inverse_dynamics returns every joint of the reduced model in pinocchio
     * order, so the torques are matched back by name.
     */
    std::vector<double> feedforward_torque_(
        const std::vector<std::string>& names,
        const std::vector<double>& position,
        const std::vector<double>& velocity,
        const std::vector<double>& acceleration)
    {
        JointState query;
        query.name = names;
        query.position = position;
        query.velocity = velocity;
        query.effort.assign(names.size(), 0.0);

        const JointState ff = arm_->inverse_dynamics(query, acceleration);

        std::unordered_map<std::string, double> ff_by_name;
        for (size_t j = 0; j < ff.name.size() && j < ff.effort.size(); ++j) {
            ff_by_name[ff.name[j]] = ff.effort[j];
        }

        std::vector<double> effort(names.size(), 0.0);
        for (size_t j = 0; j < names.size(); ++j) {
            auto it = ff_by_name.find(names[j]);
            if (it != ff_by_name.end()) effort[j] = it->second;
        }
        return effort;
    }

    double compute_position_error_(size_t waypoint_idx)
    {
        double sum_sq = 0.0;
        size_t counted = 0;
        const size_t num_joints = trajectory_.name.size();

        for (size_t j = 0; j < num_joints; ++j) {
            auto it = joint_name_to_index_.find(trajectory_.name[j]);
            if (it != joint_name_to_index_.end()) {
                const size_t feedback_idx = it->second;
                if (feedback_idx < current_state_.position.size() && waypoint_idx < trajectory_.position[j].size()) {
                    const double d = trajectory_.position[j][waypoint_idx] - current_state_.position[feedback_idx];
                    sum_sq += d * d;
                    ++counted;
                }
            }
        }
        return counted > 0 ? std::sqrt(sum_sq / counted) : 0.0;
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<JointTrajPublisher>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}