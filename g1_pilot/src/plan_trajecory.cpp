#include <g1_pilot/g1_pilot.h>

#include "rclcpp/rclcpp.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "helper_funcs.h"

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <mutex>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "humanoid_manipulation_interfaces/srv/plan_joint_trajectory.hpp"

using namespace HumanoidPilot;

using PlanTrajectoryService = humanoid_manipulation_interfaces::srv::PlanJointTrajectory;

class PlanTrajectoryServer : public rclcpp::Node
{
public:
    PlanTrajectoryServer() : Node("plan_trajectory_server")
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

        /* TF2 */
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        /* Publishers */
        this->declare_parameter<std::string>("traj_topic", "traj");

        path_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            this->get_parameter("traj_topic").as_string(), 10);

        this->declare_parameter<std::string>("feedback_topic", "feedback");
        feedback_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            this->get_parameter("feedback_topic").as_string(), 10,
            [this](sensor_msgs::msg::JointState::UniquePtr msg){
                std::lock_guard<std::mutex> lock(feedback_mutex_);
                current_state_.name = msg->name;
                current_state_.position = msg->position;
                current_state_.velocity = msg->velocity;
                current_state_.effort = msg->effort;
            });

        /* EE pose subscribers — published for visualisation//diagnostics */
        this->declare_parameter<std::string>("left_ee_pose_topic", "/left_ee_pose");
        this->declare_parameter<std::string>("right_ee_pose_topic", "/right_ee_pose");

        left_ee_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            this->get_parameter("left_ee_pose_topic").as_string(), 5,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(ee_pose_mutex_);
                left_ee_pose_ = *msg;
                have_left_ee_pose_ = true;
            });

        right_ee_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            this->get_parameter("right_ee_pose_topic").as_string(), 5,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(ee_pose_mutex_);
                right_ee_pose_ = *msg;
                have_right_ee_pose_ = true;
            });

        /* Service server */
        service_ = this->create_service<PlanTrajectoryService>(
            "plan_trajectory",
            std::bind(&PlanTrajectoryServer::handle_service_, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Plan trajectory service server ready");
    }

private:
    // Publishers
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

    // TF2
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Service server
    rclcpp::Service<PlanTrajectoryService>::SharedPtr service_;

    // EE pose subscribers + latest-pose cache
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr left_ee_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr right_ee_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr feedback_sub_;
    std::mutex feedback_mutex_;
    JointState current_state_;

    std::mutex ee_pose_mutex_;
    geometry_msgs::msg::PoseStamped left_ee_pose_;
    geometry_msgs::msg::PoseStamped right_ee_pose_;
    bool have_left_ee_pose_{false};
    bool have_right_ee_pose_{false};

    // Arm Handle
    std::unique_ptr<Humanoid> arm_handle_;

    Eigen::MatrixXd get_pose_in_pelvis_(const geometry_msgs::msg::PoseStamped &pose_msg)
    {
        try {
            auto latest_pose = pose_msg;
            latest_pose.header.stamp = rclcpp::Time(0);
            auto transformed = tf_buffer_->transform(latest_pose, "pelvis", tf2::durationFromSec(0.5));
            Eigen::Isometry3d pose_in_pelvis;
            tf2::fromMsg(transformed.pose, pose_in_pelvis);
            return pose_in_pelvis.matrix();
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
            return Eigen::MatrixXd();
        }
    }

    void handle_service_(
        const std::shared_ptr<PlanTrajectoryService::Request> request,
        std::shared_ptr<PlanTrajectoryService::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Received request for goal in frame: %s",
                     request->target_pose.header.frame_id.c_str());

        JointState start_state;
        {
            std::lock_guard<std::mutex> lock(feedback_mutex_);
            if (current_state_.name.empty()) {
                RCLCPP_ERROR(this->get_logger(),
                    "No joint feedback received yet — cannot plan");
                response->success = false;
                return;
            }
            start_state = current_state_;
        }

        Eigen::MatrixXd goal_matrix = get_pose_in_pelvis_(request->target_pose);
        if (goal_matrix.size() == 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to transform goal pose to pelvis frame");
            response->success = false;
            return;
        }

        arm_handle_->setLeftArm(request->left_arm);

        auto joint_trajectory = arm_handle_->planTrajectory(start_state, goal_matrix);

        std::vector<Eigen::VectorXd> joint_traj = toJointConfigs(joint_trajectory);

        if (joint_traj.empty()) {
            RCLCPP_ERROR(this->get_logger(),
                "Trajectory planning failed: empty trajectory (likely IK error exceeded threshold)");
            response->success = false;
            return;
        }

        std::vector<Eigen::MatrixXd> trajectory;
        trajectory.reserve(joint_traj.size());
        for (const auto& q : joint_traj) {
            trajectory.push_back(arm_handle_->forward_kinematics(q, request->left_arm));
        }

        response->success = true;
        response->trajectory = convertToTrajectory(joint_trajectory, "pelvis", this->get_clock()->now());

        path_pub_->publish(convertToPath(trajectory, "pelvis", this->get_clock()->now()));

        RCLCPP_INFO(this->get_logger(), "Planned trajectory with %zu waypoints",
                     trajectory.size());
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlanTrajectoryServer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
