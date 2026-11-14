#include <g1_pilot/g1_pilot.h>

#include "helper_funcs.h"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <humanoid_manipulation_interfaces/srv/check_reach.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

namespace g1_pilot {

using HumanoidPilot::Humanoid;
using HumanoidPilot::JointState;
using CheckReach = humanoid_manipulation_interfaces::srv::CheckReach;
using JointStateMsg = sensor_msgs::msg::JointState;

// Plans as move_to_pose does from the current joint state and answers its reach check, without commanding the arm
class IkServer : public rclcpp::Node {
public:
    IkServer() : Node("ik_server")
    {
        declare_parameters_();

        HumanoidPilot::RobotConfig config;
        config.asset_file = get_parameter("asset_file").as_string();
        config.asset_root = get_parameter("asset_root").as_string();
        config.config_file = get_parameter("config_file").as_string();
        config.NUM_DOF = get_parameter("num_dof").as_int();
        planner_ = std::make_unique<Humanoid>(&config);

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // The service plans for up to a second; feedback keeps arriving on its own group meanwhile
        feedback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        rclcpp::SubscriptionOptions options;
        options.callback_group = feedback_group_;
        feedback_sub_ = create_subscription<JointStateMsg>(
            get_parameter("feedback_topic").as_string(), 10,
            [this](JointStateMsg::UniquePtr msg) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                state_ = {msg->name, msg->position, msg->velocity, msg->effort};
                last_feedback_ = now();
            },
            options);

        service_ = create_service<CheckReach>(
            get_parameter("service_name").as_string(),
            [this](const std::shared_ptr<CheckReach::Request> request, std::shared_ptr<CheckReach::Response> response) {
                check_(*request, *response);
            });

        RCLCPP_INFO(get_logger(), "ik_server ready: service '%s'", service_->get_service_name());
    }

private:
    void declare_parameters_()
    {
        // Names and defaults as move_to_pose, so one params file can serve both
        const auto description = ament_index_cpp::get_package_share_directory("g1_description");
        const auto pilot = ament_index_cpp::get_package_share_directory("g1_pilot");
        declare_parameter<std::string>("asset_file", description + "/assets/g1/g1_29dof_with_hand_rev_1_0.urdf");
        declare_parameter<std::string>("asset_root", description + "/assets/g1/");
        declare_parameter<std::string>("config_file", pilot + "/config/g1.yaml");
        declare_parameter<int>("num_dof", 29);
        declare_parameter<std::string>("feedback_topic", "feedback");
        declare_parameter<std::string>("service_name", "check_reach");
        declare_parameter<double>("feedback_timeout", 0.5);
        declare_parameter<double>("plan_time_step", 0.01);
        declare_parameter<double>("max_joint_vel", 0.5);
        declare_parameter<double>("max_joint_acc", 1.0);
        declare_parameter<int>("final_leg_steps", 10);
        declare_parameter<double>("tf_timeout", 0.5);
        declare_parameter<double>("arm_safety_distance_x", 0.0);
        declare_parameter<double>("arm_safety_distance_y", 0.0);
        declare_parameter<double>("arm_safety_distance_z", 0.0);
        declare_parameter<double>("goal_position_tolerance", 0.005);
        declare_parameter<double>("goal_orientation_tolerance", 0.1);
    }

    void check_(const CheckReach::Request & request, CheckReach::Response & response)
    {
        // (1) The current joint state is the start, as for move_to_pose at rest
        response.reachable = false;
        response.position_error = std::numeric_limits<double>::infinity();
        response.orientation_error = std::numeric_limits<double>::infinity();
        JointState start;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (state_.name.empty()) {
                response.message = "no joint feedback received yet";
                return;
            }
            const double age = (now() - last_feedback_).seconds();
            if (age > get_parameter("feedback_timeout").as_double()) {
                char message[64];
                std::snprintf(message, sizeof(message), "joint feedback is stale (%.2f s old)", age);
                response.message = message;
                return;
            }
            start = state_;
        }

        // (2) The target into pelvis, with the safety offset move_to_pose adds
        Eigen::Isometry3d target;
        try {
            auto stamped = request.target_pose;
            stamped.header.stamp = rclcpp::Time(0);
            response.target_in_pelvis = tf_buffer_->transform(
                stamped, "pelvis", tf2::durationFromSec(get_parameter("tf_timeout").as_double()));
            tf2::fromMsg(response.target_in_pelvis.pose, target);
            target.translate(Eigen::Vector3d(
                get_parameter("arm_safety_distance_x").as_double(),
                get_parameter("arm_safety_distance_y").as_double(),
                get_parameter("arm_safety_distance_z").as_double()));
        } catch (const tf2::TransformException & error) {
            response.message = std::string("no transform from '") + request.target_pose.header.frame_id + "' to pelvis: " + error.what();
            return;
        }

        // (3) Plan as move_to_pose does; an empty plan carries the planner's reason
        std::vector<Eigen::VectorXd> configs;
        Eigen::Vector3d miss = Eigen::Vector3d::Zero();
        const bool left = request.left_arm;
        {
            std::lock_guard<std::mutex> lock(plan_mutex_);
            planner_->setLeftArm(left);
            const auto planned = planner_->planTrajectory(
                start, target.matrix(), static_cast<int>(get_parameter("final_leg_steps").as_int()), false,
                get_parameter("plan_time_step").as_double(), get_parameter("max_joint_vel").as_double(),
                get_parameter("max_joint_acc").as_double());
            configs = toJointConfigs(planned);
            if (configs.empty()) {
                const auto reason = planner_->plan_failure();
                response.message = "planning failed: " + (reason.empty() ? std::string("empty trajectory") : reason);
                return;
            }

            // (4) move_to_pose's reach check: the path pose nearest the target, against the goal tolerances
            for (const auto & q : configs) {
                const Eigen::MatrixXd pose = planner_->forward_kinematics(q, left);
                const double distance = (pose.block<3, 1>(0, 3) - target.translation()).norm();
                if (distance >= response.position_error) continue;
                response.position_error = distance;
                const Eigen::Matrix3d turn = pose.block<3, 3>(0, 0).transpose() * target.rotation();
                response.orientation_error = std::abs(Eigen::AngleAxisd(turn).angle());
                miss = pose.block<3, 1>(0, 3) - target.translation();
            }
        }
        response.position_error_pelvis = tf2::toMsg2(miss);
        response.position_error_target = tf2::toMsg2(Eigen::Vector3d(target.rotation().transpose() * miss));

        response.reachable = response.position_error <= get_parameter("goal_position_tolerance").as_double() &&
                             response.orientation_error <= get_parameter("goal_orientation_tolerance").as_double();
        char message[192];
        std::snprintf(message, sizeof(message),
                      "%s: the arm ends %.1f mm and %.1f deg from it (pelvis x %+.1f y %+.1f z %+.1f mm)",
                      response.reachable ? "in reach" : "target out of reach",
                      response.position_error * 1e3, response.orientation_error * 180.0 / M_PI,
                      miss.x() * 1e3, miss.y() * 1e3, miss.z() * 1e3);
        response.message = message;
    }

    std::unique_ptr<Humanoid> planner_;
    std::mutex plan_mutex_;
    std::mutex state_mutex_;
    JointState state_;
    rclcpp::Time last_feedback_{0, 0, RCL_ROS_TIME};

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::CallbackGroup::SharedPtr feedback_group_;
    rclcpp::Subscription<JointStateMsg>::SharedPtr feedback_sub_;
    rclcpp::Service<CheckReach>::SharedPtr service_;
};

} // namespace g1_pilot

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<g1_pilot::IkServer>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
