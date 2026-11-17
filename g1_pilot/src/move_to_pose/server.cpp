#include "server.h"
#include "../helper_funcs.h"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <functional>

namespace g1_pilot {
using namespace std::chrono_literals;

MoveToPoseServer::MoveToPoseServer() : Node("move_to_pose_server")
{
    // (1) Initialize configuration, models, and ROS interfaces.
    declare_parameters_();
    create_models_();
    create_publishers_();
    create_action_server_();

    // (2) Start feedback and control.
    feedback_sub_ = create_subscription<JointStateMsg>(
        get_parameter("feedback_topic").as_string(), 10,
        [this](JointStateMsg::UniquePtr msg) {
            std::lock_guard<std::mutex> lock(ctrl_mutex_);
            state_ = {msg->name, msg->position, msg->velocity, msg->effort};
            last_feedback_ = rclcpp::Time(msg->header.stamp, RCL_ROS_TIME);
        });
    control_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(control_period_)),
        std::bind(&MoveToPoseServer::control_tick_, this));
    estimate_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    estimate_timer_ = create_wall_timer(50ms, std::bind(&MoveToPoseServer::record_gripper_estimates_, this),
                                        estimate_group_);
}

MoveToPoseServer::~MoveToPoseServer()
{
    stopping_.store(true);
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void MoveToPoseServer::declare_parameters_()
{
    // (1) Robot assets and ROS endpoints.
    const auto description = ament_index_cpp::get_package_share_directory("g1_description");
    const auto pilot = ament_index_cpp::get_package_share_directory("g1_pilot");
    declare_parameter<std::string>("asset_file", description + "/assets/g1/g1_29dof_with_hand_rev_1_0.urdf");
    declare_parameter<std::string>("asset_root", description + "/assets/g1/");
    declare_parameter<std::string>("config_file", pilot + "/config/g1.yaml");
    declare_parameter<int>("num_dof", 29);
    declare_parameter<std::string>("feedback_topic", "feedback");
    declare_parameter<std::string>("position_control_topic", "position_control");
    declare_parameter<std::string>("traj_topic", "traj");
    declare_parameter<std::string>("action_name", "move_to_pose");
    declare_parameter<std::string>("planning_frame", "pelvis");

    // (2) Control timing and planning limits.
    declare_parameter<double>("control_period", 0.01);
    declare_parameter<double>("feedback_period", 0.1);
    declare_parameter<double>("plan_time_step", 0.01);
    declare_parameter<double>("max_joint_vel", 0.5);
    declare_parameter<double>("max_joint_acc", 1.0);
    declare_parameter<double>("tf_timeout", 0.5);
    declare_parameter<double>("replan_position_delta", 0.01);
    declare_parameter<double>("replan_orientation_delta", 0.05);
    declare_parameter<double>("replan_min_interval", 0.25);
    declare_parameter<double>("arm_safety_distance_x", 0.0);
    declare_parameter<double>("arm_safety_distance_y", 0.0);
    declare_parameter<double>("arm_safety_distance_z", 0.0);
    declare_parameter<bool>("final_leg_stepping", true);
    declare_parameter<std::string>("final_leg_anchor_frame", "left_ankle_roll_link");
    declare_parameter<int>("final_leg_steps", 10);
    declare_parameter<double>("final_leg_settle_time", 0.5);
    declare_parameter<bool>("use_gripper_estimate", true);
    declare_parameter<double>("gripper_estimate_max_age", 0.5);
    declare_parameter<double>("gripper_estimate_max_error", 0.03);

    // (3) Failure detection.
    declare_parameter<double>("goal_position_tolerance", 0.005);
    declare_parameter<double>("goal_orientation_tolerance", 0.1);
    declare_parameter<int>("max_plan_failures", 3);
    declare_parameter<double>("feedback_timeout", 0.5);
    declare_parameter<double>("max_tracking_error", 0.2);
    declare_parameter<double>("tracking_error_timeout", 1.0);
    declare_parameter<double>("near_target_distance", 0.02);
    declare_parameter<double>("gripper_estimate_hold_time", 60.0);
    frame_ = get_parameter("planning_frame").as_string();
    control_period_ = get_parameter("control_period").as_double();
}

void MoveToPoseServer::create_models_()
{
    HumanoidPilot::RobotConfig config;
    config.asset_file = get_parameter("asset_file").as_string();
    config.asset_root = get_parameter("asset_root").as_string();
    config.config_file = get_parameter("config_file").as_string();
    config.NUM_DOF = get_parameter("num_dof").as_int();
    planner_ = std::make_unique<Humanoid>(&config);
    controller_model_ = std::make_unique<Humanoid>(&config);
}

void MoveToPoseServer::create_publishers_()
{
    command_pub_ = create_publisher<JointStateMsg>(get_parameter("position_control_topic").as_string(), 10);
    path_pub_ = create_publisher<Path>(get_parameter("traj_topic").as_string(), 10);

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    ee_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    ee_broadcaster_->sendTransform(ee_static_transforms(*planner_, get_clock()->now()));
}

void MoveToPoseServer::create_action_server_()
{
    const auto action = get_parameter("action_name").as_string();
    cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    action_server_ = rclcpp_action::create_server<MoveToPoseAction>(
        this, action,
        std::bind(&MoveToPoseServer::on_goal_, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&MoveToPoseServer::on_cancel_, this, std::placeholders::_1),
        std::bind(&MoveToPoseServer::on_accepted_, this, std::placeholders::_1),
        rcl_action_server_get_default_options(), cb_group_);
    worker_ = std::thread(&MoveToPoseServer::worker_loop_, this);
    RCLCPP_INFO(get_logger(), "move_to_pose ready: action '%s'", action.c_str());
}

rclcpp_action::GoalResponse MoveToPoseServer::on_goal_(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const MoveToPoseGoal> goal)
{
    RCLCPP_INFO(get_logger(), "Goal: %s arm, target in '%s'",
                goal->left_arm ? "left" : "right",
                goal->target_pose.header.frame_id.c_str());
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse MoveToPoseServer::on_cancel_(const std::shared_ptr<GoalHandle> goal)
{
    // Commands stop at once; the worker then ends the goal as cancelled
    {
        std::scoped_lock lock(goal_mutex_, ctrl_mutex_);
        if (active_ == goal) clear_trajectory_locked_();
    }
    cv_.notify_all();
    return rclcpp_action::CancelResponse::ACCEPT;
}

void MoveToPoseServer::on_accepted_(const std::shared_ptr<GoalHandle> goal)
{
    {
        std::lock_guard<std::mutex> lock(goal_mutex_);
        active_ = goal;
        pending_ = goal;
    }
    set_target_topic_(goal->get_goal()->target_topic);
    cv_.notify_all();
}

void MoveToPoseServer::finish_(const std::shared_ptr<GoalHandle> & goal,
                               bool success, double error, const std::string & message)
{
    if (!goal->is_active()) return;
    auto result = std::make_shared<MoveToPoseResult>();
    result->success = success;
    result->final_error = error;
    result->message = message;

    if (goal->is_canceling()) goal->canceled(result);
    else if (success) goal->succeed(result);
    else goal->abort(result);

    bool was_active = false;
    {
        std::lock_guard<std::mutex> lock(goal_mutex_);
        if (active_ == goal) {
            active_ = nullptr;
            was_active = true;
        }
    }
    if (was_active) clear_target_topic_();
    cv_.notify_all();
}

void MoveToPoseServer::fail_(const std::shared_ptr<GoalHandle> & goal, const std::string & message)
{
    // A failure mid-motion brakes before the node stops commanding, as a finished goal does
    RCLCPP_ERROR(get_logger(), "Goal failed: %s", message.c_str());
    stepping_ = false;
    leg_step_ = 0;
    double error = 0.0;
    {
        std::scoped_lock lock(goal_mutex_, ctrl_mutex_);
        hold_between_steps_ = false;
        if (active_ == goal) {
            if (mode_ == ControlMode::Tracking) brake_locked_();
            if (mode_ == ControlMode::Braking) release_after_brake_ = true;
            else if (mode_ == ControlMode::Holding) clear_trajectory_locked_();
        }
        error = error_;
    }
    finish_(goal, false, error, message);
}

void MoveToPoseServer::set_target_topic_(const std::string & topic)
{
    std::lock_guard<std::mutex> lock(target_mutex_);
    if (topic == target_topic_ && (topic.empty() || target_sub_)) {
        return;
    }

    target_sub_.reset();
    target_topic_ = topic;
    have_live_target_ = false;
    live_target_count_ = 0;
    if (topic.empty()) return;

    rclcpp::SubscriptionOptions options;
    options.callback_group = cb_group_;
    target_sub_ = create_subscription<PoseStamped>(
        topic, rclcpp::SensorDataQoS(),
        std::bind(&MoveToPoseServer::on_live_target_, this, std::placeholders::_1),
        options);
    RCLCPP_INFO(get_logger(), "Tracking target updates on '%s'", topic.c_str());
}

void MoveToPoseServer::clear_target_topic_()
{
    std::lock_guard<std::mutex> lock(target_mutex_);
    if (!target_sub_) return;
    RCLCPP_INFO(get_logger(), "Stopped tracking '%s' after %lu update(s)",
                target_topic_.c_str(),
                static_cast<unsigned long>(live_target_count_));
    target_sub_.reset();
    target_topic_.clear();
    have_live_target_ = false;
}

void MoveToPoseServer::on_live_target_(const PoseStamped::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(target_mutex_);
    live_target_ = *msg;
    have_live_target_ = true;
    ++live_target_count_;
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Target update #%lu: [%.3f %.3f %.3f] in '%s'",
                         static_cast<unsigned long>(live_target_count_),
                         msg->pose.position.x, msg->pose.position.y,
                         msg->pose.position.z, msg->header.frame_id.c_str());
}

MoveToPoseServer::GoalStatus MoveToPoseServer::goal_status_(const std::shared_ptr<GoalHandle> & goal)
{
    std::lock_guard<std::mutex> lock(goal_mutex_);
    if (active_ != goal) return GoalStatus::Preempted;
    if (goal->is_canceling()) return GoalStatus::Cancelled;
    return GoalStatus::Active;
}

bool MoveToPoseServer::read_target_(PoseStamped & target)
{
    std::lock_guard<std::mutex> lock(target_mutex_);
    if (!target_sub_ || !have_live_target_) return false;
    target = live_target_;
    return true;
}

} // namespace g1_pilot
