/**
 * @file move_to_pose.cpp
 * @brief Plan to a target pose and track the trajectory, in one node.
 *
 * Serves humanoid_arm_skills:
 *   action   move_to_pose          target pose + arm -> plan, track, report
 *   service  move_to_pose/cancel   stop now: clear trajectory, stop commanding
 *
 * A goal may also name a target_topic. While that goal runs, PoseStamped
 * messages on it are the moving target: when the target drifts past
 * replan_position_delta / replan_orientation_delta the worker calls
 * Humanoid::replanTrajectory and swaps the result in. That call reuses the
 * cached RRT* search when the new goal is close to the last one and the arm
 * has not drifted, and falls back to a full plan when it is not, so tracking a
 * moving target costs far less than replanning from scratch.
 *
 * A failed replan is not a failed goal: the trajectory already running stays,
 * and the next update tries again.
 *
 * Planning and control both use the g1_pilot library directly. Two rules matter:
 *
 *   - A new goal preempts the old one but does NOT clear the trajectory. The
 *     control loop keeps driving the old one for the whole planning latency;
 *     the swap happens once the new trajectory is ready. Clearing on arrival
 *     would leave the arm uncommanded while the planner works.
 *   - Cancelling DOES clear the trajectory, so the control loop stops
 *     publishing on its next tick.
 *
 * Threading: the control loop runs on a timer, planning runs on one worker
 * thread. Two mutexes, and the split matters: ctrl_mutex_ is held by the
 * control loop on every tick, so goal bookkeeping must NOT live under it, or
 * the action server and cancel service starve behind a 100 Hz loop and stop
 * answering. Only the install site takes both, in the order goal -> ctrl.
 * The worker holds neither while planning, so the control loop keeps its
 * period throughout. Planning and control use separate Humanoid instances
 * because both mutate model state and neither can wait for the other.
 */

#include <g1_pilot/g1_pilot.h>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "helper_funcs.h"

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

#include "humanoid_manipulation_interfaces/action/move_to_pose.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

using namespace HumanoidPilot;
using namespace std::chrono_literals;

using MoveToPoseAction = humanoid_manipulation_interfaces::action::MoveToPose;
using GoalHandle = rclcpp_action::ServerGoalHandle<MoveToPoseAction>;
using Trigger = std_srvs::srv::Trigger;

class MoveToPoseServer : public rclcpp::Node
{
public:
    MoveToPoseServer() : Node("move_to_pose_server")
    {
        const std::string description_share =
            ament_index_cpp::get_package_share_directory("g1_description");
        const std::string pilot_share =
            ament_index_cpp::get_package_share_directory("g1_pilot");

        declare_parameter<std::string>(
            "asset_file", description_share + "/assets/g1/g1_29dof_with_hand_rev_1_0.urdf");
        declare_parameter<std::string>("asset_root", description_share + "/assets/g1/");
        declare_parameter<std::string>("config_file", pilot_share + "/config/g1.yaml");
        declare_parameter<int>("num_dof", 29);
        declare_parameter<std::string>("feedback_topic", "feedback");
        declare_parameter<std::string>("position_control_topic", "position_control");
        declare_parameter<std::string>("left_ee_pose_topic", "left_ee_pose");
        declare_parameter<std::string>("right_ee_pose_topic", "right_ee_pose");
        declare_parameter<std::string>("traj_topic", "traj");
        declare_parameter<std::string>("action_name", "move_to_pose");
        declare_parameter<std::string>("planning_frame", "pelvis");
        declare_parameter<double>("control_period", 0.01);
        declare_parameter<double>("feedback_period", 0.1);
        declare_parameter<double>("plan_time_step", 0.01);
        declare_parameter<double>("max_joint_vel", 1.0);
        declare_parameter<double>("max_joint_acc", 2.0);
        declare_parameter<double>("tf_timeout", 0.5);
        declare_parameter<double>("replan_position_delta", 0.005);
        declare_parameter<double>("replan_orientation_delta", 0.05);
        declare_parameter<double>("replan_min_interval", 0.25);

        frame_ = get_parameter("planning_frame").as_string();
        control_period_ = get_parameter("control_period").as_double();

        RobotConfig config;
        config.asset_file = get_parameter("asset_file").as_string();
        config.asset_root = get_parameter("asset_root").as_string();
        config.config_file = get_parameter("config_file").as_string();
        config.NUM_DOF = get_parameter("num_dof").as_int();
        planner_ = std::make_unique<Humanoid>(&config);
        controller_model_ = std::make_unique<Humanoid>(&config);

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        command_pub_ = create_publisher<sensor_msgs::msg::JointState>(
            get_parameter("position_control_topic").as_string(), 10);
        left_ee_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            get_parameter("left_ee_pose_topic").as_string(), 10);
        right_ee_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            get_parameter("right_ee_pose_topic").as_string(), 10);
        path_pub_ = create_publisher<nav_msgs::msg::Path>(
            get_parameter("traj_topic").as_string(), 10);

        feedback_sub_ = create_subscription<sensor_msgs::msg::JointState>(
            get_parameter("feedback_topic").as_string(), 10,
            [this](sensor_msgs::msg::JointState::UniquePtr msg) {
                std::lock_guard<std::mutex> lock(ctrl_mutex_);
                state_.name = msg->name;
                state_.position = msg->position;
                state_.velocity = msg->velocity;
                state_.effort = msg->effort;
            });

        const std::string action_name = get_parameter("action_name").as_string();
        /* Held as a member on purpose: the node keeps only a weak_ptr to a
         * callback group, so a local would be destroyed at the end of this
         * constructor and the executor would silently never service the action
         * server or the cancel service. */
        cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

        action_server_ = rclcpp_action::create_server<MoveToPoseAction>(
            this, action_name,
            std::bind(&MoveToPoseServer::on_goal_, this, std::placeholders::_1, std::placeholders::_2),
            std::bind(&MoveToPoseServer::on_cancel_, this, std::placeholders::_1),
            std::bind(&MoveToPoseServer::on_accepted_, this, std::placeholders::_1),
            rcl_action_server_get_default_options(), cb_group_);

        cancel_service_ = create_service<Trigger>(
            action_name + "/cancel",
            std::bind(&MoveToPoseServer::on_cancel_service_, this,
                      std::placeholders::_1, std::placeholders::_2),
            rclcpp::ServicesQoS(), cb_group_);

        control_timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(control_period_)),
            std::bind(&MoveToPoseServer::control_tick_, this));

        worker_ = std::thread(&MoveToPoseServer::worker_loop_, this);

        RCLCPP_INFO(get_logger(), "move_to_pose ready: action '%s', cancel '%s/cancel'",
                    action_name.c_str(), action_name.c_str());
    }

    ~MoveToPoseServer() override
    {
        stopping_.store(true);
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

private:
    // ── Model ──
    std::unique_ptr<Humanoid> planner_;           // worker thread only
    std::unique_ptr<Humanoid> controller_model_;  // control loop only
    std::string frame_;
    double control_period_ = 0.01;

    // ── ROS ──
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr command_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr left_ee_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr right_ee_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr feedback_sub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::CallbackGroup::SharedPtr cb_group_;
    rclcpp_action::Server<MoveToPoseAction>::SharedPtr action_server_;
    rclcpp::Service<Trigger>::SharedPtr cancel_service_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    /* ── Control state: guarded by ctrl_mutex_, taken every control tick ── */
    std::mutex ctrl_mutex_;

    JointState state_;                  // latest joint feedback

    std::vector<std::string> traj_names_;
    std::vector<trajectory_msgs::msg::JointTrajectoryPoint> traj_points_;
    size_t idx_ = 0;
    rclcpp::Time traj_start_;
    double error_ = 0.0;
    bool tracking_ = false;
    Eigen::MatrixXd planned_for_;       // goal pose the live trajectory serves
    bool have_planned_for_ = false;

    /* ── Live target: guarded by target_mutex_. Kept off both other mutexes
     * so an update at sensor rate never queues behind the control loop or the
     * goal bookkeeping. ── */
    std::mutex target_mutex_;
    std::string target_topic_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_sub_;
    geometry_msgs::msg::PoseStamped live_target_;
    bool have_live_target_ = false;
    uint64_t live_target_count_ = 0;

    /* ── Goal state: guarded by goal_mutex_, deliberately separate from the
     * control state so action and service callbacks never queue behind the
     * control loop. ── */
    std::mutex goal_mutex_;
    std::condition_variable cv_;
    std::shared_ptr<GoalHandle> active_;
    std::shared_ptr<GoalHandle> pending_;
    bool cancel_requested_ = false;

    // ── Worker ──
    std::thread worker_;
    std::atomic<bool> stopping_{false};
    std::chrono::steady_clock::time_point last_tick_{};
    std::chrono::steady_clock::time_point last_replan_{};

    // ══════════════════ Action ══════════════════

    rclcpp_action::GoalResponse on_goal_(
        const rclcpp_action::GoalUUID &,
        std::shared_ptr<const MoveToPoseAction::Goal> goal)
    {
        RCLCPP_INFO(get_logger(), "Goal: %s arm, target in '%s'",
                    goal->left_arm ? "left" : "right",
                    goal->target_pose.header.frame_id.c_str());
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse on_cancel_(const std::shared_ptr<GoalHandle>)
    {
        cv_.notify_all();
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void on_accepted_(const std::shared_ptr<GoalHandle> goal)
    {
        /* Handed to the worker instead of a thread per goal: planning blocks
         * for seconds and cannot be interrupted, so joining a per-goal thread
         * here would stall the executor. Publishing the new handle is what
         * preempts the running goal, which retires itself. */
        {
            std::lock_guard<std::mutex> lock(goal_mutex_);
            active_ = goal;
            pending_ = goal;
            cancel_requested_ = false;
        }
        // Subscribed before planning starts, so updates are already arriving by
        // the time the arm is moving. Outside the lock: target_mutex_ is never
        // taken while holding goal_mutex_.
        set_target_topic_(goal->get_goal()->target_topic);
        cv_.notify_all();
    }

    // ══════════════════ Live target ══════════════════

    /** (Re)subscribe to the topic carrying updated target poses. Empty drops it. */
    void set_target_topic_(const std::string & topic)
    {
        std::lock_guard<std::mutex> lock(target_mutex_);
        if (topic == target_topic_ && (topic.empty() || target_sub_)) {
            return;     // a replan on the same topic keeps the same subscription
        }

        target_sub_.reset();
        target_topic_ = topic;
        have_live_target_ = false;
        live_target_count_ = 0;
        if (topic.empty()) return;

        /* Best-effort: a target stream is sensor data, and a best-effort reader
         * accepts both reliable and best-effort publishers, where a reliable
         * one would refuse a best-effort publisher outright. */
        rclcpp::SubscriptionOptions options;
        options.callback_group = cb_group_;
        target_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            topic, rclcpp::SensorDataQoS(),
            std::bind(&MoveToPoseServer::on_live_target_, this, std::placeholders::_1),
            options);
        RCLCPP_INFO(get_logger(), "Tracking target updates on '%s'", topic.c_str());
    }

    void clear_target_topic_()
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

    void on_live_target_(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
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

    /** Latest pose seen on the target topic. False if none has arrived.
     *  Nothing consumes this yet; it is what the moving-goal planner API will
     *  read once g1_pilot accepts one. */
    bool live_target(geometry_msgs::msg::PoseStamped & out)
    {
        std::lock_guard<std::mutex> lock(target_mutex_);
        if (!have_live_target_) return false;
        out = live_target_;
        return true;
    }

    // ══════════════════ Cancel service ══════════════════

    void on_cancel_service_(const std::shared_ptr<Trigger::Request>,
                            std::shared_ptr<Trigger::Response> response)
    {
        std::shared_ptr<GoalHandle> goal;
        {
            std::lock_guard<std::mutex> lock(goal_mutex_);
            cancel_requested_ = true;
            goal = active_;
        }
        // Stop commanding straight away rather than waiting for the worker:
        // an empty trajectory means the control loop publishes nothing.
        clear_trajectory_();
        cv_.notify_all();

        if (!goal) {
            response->success = false;
            response->message = "No active goal; trajectory cleared anyway";
            RCLCPP_WARN(get_logger(), "Cancel: %s", response->message.c_str());
            return;
        }

        // Wait for the goal to terminate so the reply is not ahead of reality.
        std::unique_lock<std::mutex> lock(goal_mutex_);
        const bool finished = cv_.wait_for(lock, 5s, [this, &goal] {
            return active_ != goal || stopping_.load();
        });
        lock.unlock();

        response->success = finished;
        response->message = finished
            ? "Cancelled: trajectory reset, controller stopped commanding"
            : "Trajectory reset, but the goal did not terminate within 5 s";
        RCLCPP_INFO(get_logger(), "Cancel: %s", response->message.c_str());
    }

    void clear_trajectory_()
    {
        std::lock_guard<std::mutex> lock(ctrl_mutex_);
        traj_points_.clear();
        idx_ = 0;
        tracking_ = false;
    }

    // ══════════════════ Worker: plan, install, monitor ══════════════════

    void worker_loop_()
    {
        while (rclcpp::ok() && !stopping_.load()) {
            std::shared_ptr<GoalHandle> goal;
            {
                std::unique_lock<std::mutex> lock(goal_mutex_);
                cv_.wait(lock, [this] { return pending_ || stopping_.load(); });
                if (stopping_.load()) return;
                goal = pending_;
                pending_ = nullptr;
            }
            run_(goal);
        }
    }

    enum class Install { Ok, Preempted, Cancelled };

    /** Swap a freshly planned trajectory in, unless this goal has lost the arm.
     *  Everything up to this point ran against the previous trajectory. */
    Install install_(const std::shared_ptr<GoalHandle> & goal,
                     const trajectory_msgs::msg::JointTrajectory & traj,
                     const Eigen::MatrixXd & planned_for)
    {
        // The only site that needs both. Order: goal_mutex_ then ctrl_mutex_.
        std::scoped_lock lock(goal_mutex_, ctrl_mutex_);
        if (active_ != goal) return Install::Preempted;
        if (cancel_requested_ || goal->is_canceling()) {
            traj_points_.clear();
            idx_ = 0;
            tracking_ = false;
            return Install::Cancelled;
        }
        traj_names_ = traj.joint_names;
        traj_points_ = traj.points;
        idx_ = 0;
        traj_start_ = get_clock()->now();
        tracking_ = true;
        planned_for_ = planned_for;
        have_planned_for_ = true;
        return Install::Ok;
    }

    /// True if b is far enough from a to be worth replanning for.
    static bool pose_moved_(const Eigen::MatrixXd & a, const Eigen::MatrixXd & b,
                            double pos_delta, double rot_delta)
    {
        const double dp = (b.block<3, 1>(0, 3) - a.block<3, 1>(0, 3)).norm();
        if (dp >= pos_delta) return true;
        const Eigen::Matrix3d dR =
            a.block<3, 3>(0, 0).transpose() * b.block<3, 3>(0, 0);
        return std::abs(Eigen::AngleAxisd(dR).angle()) >= rot_delta;
    }

    void run_(const std::shared_ptr<GoalHandle> & goal)
    {
        const auto request = goal->get_goal();
        last_replan_ = {};      // a new goal may replan immediately

        JointState start;
        {
            std::lock_guard<std::mutex> lock(ctrl_mutex_);
            start = state_;
        }
        if (start.name.empty()) {
            // finish_ takes goal_mutex_, so never call it while holding a lock.
            finish_(goal, false, 0.0, "No joint feedback received yet");
            return;
        }

        const Eigen::MatrixXd target = to_planning_frame_(request->target_pose);
        if (target.size() == 0) {
            finish_(goal, false, 0.0, "Target pose transform failed");
            return;
        }

        // ── Plan. Long, uninterruptible, and holds no lock: the control loop
        // keeps driving the previous trajectory throughout. ──
        RCLCPP_INFO(get_logger(), "Planning for %s arm ...",
                    request->left_arm ? "left" : "right");

        trajectory_msgs::msg::JointTrajectory traj;
        std::vector<Eigen::MatrixXd> ee_path;
        try {
            planner_->setLeftArm(request->left_arm);
            const JointTrajectory planned = planner_->planTrajectory(
                start, target,
                get_parameter("plan_time_step").as_double(),
                get_parameter("max_joint_vel").as_double(),
                get_parameter("max_joint_acc").as_double());

            const std::vector<Eigen::VectorXd> configs = toJointConfigs(planned);
            if (configs.empty()) {
                finish_(goal, false, 0.0, "Planning failed: empty trajectory");
                return;
            }
            ee_path.reserve(configs.size());
            for (const auto & q : configs) {
                ee_path.push_back(planner_->forward_kinematics(q, request->left_arm));
            }
            traj = convertToTrajectory(planned, frame_, get_clock()->now());
        } catch (const std::exception & e) {
            RCLCPP_ERROR(get_logger(), "Planning threw: %s", e.what());
            finish_(goal, false, 0.0, std::string("Planning failed: ") + e.what());
            return;
        }

        if (traj.points.empty()) {
            finish_(goal, false, 0.0, "Planning produced no waypoints");
            return;
        }
        RCLCPP_INFO(get_logger(), "Planned %zu waypoints", traj.points.size());

        // ── Install, unless this goal lost the arm while planning ──
        switch (install_(goal, traj, target)) {
            case Install::Preempted:
                RCLCPP_INFO(get_logger(), "Preempted while planning");
                finish_(goal, false, 0.0, "Preempted by a newer goal");
                return;
            case Install::Cancelled:
                RCLCPP_INFO(get_logger(), "Cancelled while planning");
                finish_(goal, false, 0.0, "Cancelled while planning");
                return;
            case Install::Ok:
                break;
        }

        path_pub_->publish(convertToPath(ee_path, frame_, get_clock()->now()));
        RCLCPP_INFO(get_logger(), "Tracking trajectory");
        monitor_(goal);
    }

    /** Replan toward the latest pose on the target topic, if it has moved far
     *  enough since the trajectory now running was planned.
     *
     *  Deliberately forgiving: any reason not to replan leaves the current
     *  trajectory alone and returns. A target that stops publishing, a TF that
     *  drops out, or a planner that fails on one update must not disturb a
     *  motion that is already valid. */
    void try_replan_(const std::shared_ptr<GoalHandle> & goal)
    {
        geometry_msgs::msg::PoseStamped update;
        {
            std::lock_guard<std::mutex> lock(target_mutex_);
            if (!target_sub_ || !have_live_target_) return;
            update = live_target_;
        }

        const auto now = std::chrono::steady_clock::now();
        if (last_replan_.time_since_epoch().count() != 0) {
            const double since = std::chrono::duration<double>(now - last_replan_).count();
            if (since < get_parameter("replan_min_interval").as_double()) return;
        }

        const Eigen::MatrixXd target = to_planning_frame_(update);
        if (target.size() == 0) return;     // to_planning_frame_ already logged

        {
            std::lock_guard<std::mutex> lock(ctrl_mutex_);
            if (!have_planned_for_ ||
                !pose_moved_(planned_for_, target,
                             get_parameter("replan_position_delta").as_double(),
                             get_parameter("replan_orientation_delta").as_double())) {
                return;
            }
        }

        JointState start;
        {
            std::lock_guard<std::mutex> lock(ctrl_mutex_);
            start = state_;
        }
        if (start.name.empty()) return;

        const bool left = goal->get_goal()->left_arm;
        trajectory_msgs::msg::JointTrajectory traj;
        std::vector<Eigen::MatrixXd> ee_path;
        try {
            planner_->setLeftArm(left);
            const JointTrajectory planned = planner_->replanTrajectory(
                start, target,
                get_parameter("plan_time_step").as_double(),
                get_parameter("max_joint_vel").as_double(),
                get_parameter("max_joint_acc").as_double());

            const std::vector<Eigen::VectorXd> configs = toJointConfigs(planned);
            if (configs.empty()) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                     "Replan produced no trajectory; keeping the current one");
                last_replan_ = now;     // back off rather than retry every pass
                return;
            }
            ee_path.reserve(configs.size());
            for (const auto & q : configs) {
                ee_path.push_back(planner_->forward_kinematics(q, left));
            }
            traj = convertToTrajectory(planned, frame_, get_clock()->now());
        } catch (const std::exception & e) {
            RCLCPP_WARN(get_logger(), "Replan threw: %s; keeping the current trajectory",
                        e.what());
            last_replan_ = now;
            return;
        }

        last_replan_ = now;
        if (traj.points.empty()) return;

        // Preempted or cancelled mid-replan: drop the new trajectory and let
        // the monitor's next pass terminate the goal.
        if (install_(goal, traj, target) != Install::Ok) return;

        path_pub_->publish(convertToPath(ee_path, frame_, get_clock()->now()));
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                             "Replanned to [%.3f %.3f %.3f], %zu waypoints",
                             target(0, 3), target(1, 3), target(2, 3),
                             traj.points.size());
    }

    void monitor_(const std::shared_ptr<GoalHandle> & goal)
    {
        const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(get_parameter("feedback_period").as_double()));
        auto feedback = std::make_shared<MoveToPoseAction::Feedback>();

        while (rclcpp::ok() && !stopping_.load()) {
            // ── Has this goal lost the arm? (goal state only) ──
            const char * ended = nullptr;
            bool cancelled = false;
            {
                std::lock_guard<std::mutex> lock(goal_mutex_);
                if (active_ != goal) {
                    // Trajectory left alone: it belongs to the newer goal now.
                    ended = "Preempted by a newer goal";
                } else if (cancel_requested_ || goal->is_canceling()) {
                    ended = "Cancelled";
                    cancelled = true;
                }
            }
            if (ended) {
                if (cancelled) clear_trajectory_();
                RCLCPP_INFO(get_logger(), "%s", ended);
                finish_(goal, false, 0.0, ended);
                return;
            }

            // ── Progress (control state only) ──
            bool done = false;
            double final_error = 0.0;
            {
                std::lock_guard<std::mutex> lock(ctrl_mutex_);
                if (traj_points_.empty()) {
                    // An empty buffer must never read as "ran to completion".
                    ended = "Trajectory cleared";
                } else if (idx_ >= traj_points_.size()) {
                    done = true;
                    final_error = error_;
                    tracking_ = false;
                } else {
                    feedback->tracking = tracking_;
                    feedback->current_error = error_;
                    feedback->total_waypoints = static_cast<uint32_t>(traj_points_.size());
                    feedback->waypoints_remaining =
                        static_cast<uint32_t>(traj_points_.size() - idx_);
                }
            }
            if (ended) {
                RCLCPP_WARN(get_logger(), "%s", ended);
                finish_(goal, false, 0.0, ended);
                return;
            }
            if (done) {
                RCLCPP_INFO(get_logger(), "Trajectory complete (error=%f)", final_error);
                finish_(goal, true, final_error, "Target reached");
                return;
            }

            goal->publish_feedback(feedback);

            // Track the moving target, if this goal named one. Runs on the
            // worker thread, which is the only thread allowed to touch
            // planner_, and holds no lock while planning, so the control loop
            // keeps driving the current trajectory throughout.
            try_replan_(goal);

            std::unique_lock<std::mutex> lock(goal_mutex_);
            cv_.wait_for(lock, period);
        }

        finish_(goal, false, 0.0, "Node shutting down");
    }

    void finish_(const std::shared_ptr<GoalHandle> & goal,
                 bool success, double error, const std::string & message)
    {
        auto result = std::make_shared<MoveToPoseAction::Result>();
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
        // Only the goal that still owned the arm drops the subscription; a
        // preempted goal must not tear down the newer goal's target stream.
        if (was_active) clear_target_topic_();
        cv_.notify_all();   // wakes the cancel service waiting on this goal
    }

    Eigen::MatrixXd to_planning_frame_(const geometry_msgs::msg::PoseStamped & pose)
    {
        try {
            auto stamped = pose;
            stamped.header.stamp = rclcpp::Time(0);
            const auto out = tf_buffer_->transform(
                stamped, frame_,
                tf2::durationFromSec(get_parameter("tf_timeout").as_double()));
            Eigen::Isometry3d transformed;
            tf2::fromMsg(out.pose, transformed);
            return transformed.matrix();
        } catch (const tf2::TransformException & ex) {
            RCLCPP_ERROR(get_logger(), "TF lookup failed: %s", ex.what());
            return Eigen::MatrixXd();
        }
    }

    // ══════════════════ Control loop ══════════════════

    void control_tick_()
    {
        /* A tracking loop that quietly stops running is the worst failure this
         * node has, so it reports its own overruns. */
        const auto tick = std::chrono::steady_clock::now();
        if (last_tick_.time_since_epoch().count() != 0) {
            const double dt = std::chrono::duration<double>(tick - last_tick_).count();
            if (dt > 5.0 * control_period_) {
                RCLCPP_WARN(get_logger(), "Control loop overran: %.1f ms since last tick",
                            dt * 1e3);
            }
        }
        last_tick_ = tick;

        std::lock_guard<std::mutex> lock(ctrl_mutex_);

        if (state_.name.empty()) return;

        controller_model_->update(state_);
        left_ee_pub_->publish(convertToPoseStamped(
            controller_model_->get_current_left_ee_pose(), frame_, get_clock()->now()));
        right_ee_pub_->publish(convertToPoseStamped(
            controller_model_->get_current_right_ee_pose(), frame_, get_clock()->now()));

        // No trajectory means no commands. This is what cancelling relies on.
        if (traj_points_.empty() || idx_ >= traj_points_.size()) return;

        const double elapsed = (get_clock()->now() - traj_start_).seconds();
        const size_t last = traj_points_.size() - 1;

        while (idx_ < last &&
               rclcpp::Duration(traj_points_[idx_ + 1].time_from_start).seconds() <= elapsed) {
            ++idx_;
        }

        const auto & point = traj_points_[idx_];
        error_ = joint_error_(point.positions);

        sensor_msgs::msg::JointState cmd;
        cmd.header.stamp = get_clock()->now();
        cmd.name = traj_names_;
        cmd.position = point.positions;
        cmd.velocity = point.velocities;
        cmd.effort = feedforward_torque_(point);
        command_pub_->publish(cmd);

        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Waypoint %zu/%zu, error %f",
                             idx_ + 1, traj_points_.size(), error_);

        const bool finished = idx_ >= last &&
            rclcpp::Duration(traj_points_[last].time_from_start).seconds() <= elapsed;
        if (finished) {
            ++idx_;        // past the end: the monitor reads this as complete
            cv_.notify_all();
        }
    }

    std::vector<double> feedforward_torque_(
        const trajectory_msgs::msg::JointTrajectoryPoint & point)
    {
        const size_t n = traj_names_.size();

        JointState query;
        query.name = traj_names_;
        query.position = point.positions;
        query.velocity = point.velocities;
        query.effort = point.effort;
        query.position.resize(n, 0.0);
        query.velocity.resize(n, 0.0);
        query.effort.resize(n, 0.0);

        std::vector<double> accel = point.accelerations;
        accel.resize(n, 0.0);

        const JointState ff = controller_model_->inverse_dynamics(query, accel);

        std::unordered_map<std::string, double> by_name;
        for (size_t i = 0; i < ff.name.size() && i < ff.effort.size(); ++i) {
            by_name[ff.name[i]] = ff.effort[i];
        }

        std::vector<double> effort(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            auto it = by_name.find(traj_names_[i]);
            if (it != by_name.end()) effort[i] = it->second;
        }
        return effort;
    }

    /// RMS joint-position error of a target against feedback. Caller holds ctrl_mutex_.
    double joint_error_(const std::vector<double> & target)
    {
        double sum_sq = 0.0;
        size_t counted = 0;
        const size_t n = std::min(traj_names_.size(), target.size());
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < state_.name.size() && j < state_.position.size(); ++j) {
                if (traj_names_[i] == state_.name[j]) {
                    const double d = target[i] - state_.position[j];
                    sum_sq += d * d;
                    ++counted;
                    break;
                }
            }
        }
        return counted > 0 ? std::sqrt(sum_sq / counted) : 0.0;
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MoveToPoseServer>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
