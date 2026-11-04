#include "core/body_ekf/body_ekf.h"

#include "rclcpp/rclcpp.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include "helper_funcs.h"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

using namespace HumanoidPilot;
using namespace HumanoidPilot::Attitude;
using namespace std::chrono_literals;

constexpr double DEG = M_PI / 180.0;

Eigen::Matrix4d to_matrix(const geometry_msgs::msg::Transform& t) {
    return create_se3(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z,
                      t.translation.x, t.translation.y, t.translation.z);
}

class BodyEKFNode : public rclcpp::Node
{
public:
    BodyEKFNode() : Node("body_ekf")
    {   
        auto string_param = [this](std::string& var, std::string param, std::string default_val){
            var = this->declare_parameter<std::string>(param, default_val);
        };

        // (1) Humanoid model, robot configuration and noise.
        std::string description_share = ament_index_cpp::get_package_share_directory("g1_description");
        std::string g1_pilot_share = ament_index_cpp::get_package_share_directory("g1_pilot");
        RobotConfig config;
        string_param(config.asset_file, "asset_file", description_share + "/assets/g1/g1_29dof_with_hand_rev_1_0.urdf");
        string_param(config.asset_root, "asset_root", description_share + "/assets/g1/");
        string_param(config.config_file, "config_file", g1_pilot_share + "/config/g1.yaml");
        
        humanoid_ = std::make_shared<Humanoid>(&config);
        robot_ = declare_robot_();
        params_ = declare_ekf_params_();
        calibration_duration_ = this->declare_parameter<double>("calibration_duration", 2.0);
        still_threshold_ = this->declare_parameter<double>("still_threshold", 0.15);

        // (2) Frames, output options and TF.
        tf_decimation_ = std::max<int64_t>(1, this->declare_parameter<int64_t>("tf_decimation", 1));
        pose_decimation_ = std::max<int64_t>(1, this->declare_parameter<int64_t>("pose_publish_decimation", 4));
        
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        string_param(fixed_frame_, "fixed_frame", "foot_world");
        string_param(camera_link_frame_, "camera_link_frame", "camera_link");
        string_param(camera_imu_frame_, "camera_imu_frame", "camera_imu_optical_frame");
        string_param(pelvis_imu_topic_, "pelvis_imu_topic", "/pelvis_imu");
        string_param(torso_imu_topic_, "torso_imu_topic", "/torso_imu");
        string_param(joint_states_topic_, "joint_states_topic", "/feedback");
        string_param(torso_pose_topic_, "torso_pose_topic", "/body_ekf/torso_pose");

        // (3) Publishers and subscribers.
        const auto qos = rclcpp::SensorDataQoS();
        torso_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(torso_pose_topic_, 5);
        joint_sub_ = encoder_subscription_(joint_states_topic_);
        pelvis_imu_sub_ = imu_subscription_(pelvis_imu_topic_, Body::Pelvis);
        torso_imu_sub_ = imu_subscription_(torso_imu_topic_, Body::Torso);

        // (4) Camera IMU after its TF resolves, or the torso gyro alone right away.
        const std::string robot_name = this->declare_parameter<std::string>("robot_name", "unknown");
        if (params_.use_camera_imu) {
            camera_imu_sub_ = imu_subscription_(camera_imu_frame_, Body::Camera);
            camera_timer_ = this->create_wall_timer(500ms, std::bind(&BodyEKFNode::resolve_camera_frames_, this));
            RCLCPP_INFO(this->get_logger(), "body_ekf (%s): waiting for camera TF, then a still period",
                        robot_name.c_str());
        } else {
            start_estimator_(Eigen::Matrix3d::Identity());
            RCLCPP_INFO(this->get_logger(), "body_ekf (%s): camera IMU off; waiting for a still period",
                        robot_name.c_str());
        }
    }

private:
    // Model and estimator
    std::shared_ptr<Humanoid> humanoid_;
    BodyConfig robot_;
    EKFParams params_;
    double calibration_duration_ = 2.0;
    double still_threshold_ = 0.15;
    std::unique_ptr<OnlineBodyEstimator> estimator_;

    // Frames and output options
    std::string fixed_frame_;
    std::string camera_link_frame_;
    std::string camera_imu_frame_;
    int64_t tf_decimation_ = 1;
    int64_t pose_decimation_ = 4;
    int64_t grid_count_ = 0;

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr camera_timer_;

    // Publishers and subscribers
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr torso_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr pelvis_imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr torso_imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr camera_imu_sub_;
    std::string pelvis_imu_topic_;
    std::string torso_imu_topic_;
    std::string joint_states_topic_;
    std::string torso_pose_topic_;

    BodyConfig declare_robot_()
    {
        using Doubles = std::vector<double>;
        return make_body_config(
            this->declare_parameter<Doubles>("camera_mount_xyz", Doubles{0.0, 0.0, 0.0}),
            this->declare_parameter<Doubles>("camera_mount_rpy_deg", Doubles{0.0, 0.0, 0.0}),
            this->declare_parameter<Doubles>("camera_mount_quat_xyzw", Doubles{}));
    }

    EKFParams declare_ekf_params_()
    {
        EKFParams p;
        p.sig_acc = this->declare_parameter<double>("sig_acc_deg", 3.0) * DEG;
        p.sig_waist = this->declare_parameter<double>("sig_waist_deg", 10.0) * DEG;
        p.sig_leg_tilt = this->declare_parameter<double>("sig_leg_tilt_deg", 1.0) * DEG;
        p.sig_leg_yaw = this->declare_parameter<double>("sig_leg_yaw_deg", 1.0) * DEG;
        p.sig_bias_walk = this->declare_parameter<double>("sig_bias_walk", 1e-5);
        p.gyro_density_pelvis = this->declare_parameter<double>("gyro_density_pelvis", 0.0);
        p.gyro_density_torso = this->declare_parameter<double>("gyro_density_torso", 0.0);
        p.gyro_density_sensor = this->declare_parameter<double>("gyro_density_sensor", 0.0);
        p.use_camera_imu = this->declare_parameter<bool>("use_camera_imu", false);
        return p;
    }

    void resolve_camera_frames_()
    {
        Eigen::Matrix4d T_cl_imu;
        try {
            T_cl_imu = to_matrix(tf_buffer_->lookupTransform(
                camera_link_frame_, camera_imu_frame_, tf2::TimePointZero).transform);
        } catch (const tf2::TransformException& e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Camera TF not available yet: %s",
                                 e.what());
            return;
        }
        // The camera IMU frame is fixed on the torso: torso <- camera_link (mount) <- IMU (driver TF).
        start_estimator_((robot_.T_torso_camera_link * T_cl_imu).topLeftCorner<3, 3>());
        camera_timer_->cancel();
        RCLCPP_INFO(this->get_logger(), "body_ekf: camera TF resolved; collecting still period");
    }

    void start_estimator_(const Eigen::Matrix3d& R_torso_sensor)
    {
        estimator_ = std::make_unique<OnlineBodyEstimator>(
            std::make_unique<BodyAttitudeEstimator>(humanoid_, robot_, R_torso_sensor, params_),
            calibration_duration_, still_threshold_);
    }

    // Pelvis and torso IMU axes are taken as the pelvis and torso link axes.
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_(
        const std::string& topic, Body body)
    {
        return this->create_subscription<sensor_msgs::msg::Imu>(topic, rclcpp::SensorDataQoS(),
            [this, body](const sensor_msgs::msg::Imu::SharedPtr msg) {
                if (!estimator_) {
                    return;
                }
                const auto& w = msg->angular_velocity;
                const auto& a = msg->linear_acceleration;
                estimator_->add_imu(
                    body, rclcpp::Time(msg->header.stamp).nanoseconds(), 
                    Eigen::Vector3d(w.x, w.y, w.z),
                    Eigen::Vector3d(a.x, a.y, a.z)
                );
                process_();
            });
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr encoder_subscription_(const std::string topic)
    {
        return this->create_subscription<sensor_msgs::msg::JointState>(topic, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::JointState::SharedPtr msg){
                if (!estimator_) {
                    return;
                }
                estimator_->add_joint_state(
                    rclcpp::Time(msg->header.stamp).nanoseconds(), 
                    JointState{msg->name, msg->position, msg->velocity, {}}
                );
                process_();
            }
        );
    }

    void process_()
    {
        const ProcessReport report = estimator_->process();
        if (report.calibrated) {
            const Eigen::Vector3d d = estimator_->estimator().calibration().gyro_density;
            RCLCPP_INFO(this->get_logger(),
                "body_ekf: calibrated; gyro densities [rad/s/sqrt(Hz)] pelvis %.2e torso %.2e camera %.2e",
                d(0), d(1), d(2));
        }
        if (report.dropped > 0) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Dropping grid samples older than the joint / IMU history (check sensor clocks)");
        }
        if (report.camera_ahead) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Camera IMU stamps are >1 s ahead of the joint stamps (check sensor clocks)");
        }
        for (const auto& output : report.outputs) {
            publish_(output);
        }
    }

    void publish_(const GridOutput& output)
    {
        const rclcpp::Time stamp(output.t, this->get_clock()->get_clock_type());
        if (grid_count_ % pose_decimation_ == 0) {
            torso_pub_->publish(convertToPoseStamped(output.T_foot_torso, fixed_frame_, stamp));
        }
        if (grid_count_ % tf_decimation_ == 0) {
            const auto pose = convertToPoseStamped(output.T_foot_torso.inverse(), robot_.torso_link, stamp);
            geometry_msgs::msg::TransformStamped message;
            message.header = pose.header;
            message.child_frame_id = fixed_frame_;
            message.transform.translation.x = pose.pose.position.x;
            message.transform.translation.y = pose.pose.position.y;
            message.transform.translation.z = pose.pose.position.z;
            message.transform.rotation = pose.pose.orientation;
            tf_broadcaster_->sendTransform(message);
        }
        ++grid_count_;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BodyEKFNode>());
    rclcpp::shutdown();
    return 0;
}
