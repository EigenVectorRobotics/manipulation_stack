#ifndef HELPER_FUNCS_H_
#define HELPER_FUNCS_H_

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <nav_msgs/msg/path.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp/clock.hpp>
#include <rclcpp/duration.hpp>
#include <g1_pilot/g1_pilot.h>
#include <stdio.h>

inline Eigen::Matrix4d create_se3(const Eigen::Quaterniond &q, const Eigen::Vector3d &t)
{
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    transform.block<3, 1>(0, 3) = t;
    return transform;
}

inline Eigen::Matrix4d create_se3(double qw, double qx, double qy, double qz,
                            double tx, double ty, double tz)
{
    Eigen::Quaterniond q(qw, qx, qy, qz);
    Eigen::Vector3d t(tx, ty, tz);
    return create_se3(q, t);
}

inline nav_msgs::msg::Path convertToPath(const std::vector<Eigen::MatrixXd>& poses, std::string frame_id, builtin_interfaces::msg::Time stamp) {
    nav_msgs::msg::Path path;
    geometry_msgs::msg::PoseStamped pose_stamped;

    path.header.frame_id = frame_id;
    path.header.stamp = stamp;
    
    pose_stamped.header = path.header;

    for (const auto& T : poses) {

        pose_stamped.pose.position.x = T(0, 3);
        pose_stamped.pose.position.y = T(1, 3);
        pose_stamped.pose.position.z = T(2, 3);

        Eigen::Quaterniond q(T.block<3, 3>(0, 0));
        q.normalize();
        
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        path.poses.push_back(pose_stamped);
    }

    return path;
};

inline std::vector<Eigen::VectorXd> toJointConfigs(
    const HumanoidPilot::JointTrajectory& traj)
{
    std::vector<Eigen::VectorXd> configs;
    if (traj.position.empty()) return configs;

    const std::size_t n_joints = traj.position.size();
    const std::size_t n_samples = traj.position.front().size();
    configs.reserve(n_samples);

    for (std::size_t k = 0; k < n_samples; ++k) {
        Eigen::VectorXd q(n_joints);
        for (std::size_t j = 0; j < n_joints; ++j) q(j) = traj.position[j][k];
        configs.push_back(std::move(q));
    }
    return configs;
}

inline trajectory_msgs::msg::JointTrajectory convertToTrajectory(
    const HumanoidPilot::JointTrajectory& traj,
    const std::string& frame_id,
    const rclcpp::Time& stamp)
{
    trajectory_msgs::msg::JointTrajectory out;
    out.header.frame_id = frame_id;
    out.header.stamp = stamp;
    out.joint_names = traj.name;

    if (traj.position.empty()) return out;

    const std::size_t n_joints = traj.position.size();
    const std::size_t n_samples = traj.position.front().size();
    out.points.reserve(n_samples);

    for (std::size_t k = 0; k < n_samples; ++k) {
        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions.resize(n_joints);
        point.velocities.resize(n_joints);
        point.accelerations.resize(n_joints);
        for (std::size_t j = 0; j < n_joints; ++j) {
            point.positions[j] = traj.position[j][k];
            point.velocities[j] = traj.velocity[j][k];
            point.accelerations[j] = traj.acceleration[j][k];
        }
        point.time_from_start = rclcpp::Duration::from_seconds(k * traj.time_step);
        out.points.push_back(std::move(point));
    }

    return out;
}

inline trajectory_msgs::msg::JointTrajectory convertToTrajectory(
    const std::vector<Eigen::VectorXd>& joint_traj,
    const std::vector<std::string>& joint_names,
    const std::string& frame_id,
    const rclcpp::Time& stamp)
{
    trajectory_msgs::msg::JointTrajectory traj;
    traj.header.frame_id = frame_id;
    traj.header.stamp = stamp;
    traj.joint_names = joint_names;
    traj.points.reserve(joint_traj.size());

    const std::size_t n = joint_names.size();
    for (const auto& q : joint_traj) {
        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions.assign(q.data(), q.data() + q.size());
        point.velocities.assign(n, 0.0);
        point.accelerations.assign(n, 0.0);
        point.effort.assign(n, 0.0);
        traj.points.push_back(std::move(point));
    }

    return traj;
}

inline std::vector<Eigen::MatrixXd> convertToTrajectory(const nav_msgs::msg::Path& path,
                                                bool reverse = false) {
    std::vector<Eigen::MatrixXd> trajectory;
    trajectory.reserve(path.poses.size());

    for (const auto& pose_stamped : path.poses) {
        const auto& p = pose_stamped.pose;
        trajectory.push_back(create_se3(
            p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z,
            p.position.x, p.position.y, p.position.z));
    }

    if (reverse) {
        std::reverse(trajectory.begin(), trajectory.end());
    }

    return trajectory;
}

inline geometry_msgs::msg::PoseStamped convertToPoseStamped(
    const Eigen::MatrixXd& T,
    const std::string& frame_id,
    const rclcpp::Time& stamp)
{
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.frame_id = frame_id;
    pose_stamped.header.stamp = stamp;

    pose_stamped.pose.position.x = T(0, 3);
    pose_stamped.pose.position.y = T(1, 3);
    pose_stamped.pose.position.z = T(2, 3);

    Eigen::Quaterniond q(T.block<3, 3>(0, 0));
    q.normalize();

    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    return pose_stamped;
}

inline bool isWithinLimits(const Eigen::MatrixXd& T,
                    std::string& reason,
                    const Eigen::Vector3d& minPos = Eigen::Vector3d(0.1, -0.6, 0.0),
                    const Eigen::Vector3d& maxPos = Eigen::Vector3d(0.6, 0.6, 1.0),
                    double maxAngle = 1.04 // max rotation angle from identity (pi/2)
                )
{
    Eigen::Vector3d pos = T.block<3, 1>(0, 3);
    Eigen::AngleAxisd aa(T.block<3, 3>(0, 0).cast<double>());
    double angle = aa.angle();
    Eigen::Vector3d axis = aa.axis();

    reason = "pos=[" + std::to_string(pos[0]) + ", " + std::to_string(pos[1]) + ", " + std::to_string(pos[2]) + "]"
           + " axis-angle: angle=" + std::to_string(angle)
           + " axis=[" + std::to_string(axis[0]) + ", " + std::to_string(axis[1]) + ", " + std::to_string(axis[2]) + "]"
           + " | violated: ";

    std::string violations;

    const char* axisNames[] = {"x", "y", "z"};
    for (int i = 0; i < 3; ++i) {
        if (pos[i] < minPos[i])
            violations += std::string(axisNames[i]) + " below min " + std::to_string(minPos[i]) + "; ";
        else if (pos[i] > maxPos[i])
            violations += std::string(axisNames[i]) + " above max " + std::to_string(maxPos[i]) + "; ";
    }

    if (angle > maxAngle)
        violations += "rotation angle " + std::to_string(angle) + " exceeds max " + std::to_string(maxAngle) + "; ";

    if (violations.empty()) {
        reason.clear();
        return true;
    }

    reason += violations;
    return false;
}

// Static transforms placing the left_ee and right_ee gripper frames on their wrist links.
inline std::vector<geometry_msgs::msg::TransformStamped> ee_static_transforms(
    const HumanoidPilot::Humanoid& humanoid, const rclcpp::Time& stamp)
{
    std::vector<geometry_msgs::msg::TransformStamped> out;
    for (const auto& [name, ee] : {std::make_pair(std::string("left_ee"), humanoid.get_left_ee()),
                                   std::make_pair(std::string("right_ee"), humanoid.get_right_ee())}) {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;
        t.header.frame_id = ee.parent_link;
        t.child_frame_id = name;
        t.transform.translation.x = ee.offset.x();
        t.transform.translation.y = ee.offset.y();
        t.transform.translation.z = ee.offset.z();
        t.transform.rotation.w = 1.0;
        out.push_back(t);
    }
    return out;
}

#endif