#include "core/arm_mp/joint_space_planner.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <random>

using HumanoidPilot::JointSpacePlanner;
using HumanoidPilot::JointState;
using HumanoidPilot::JointTrajectory;

namespace {

JointState stateWithVelocities(const std::vector<double>& velocities) {
    JointState state;
    state.velocity = velocities;
    for (std::size_t j = 0; j < velocities.size(); ++j) {
        state.name.push_back("joint_" + std::to_string(j));
        state.position.push_back(0.1 * static_cast<double>(j));
    }
    return state;
}

void verifySegment(const JointTrajectory& trajectory, std::size_t joint, std::size_t sample,
                   double max_velocity, double max_acceleration) {
    const double dt = trajectory.time_step;
    const double q0 = trajectory.position[joint][sample];
    const double q1 = trajectory.position[joint][sample + 1];
    const double v0 = trajectory.velocity[joint][sample];
    const double v1 = trajectory.velocity[joint][sample + 1];
    const double c2 = 3 * (q1 - q0) / (dt * dt) - (2 * v0 + v1) / dt;
    const double c3 = 2 * (q0 - q1) / (dt * dt * dt) + (v0 + v1) / (dt * dt);
    for (int point = 0; point <= 10; ++point) {
        const double t = dt * point / 10.0;
        const double velocity = v0 + 2 * c2 * t + 3 * c3 * t * t;
        const double acceleration = 2 * c2 + 6 * c3 * t;
        EXPECT_LE(std::abs(velocity), max_velocity + 1e-8);
        EXPECT_LE(std::abs(acceleration), max_acceleration + 1e-7);
        EXPECT_GE(velocity * v0, -1e-8);
    }
}

void verifyJoint(const JointState& state, const JointTrajectory& trajectory, std::size_t joint,
                 double max_velocity, double max_acceleration) {
    const auto& velocity = trajectory.velocity[joint];
    EXPECT_DOUBLE_EQ(trajectory.position[joint].front(), state.position[joint]);
    EXPECT_DOUBLE_EQ(velocity.front(), state.velocity[joint]);
    EXPECT_DOUBLE_EQ(velocity.back(), 0.0);
    EXPECT_DOUBLE_EQ(trajectory.acceleration[joint].back(), 0.0);
    const auto stopped = std::find(velocity.begin(), velocity.end(), 0.0);
    const double stop_time = std::distance(velocity.begin(), stopped) * trajectory.time_step;
    double max_speed = 0.0;
    for (double value : state.velocity) max_speed = std::max(max_speed, std::abs(value));
    const double minimum_time = state.velocity[joint] == 0.0 ? 0.0 : max_speed / max_acceleration;
    EXPECT_GE(stop_time + 1e-12, minimum_time);
    EXPECT_LT(stop_time - minimum_time, trajectory.time_step + 1e-12);
    for (std::size_t k = 0; k + 1 < velocity.size(); ++k) {
        EXPECT_LE(std::abs(velocity[k + 1]), std::abs(velocity[k]));
        EXPECT_LE(std::abs(trajectory.acceleration[joint][k]), max_acceleration + 1e-12);
        verifySegment(trajectory, joint, k, max_velocity, max_acceleration);
    }
}

TEST(Brake, StopsAllJointsAtEarliestSharedSample) {
    const auto state = stateWithVelocities({1.0, -0.413, 0.0, 0.02});
    const auto trajectory = JointSpacePlanner::brake(state);
    EXPECT_EQ(trajectory.name, state.name);
    EXPECT_EQ(trajectory.position.front().size(), 51u);
    EXPECT_DOUBLE_EQ(trajectory.time_step, 0.01);
    EXPECT_NEAR(trajectory.position[0].back(), 0.25, 1e-12);
    EXPECT_NEAR(trajectory.position[1].back(), state.position[1] - 0.413 * 0.25, 1e-12);
    for (std::size_t j = 0; j < state.name.size(); ++j) verifyJoint(state, trajectory, j, 1.0, 2.0);
}

TEST(Brake, StationaryStateHasOneSample) {
    const auto state = stateWithVelocities({0.0, 0.0});
    const auto trajectory = JointSpacePlanner::brake(state);
    ASSERT_EQ(trajectory.position.front().size(), 1u);
    for (std::size_t j = 0; j < state.name.size(); ++j) verifyJoint(state, trajectory, j, 1.0, 2.0);
}

TEST(Brake, EnforcesLimitsBetweenSamplesAcrossRandomStates) {
    std::mt19937 generator(7);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (int trial = 0; trial < 100; ++trial) {
        std::vector<double> velocities(7);
        for (auto& velocity : velocities) velocity = uniform(generator);
        const auto state = stateWithVelocities(velocities);
        const double dt = 0.003 + 0.002 * (trial % 6);
        const double max_acc = 0.5 + 0.3 * (trial % 9);
        const auto trajectory = JointSpacePlanner::brake(state, dt, 1.0, max_acc);
        for (std::size_t j = 0; j < state.name.size(); ++j) verifyJoint(state, trajectory, j, 1.0, max_acc);
    }
}

} // namespace
