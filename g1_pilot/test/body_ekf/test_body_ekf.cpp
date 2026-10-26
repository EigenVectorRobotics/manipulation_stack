#include "core/body_ekf/body_ekf.h"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <memory>
#include <random>

using namespace HumanoidPilot;
using namespace HumanoidPilot::Attitude;

constexpr double DEG = M_PI / 180.0;
const Eigen::Vector3d EZ = Eigen::Vector3d::UnitZ();
const std::vector<std::string> BODY_JOINTS = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_elbow_joint",
};

std::shared_ptr<Humanoid> humanoid() {
    static std::shared_ptr<Humanoid> handle = [] {
        RobotConfig cfg;
        cfg.asset_root = G1_ASSET_ROOT;
        cfg.asset_file = std::string(G1_ASSET_ROOT) + "g1_29dof_with_hand_rev_1_0.urdf";
        cfg.config_file = G1_CONFIG_FILE;
        return std::make_shared<Humanoid>(&cfg);
    }();
    return handle;
}

JointState joint_state(const std::vector<double>& position) {
    JointState state;
    state.name = BODY_JOINTS;
    state.position = position;
    state.velocity.assign(position.size(), 0.0);
    return state;
}

JointState random_joints(std::mt19937& rng, double range) {
    std::uniform_real_distribution<double> u(-range, range);
    std::vector<double> q(BODY_JOINTS.size());
    for (auto& v : q) {
        v = u(rng);
    }
    return joint_state(q);
}

double angle_deg(const Eigen::Matrix3d& A, const Eigen::Matrix3d& B) {
    return so3_log(A.transpose() * B).norm() / DEG;
}

double tilt_deg(const Eigen::Matrix3d& A, const Eigen::Matrix3d& B) {
    return std::acos(std::clamp((A.transpose() * EZ).dot(B.transpose() * EZ), -1.0, 1.0)) / DEG;
}

Body2EKF make_filter(
    const Eigen::Matrix3d& R_p,
    const Eigen::Matrix3d& R_t,
    const Eigen::Matrix3d& R_ts = Eigen::Matrix3d::Identity(),
    double sig_attitude = 10.0 * DEG,
    const Eigen::Vector3d& gyro_density = Eigen::Vector3d::Constant(2e-4)
){
    Vector15d diag;
    diag << Eigen::Matrix<double, 6, 1>::Constant(sig_attitude * sig_attitude),
            Eigen::Matrix<double, 9, 1>::Constant(std::pow(0.05 * DEG, 2));
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    return Body2EKF(R_p, R_t, zero, zero, zero, R_ts, gyro_density, 1e-5, Matrix15d(diag.asDiagonal()));
}

const Eigen::Vector3d TORSO_ONLY(2e-4, 2e-4, std::numeric_limits<double>::infinity());

TEST(BodyEKF, GravityOnlyConvergesTilt) {
    const Eigen::Matrix3d R_p_true = rpy_matrix(0.05, -0.08, 0.3);
    const Eigen::Matrix3d R_t_true = rpy_matrix(-0.1, 0.25, 0.2);
    Body2EKF f = make_filter(so3_exp(Eigen::Vector3d(0.08, -0.06, 0.0)) * R_p_true,
                             so3_exp(Eigen::Vector3d(-0.07, 0.05, 0.1)) * R_t_true);
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    for (int k = 0; k < 400; ++k) {
        f.propagate(zero, zero, zero, 5e-3);
        f.gravity(Body::Pelvis, R_p_true.transpose() * EZ, 3.0 * DEG);
        f.gravity(Body::Torso, R_t_true.transpose() * EZ, 3.0 * DEG);
    }
    EXPECT_LT(tilt_deg(f.R_p, R_p_true), 0.05);
    EXPECT_LT(tilt_deg(f.R_t, R_t_true), 0.05);
}

TEST(BodyEKF, GyroPropagationIntegratesRatesAndBlendsSensor) {
    const Eigen::Matrix3d R_ts = rpy_matrix(M_PI / 2, 0.3, -0.2);
    const Eigen::Matrix3d R0 = rpy_matrix(0.1, -0.2, 0.3);
    Body2EKF f = make_filter(R0, R0, R_ts);
    const Eigen::Vector3d w_p(0.2, -0.1, 0.4);
    const Eigen::Vector3d w_t(-0.3, 0.5, 0.1);
    for (int k = 0; k < 200; ++k) {
        f.propagate(w_p, w_t, R_ts.transpose() * w_t, 5e-3);
    }
    EXPECT_LT(angle_deg(f.R_p, R0 * so3_exp(w_p)), 1e-9);
    EXPECT_LT(angle_deg(f.R_t, R0 * so3_exp(w_t)), 1e-9);
    EXPECT_DOUBLE_EQ(f.w_t, 0.5);
    for (int i = 0; i < 6; ++i) {
        EXPECT_GT(f.P(i, i), std::pow(10.0 * DEG, 2));
    }
}

TEST(BodyEKF, TorsoOnlyPropagationIgnoresSensorGyro) {
    const Eigen::Matrix3d R0 = rpy_matrix(0.1, -0.2, 0.3);
    Body2EKF f = make_filter(R0, R0, rpy_matrix(M_PI / 2, 0.3, -0.2), 10.0 * DEG, TORSO_ONLY);
    const Eigen::Vector3d w_t(-0.3, 0.5, 0.1);
    for (int k = 0; k < 200; ++k) {
        f.propagate(Eigen::Vector3d::Zero(), w_t, Eigen::Vector3d(5.0, -4.0, 3.0), 5e-3);
    }
    EXPECT_DOUBLE_EQ(f.w_t, 1.0);
    EXPECT_DOUBLE_EQ(f.w_s, 0.0);
    EXPECT_LT(angle_deg(f.R_t, R0 * so3_exp(w_t)), 1e-9);
    EXPECT_TRUE(f.P.allFinite());
    EXPECT_TRUE((f.P.block<3, 3>(3, 12).isZero(0.0)));
}

TEST(BodyEKF, TorsoOnlyGravityConvergesTiltAndTorsoBias) {
    const Eigen::Matrix3d R_true = rpy_matrix(-0.1, 0.25, 0.2);
    Body2EKF f = make_filter(R_true, so3_exp(Eigen::Vector3d(-0.07, 0.05, 0.0)) * R_true, Eigen::Matrix3d::Identity(),
                             10.0 * DEG, TORSO_ONLY);
    const Eigen::Vector3d bias(1e-3, -1e-3, 0.0);
    for (int k = 0; k < 4000; ++k) {
        f.propagate(Eigen::Vector3d::Zero(), bias, Eigen::Vector3d::Zero(), 5e-3);
        f.gravity(Body::Pelvis, R_true.transpose() * EZ, 3.0 * DEG);
        f.gravity(Body::Torso, R_true.transpose() * EZ, 3.0 * DEG);
    }
    EXPECT_LT(tilt_deg(f.R_t, R_true), 0.05);
    EXPECT_LT((f.bias_t - bias).head<2>().cwiseAbs().maxCoeff(), 3e-4);
    EXPECT_TRUE(f.bias_s.isZero(0.0));
}

TEST(BodyEKF, GyroBiasEstimatedFromGravityAndEncoders) {
    const Eigen::Matrix3d R_true = rpy_matrix(0.05, 0.1, 0.0);
    Body2EKF f = make_filter(R_true, R_true, Eigen::Matrix3d::Identity(), 0.5 * DEG);
    const Eigen::Vector3d bias(0.01, -0.02, 0.005);
    for (int k = 0; k < 4000; ++k) {
        f.propagate(bias, bias, bias, 5e-3);
        f.gravity(Body::Pelvis, R_true.transpose() * EZ, 3.0 * DEG);
        f.gravity(Body::Torso, R_true.transpose() * EZ, 3.0 * DEG);
        f.leg(R_true, 1.0 * DEG, 1.0 * DEG);
        f.waist(Eigen::Matrix3d::Identity(), 10.0 * DEG);
    }
    EXPECT_LT((f.bias_p - bias).cwiseAbs().maxCoeff(), 2e-3);
}

TEST(BodyEKF, WaistAndLegUpdatesEnforceEncoderRelations) {
    const Eigen::Matrix3d W = rpy_matrix(0.1, 0.3, -0.4);
    const Eigen::Matrix3d R_p = rpy_matrix(0.02, -0.05, 0.1);
    Body2EKF f = make_filter(R_p, R_p * W * so3_exp(Eigen::Vector3d(0.1, -0.08, 0.12)));
    for (int k = 0; k < 50; ++k) {
        f.waist(W, 0.5 * DEG);
        f.leg(R_p, 0.1 * DEG, 0.1 * DEG);
    }
    EXPECT_LT(angle_deg(f.R_p.transpose() * f.R_t, W), 0.05);
    EXPECT_LT(angle_deg(f.R_p, R_p), 0.05);
}

TEST(BodyEKF, BodyConfigsParse) {
    const BodyConfig stanley = make_body_config({0.081099, 0.023467, 0.414921}, {1.3, 3.4, -1.592365}, {});
    EXPECT_NEAR(stanley.T_torso_camera_link(2, 3), 0.414921, 1e-12);
    const BodyConfig patricia = make_body_config({0.05762, 0.01753, 0.42987}, {}, {0.0, 0.40355, 0.0, 0.91496});
    EXPECT_LT(angle_deg(patricia.T_torso_camera_link.topLeftCorner<3, 3>(), rpy_matrix(0.0, 47.6 * DEG, 0.0)), 0.01);
}

TEST(BodyEKF, HumanoidFrameTransformMatchesKnownPoses) {
    Eigen::Matrix4d T = humanoid()->frame_transform(joint_state(std::vector<double>(10, 0.0)), "pelvis",
                                                    "left_ankle_roll_link");
    EXPECT_TRUE((T.topLeftCorner<3, 3>().isIdentity(1e-12)));
    EXPECT_TRUE((T.topRightCorner<3, 1>().isApprox(Eigen::Vector3d(0.0, 0.118506, -0.756864), 1e-5)));
    const double yaw = 0.4, roll = -0.2, pitch = 0.3;
    T = humanoid()->frame_transform(joint_state({0, 0, 0, 0, 0, 0, yaw, roll, pitch, 0}), "pelvis", "torso_link");
    const Eigen::Matrix3d Rz = rpy_matrix(0.0, 0.0, yaw);
    EXPECT_TRUE((T.topLeftCorner<3, 3>().isApprox(Rz * rpy_matrix(roll, 0.0, 0.0) * rpy_matrix(0.0, pitch, 0.0),
                                                  1e-12)));
    EXPECT_TRUE((T.topRightCorner<3, 1>().isApprox(Rz * Eigen::Vector3d(-0.0039635, 0.0, 0.044), 1e-12)));
    EXPECT_THROW(humanoid()->frame_transform(JointState(), "pelvis", "no_such_link"), std::invalid_argument);
}

TEST(BodyEKF, RelationsComposeToFootTorso) {
    BodyAttitudeEstimator estimator(humanoid(), BodyConfig(), Eigen::Matrix3d::Identity());
    std::mt19937 rng(3);
    for (int k = 0; k < 5; ++k) {
        const Relations rel = estimator.relations(random_joints(rng, 0.3));
        EXPECT_TRUE((rel.T_foot_torso.topLeftCorner<3, 3>().isApprox(rel.L * rel.W, 1e-12)));
    }
}

TEST(BodyEKF, TiltYawRotationSetsUpAndHeading) {
    const Eigen::Vector3d up(0.1, -0.2, 0.97);
    const Eigen::Matrix3d R = tilt_yaw_rotation(up, 0.7);
    EXPECT_TRUE((R.transpose() * EZ).isApprox(up.normalized(), 1e-12));
    EXPECT_NEAR(heading(R), 0.7, 1e-12);
}

TEST(BodyEKF, SampleBufferAveragesAndInterpolates) {
    SampleBuffer buffer(0.5);
    for (int i = 0; i < 100; ++i) {
        Eigen::VectorXd row(2);
        row << i, (i % 2) ? 1.0 : -1.0;
        buffer.append(static_cast<int64_t>(i) * 1000000, row);
    }
    buffer.append(10000000, Eigen::VectorXd::Constant(2, 9.0));
    EXPECT_NEAR(buffer.boxcar(50000000, 4e-3)(1), 0.0, 1e-12);
    EXPECT_NEAR(buffer.boxcar(50500000, 4e-3)(0), 50.0, 0.6);
    EXPECT_NEAR(buffer.interp(50250000)(0), 50.25, 1e-9);
    EXPECT_TRUE(buffer.covers(90000000, 5e-3));
    EXPECT_FALSE(buffer.covers(99000000, 5e-3));
}

TEST(BodyEKF, StillCollectorRestartsOnMotion) {
    StillPeriodCollector collector(0.1, 0.15, 4);
    const GridSample sample;
    for (int k = 0; k < 10; ++k) {
        EXPECT_FALSE(collector.add(sample, 0.0));
    }
    collector.add(sample, 14.5);
    EXPECT_TRUE(collector.samples().empty());
    for (int k = 0; k < 3; ++k) {
        collector.add(sample, 0.0);
    }
    EXPECT_TRUE(collector.samples().empty());
    bool ready = false;
    for (int k = 0; k < 20; ++k) {
        ready = collector.add(sample, 0.0);
    }
    EXPECT_TRUE(ready);
}

struct StillRun {
    std::vector<GridOutput> outputs;
    bool calibrated = false;
    Relations rel;
};

StillRun run_still_robot(bool use_camera_imu) {
    // (1) Synthetic still robot: level foot, waist pitched forward, camera IMU equal to the torso IMU.
    EKFParams params;
    params.use_camera_imu = use_camera_imu;
    auto estimator = std::make_unique<BodyAttitudeEstimator>(humanoid(), BodyConfig(), Eigen::Matrix3d::Identity(),
                                                             params);
    const JointState joints = joint_state({0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0});
    StillRun run;
    run.rel = estimator->relations(joints);
    const Eigen::Matrix3d R_p = run.rel.L;
    const Eigen::Matrix3d R_t = run.rel.L * run.rel.W;
    OnlineBodyEstimator online(std::move(estimator), 0.5);
    std::mt19937 rng(5);
    std::normal_distribution<double> noise(0.0, 1.0);
    auto jitter = [&](double sig) -> Eigen::Vector3d {
        return Eigen::Vector3d(noise(rng), noise(rng), noise(rng)) * sig;
    };
    const Eigen::Vector3d bias(0.003, -0.002, 0.001);

    // (2) 1 kHz joint and body IMU streams and a 200 Hz camera IMU for 3 s.
    for (int ms = 0; ms < 3000; ++ms) {
        const int64_t t = static_cast<int64_t>(ms) * 1000000;
        online.add_joint_state(t, joints);
        online.add_imu(Body::Pelvis, t, bias + jitter(2e-3), R_p.transpose() * EZ * 9.81 + jitter(0.02));
        online.add_imu(Body::Torso, t, bias + jitter(2e-3), R_t.transpose() * EZ * 9.81 + jitter(0.02));
        if (ms % 5 == 0) {
            online.add_imu(Body::Camera, t, bias + jitter(2e-3), R_t.transpose() * EZ * 9.81 + jitter(0.02));
        }
        const ProcessReport report = online.process();
        run.calibrated = run.calibrated || report.calibrated;
        run.outputs.insert(run.outputs.end(), report.outputs.begin(), report.outputs.end());
    }
    return run;
}

TEST(BodyEKF, OnlineEstimatorTracksStaticRobot) {
    for (bool use_camera_imu : {true, false}) {
        const StillRun run = run_still_robot(use_camera_imu);
        ASSERT_TRUE(run.calibrated);
        ASSERT_GT(run.outputs.size(), 300u);
        EXPECT_LT(angle_deg(run.outputs.back().T_foot_torso.topLeftCorner<3, 3>(), run.rel.L * run.rel.W), 0.5);
        EXPECT_TRUE((run.outputs.back().T_foot_torso.topRightCorner<3, 1>().isApprox(
            run.rel.T_foot_torso.topRightCorner<3, 1>(), 1e-12)));
        // The torso-IMU grid steps every 5 ms with and without the camera IMU.
        for (std::size_t k = 1; k < run.outputs.size(); ++k) {
            EXPECT_EQ(run.outputs[k].t - run.outputs[k - 1].t, 5000000);
        }
    }
}
