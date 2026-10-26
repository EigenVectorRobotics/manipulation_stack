#include "core/body_ekf/body_ekf.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace HumanoidPilot {
namespace Attitude {

namespace {

constexpr double DEG = M_PI / 180.0;
constexpr int A = 0;
constexpr int B = 3;
constexpr int BP = 6;
constexpr int BT = 9;
constexpr int BS = 12;
constexpr std::size_t MAX_PENDING_IMU = 400;
constexpr std::size_t STILL_SMOOTH_SAMPLES = 40;
constexpr int64_t CAMERA_AHEAD_NS = 1000000000;
// Substrings naming the leg and waist joints whose speeds decide stillness.
const std::vector<std::string> STILL_JOINT_KEYS = {"hip", "knee", "ankle", "waist"};
const Eigen::Vector3d EZ = Eigen::Vector3d::UnitZ();

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d K;
    K << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;
    return K;
}

Eigen::Matrix3d axis_angle_matrix(const Eigen::Vector3d& axis, double angle) {
    const Eigen::Matrix3d K = skew(axis.normalized());
    return Eigen::Matrix3d::Identity() + std::sin(angle) * K + (1.0 - std::cos(angle)) * K * K;
}

Eigen::Matrix3d min_rotation(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const Eigen::Vector3d v = a.cross(b);
    const double s = v.norm();
    if (s < 1e-12) {
        return Eigen::Matrix3d::Identity();
    }
    return axis_angle_matrix(v / s, std::atan2(s, a.dot(b)));
}

Eigen::Matrix3d mean_rotation(const std::vector<Eigen::Matrix3d>& rotations) {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const auto& R : rotations) {
        sum += so3_log(R);
    }
    return so3_exp(sum / static_cast<double>(rotations.size()));
}

Eigen::Vector3d mean_vector(const std::vector<Eigen::Vector3d>& x) {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const auto& v : x) {
        sum += v;
    }
    return sum / static_cast<double>(x.size());
}

std::map<std::string, double> joint_map(
    const std::vector<std::string>& names,
    const std::vector<double>& values,
    double scale
){
    if (names.size() != values.size()) {
        throw std::invalid_argument("Robot config: joint names and values differ in length");
    }
    std::map<std::string, double> out;
    for (std::size_t i = 0; i < names.size(); ++i) {
        out[names[i]] = values[i] * scale;
    }
    return out;
}

double gyro_density(const std::vector<Eigen::Vector3d>& gyro, double given, double fallback, double grid_period) {
    if (given > 0.0) {
        return given;
    }
    if (gyro.size() <= 10) {
        return fallback;
    }
    const Eigen::Vector3d mean = mean_vector(gyro);
    Eigen::Vector3d var = Eigen::Vector3d::Zero();
    for (const auto& v : gyro) {
        var += (v - mean).cwiseAbs2();
    }
    // Mean over axes of the population standard deviation, scaled to a white-noise density.
    const double measured = (var / static_cast<double>(gyro.size())).cwiseSqrt().mean() * std::sqrt(grid_period);
    return std::isfinite(measured) && measured > 0.0 ? measured : fallback;
}

StillCalibration calibrate_still(
    const std::vector<Eigen::Vector3d>& gyro_p,
    const std::vector<Eigen::Vector3d>& gyro_t,
    const std::vector<Eigen::Vector3d>& gyro_s,
    const std::vector<Eigen::Vector3d>& up_p,
    const std::vector<Eigen::Vector3d>& up_t,
    const std::vector<Eigen::Vector3d>& up_s,
    const std::vector<Relations>& relations,
    const EKFParams& params
){
    StillCalibration c;

    // (1) Same rigid body: rotate the sensor accel so its mean matches the torso accel.
    const bool sensor = params.use_camera_imu;
    c.R_sensor_align = sensor ? min_rotation(mean_vector(up_s).normalized(), mean_vector(up_t).normalized())
                              : Eigen::Matrix3d::Identity();

    // (2) Encoder relations vs IMU tilt + FK heading.
    std::vector<Eigen::Matrix3d> waist_errors;
    std::vector<Eigen::Matrix3d> leg_errors;
    for (std::size_t k = 0; k < up_p.size(); ++k) {
        const Eigen::Matrix3d& W = relations[k].W;
        const Eigen::Matrix3d& L = relations[k].L;
        const Eigen::Matrix3d R_p = tilt_yaw_rotation(up_p[k], heading(L));
        const Eigen::Matrix3d R_t = tilt_yaw_rotation(up_t[k], heading(L * W));
        waist_errors.push_back(W.transpose() * R_p.transpose() * R_t);
        leg_errors.push_back(L.transpose() * R_p);
    }
    c.C_waist = mean_rotation(waist_errors);
    c.C_leg = mean_rotation(leg_errors);

    // (3) Gyro biases and white-noise densities; without the sensor its density is infinite.
    c.bias_p = mean_vector(gyro_p);
    c.bias_t = mean_vector(gyro_t);
    c.bias_s = sensor ? mean_vector(gyro_s) : Eigen::Vector3d::Zero();
    c.gyro_density = Eigen::Vector3d(
        gyro_density(gyro_p, params.gyro_density_pelvis, params.fallback_density_pelvis, params.grid_period),
        gyro_density(gyro_t, params.gyro_density_torso, params.fallback_density_torso, params.grid_period),
        sensor ? gyro_density(gyro_s, params.gyro_density_sensor, params.fallback_density_sensor, params.grid_period)
               : std::numeric_limits<double>::infinity());
    return c;
}

bool is_still_joint(const std::string& name) {
    return std::any_of(STILL_JOINT_KEYS.begin(), STILL_JOINT_KEYS.end(),
                       [&](const std::string& key) { return name.find(key) != std::string::npos; });
}

} // anonymous namespace

Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w) {
    const double angle = w.norm();
    if (angle < 1e-12) {
        return Eigen::Matrix3d::Identity();
    }
    return axis_angle_matrix(w / angle, angle);
}

Eigen::Vector3d so3_log(const Eigen::Matrix3d& R) {
    const Eigen::Vector3d v(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
    const double angle = std::acos(std::clamp((R.trace() - 1.0) / 2.0, -1.0, 1.0));
    if (angle < 1e-9) {
        return v / 2.0;
    }
    return angle / (2.0 * std::sin(angle)) * v;
}

Eigen::Matrix3d rpy_matrix(double roll, double pitch, double yaw) {
    return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())
          * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
          * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
}

double heading(const Eigen::Matrix3d& R) {
    return std::atan2(R(1, 0), R(0, 0));
}

Eigen::Matrix3d tilt_yaw_rotation(const Eigen::Vector3d& up, double yaw) {
    // World <- body rotation whose body-frame up direction is `up` and whose heading is `yaw`.
    const Eigen::Matrix3d R = min_rotation(EZ, up.normalized()).transpose();
    return Eigen::AngleAxisd(yaw - heading(R), EZ).toRotationMatrix() * R;
}

Body2EKF::Body2EKF(
    const Eigen::Matrix3d& R_p,
    const Eigen::Matrix3d& R_t,
    const Eigen::Vector3d& bias_p,
    const Eigen::Vector3d& bias_t,
    const Eigen::Vector3d& bias_s,
    const Eigen::Matrix3d& R_torso_sensor,
    const Eigen::Vector3d& gyro_density,
    double sig_bias_walk,
    const Matrix15d& P0
): R_p(R_p), R_t(R_t), bias_p(bias_p), bias_t(bias_t), bias_s(bias_s), P(P0),
   R_ts_(R_torso_sensor), sig_gyro_p_(gyro_density(0)), sig_bias_walk_(sig_bias_walk)
{
    // Inverse-variance weights of the torso and sensor gyros and the blended noise density; infinite: torso only.
    const double var_t = gyro_density(1) * gyro_density(1);
    const double var_s = gyro_density(2) * gyro_density(2);
    const bool blend = std::isfinite(var_s);
    w_t = blend ? var_s / (var_t + var_s) : 1.0;
    w_s = blend ? var_t / (var_t + var_s) : 0.0;
    sig_gyro_blend_ = blend ? std::sqrt(w_t * w_t * var_t + w_s * w_s * var_s) : gyro_density(1);
}

void Body2EKF::propagate(
    const Eigen::Vector3d& gyro_p,
    const Eigen::Vector3d& gyro_t,
    const Eigen::Vector3d& gyro_s,
    double dt
){
    // (1) Bias-corrected body rates.
    const Eigen::Vector3d w_p = gyro_p - bias_p;
    const Eigen::Vector3d w_torso = w_t * (gyro_t - bias_t) + w_s * (R_ts_ * (gyro_s - bias_s));
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    // (2) Error-state transition and process noise.
    Matrix15d F = Matrix15d::Identity();
    F.block<3, 3>(A, A) = so3_exp(-w_p * dt);
    F.block<3, 3>(A, BP) = -dt * I;
    F.block<3, 3>(B, B) = so3_exp(-w_torso * dt);
    F.block<3, 3>(B, BT) = -w_t * dt * I;
    F.block<3, 3>(B, BS) = -w_s * dt * R_ts_;
    Matrix15d Q = Matrix15d::Zero();
    Q.block<3, 3>(A, A) = I * sig_gyro_p_ * sig_gyro_p_ * dt;
    Q.block<3, 3>(B, B) = I * sig_gyro_blend_ * sig_gyro_blend_ * dt;
    for (int s : {BP, BT, BS}) {
        Q.block<3, 3>(s, s) = I * sig_bias_walk_ * sig_bias_walk_ * dt;
    }

    // (3) Covariance and attitude.
    P = F * P * F.transpose() + Q;
    R_p = R_p * so3_exp(w_p * dt);
    R_t = R_t * so3_exp(w_torso * dt);
}

void Body2EKF::update(
    const Eigen::Matrix<double, 3, 15>& H,
    const Eigen::Vector3d& residual,
    const Eigen::Matrix3d& R_meas
){
    const Eigen::Matrix3d S = H * P * H.transpose() + R_meas;
    const Eigen::Matrix<double, 15, 3> K = P * H.transpose() * S.inverse();
    const Vector15d dx = K * residual;
    P = (Matrix15d::Identity() - K * H) * P;
    P = 0.5 * (P + P.transpose()).eval();
    R_p = R_p * so3_exp(dx.segment<3>(A));
    R_t = R_t * so3_exp(dx.segment<3>(B));
    bias_p += dx.segment<3>(BP);
    bias_t += dx.segment<3>(BT);
    bias_s += dx.segment<3>(BS);
}

void Body2EKF::gravity(Body body, const Eigen::Vector3d& up, double sig) {
    const bool pelvis = body == Body::Pelvis;
    const Eigen::Vector3d predicted = (pelvis ? R_p : R_t).transpose() * EZ;
    Eigen::Matrix<double, 3, 15> H = Eigen::Matrix<double, 3, 15>::Zero();
    H.block<3, 3>(0, pelvis ? A : B) = skew(predicted);
    update(H, up - predicted, Eigen::Matrix3d::Identity() * sig * sig);
}

void Body2EKF::waist(const Eigen::Matrix3d& W, double sig) {
    const Eigen::Matrix3d M = R_p.transpose() * R_t;
    Eigen::Matrix<double, 3, 15> H = Eigen::Matrix<double, 3, 15>::Zero();
    H.block<3, 3>(0, A) = -M.transpose();
    H.block<3, 3>(0, B) = Eigen::Matrix3d::Identity();
    update(H, -so3_log(W.transpose() * M), Eigen::Matrix3d::Identity() * sig * sig);
}

void Body2EKF::leg(const Eigen::Matrix3d& L, double sig_tilt, double sig_yaw) {
    // Residual in world axes so tilt and yaw get separate noise.
    const Eigen::Vector3d residual = R_p * -so3_log(L.transpose() * R_p);
    Eigen::Matrix<double, 3, 15> H = Eigen::Matrix<double, 3, 15>::Zero();
    H.block<3, 3>(0, A) = R_p;
    const Eigen::Vector3d var(sig_tilt * sig_tilt, sig_tilt * sig_tilt, sig_yaw * sig_yaw);
    update(H, residual, var.asDiagonal());
}

BodyConfig make_body_config(
    const std::vector<double>& camera_mount_xyz,
    const std::vector<double>& camera_mount_rpy_deg,
    const std::vector<double>& camera_mount_quat_xyzw,
    const std::vector<std::string>& joint_offset_names,
    const std::vector<double>& joint_offset_deg,
    const std::vector<std::string>& joint_gain_names,
    const std::vector<double>& joint_gain_values
){
    // (1) Camera mount: torso_link <- camera_link.
    const auto& q = camera_mount_quat_xyzw;
    const auto& rpy = camera_mount_rpy_deg;
    BodyConfig robot;
    if (q.size() == 4) {
        robot.T_torso_camera_link.topLeftCorner<3, 3>() = Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized()
                                                              .toRotationMatrix();
    } else if (rpy.size() == 3) {
        robot.T_torso_camera_link.topLeftCorner<3, 3>() = rpy_matrix(rpy[0] * DEG, rpy[1] * DEG, rpy[2] * DEG);
    }
    if (camera_mount_xyz.size() == 3) {
        robot.T_torso_camera_link.topRightCorner<3, 1>() =
            Eigen::Vector3d(camera_mount_xyz[0], camera_mount_xyz[1], camera_mount_xyz[2]);
    }

    // (2) Encoder corrections.
    robot.joint_offsets = joint_map(joint_offset_names, joint_offset_deg, DEG);
    robot.joint_gains = joint_map(joint_gain_names, joint_gain_values, 1.0);
    return robot;
}

JointState correct_joints(
    const JointState& state,
    const std::map<std::string, double>& offsets,
    const std::map<std::string, double>& gains
){
    JointState out = state;
    for (std::size_t i = 0; i < out.name.size(); ++i) {
        const auto gain = gains.find(out.name[i]);
        const auto offset = offsets.find(out.name[i]);
        out.position[i] = out.position[i] * (gain != gains.end() ? gain->second : 1.0)
                        + (offset != offsets.end() ? offset->second : 0.0);
    }
    return out;
}

BodyAttitudeEstimator::BodyAttitudeEstimator(
    std::shared_ptr<Humanoid> humanoid,
    const BodyConfig& robot,
    const Eigen::Matrix3d& R_torso_sensor,
    const EKFParams& params
): humanoid_(std::move(humanoid)), robot_(robot), R_ts_(R_torso_sensor), params_(params)
{ }

Relations BodyAttitudeEstimator::relations(const JointState& joints) {
    const JointState corrected = correct_joints(joints, robot_.joint_offsets, robot_.joint_gains);
    const Eigen::Matrix4d T_base_torso = humanoid_->frame_transform(corrected, robot_.base_link, robot_.torso_link);
    const Eigen::Matrix4d T_foot_base =
        humanoid_->frame_transform(corrected, robot_.foot_link, robot_.base_link);
    return {T_base_torso.topLeftCorner<3, 3>(), T_foot_base.topLeftCorner<3, 3>(), T_foot_base * T_base_torso};
}

Eigen::Vector3d BodyAttitudeEstimator::sensor_up(const Eigen::Vector3d& acc_s) const {
    return (R_ts_ * acc_s).normalized();
}

void BodyAttitudeEstimator::initialize(const std::vector<GridSample>& still, const std::vector<GridSample>* init) {
    // (1) Calibration on the still period.
    std::vector<Eigen::Vector3d> gyro_p, gyro_t, gyro_s, up_p, up_t, up_s;
    std::vector<Relations> rel;
    for (const auto& s : still) {
        gyro_p.push_back(s.gyro_p);
        gyro_t.push_back(s.gyro_t);
        gyro_s.push_back(s.gyro_s);
        up_p.push_back(s.acc_p.normalized());
        up_t.push_back(s.acc_t.normalized());
        up_s.push_back(sensor_up(s.acc_s));
        rel.push_back(relations(s.joints));
    }
    calibration_ = calibrate_still(gyro_p, gyro_t, gyro_s, up_p, up_t, up_s, rel, params_);

    // (2) Initial tilt from the head of `init`, heading from FK.
    const auto& head = init != nullptr ? *init : still;
    const std::size_t n = std::min<std::size_t>(params_.init_samples, head.size());
    Eigen::Vector3d mean_up_p = Eigen::Vector3d::Zero();
    Eigen::Vector3d mean_up_t = Eigen::Vector3d::Zero();
    for (std::size_t k = 0; k < n; ++k) {
        mean_up_p += head[k].acc_p.normalized() / static_cast<double>(n);
        mean_up_t += head[k].acc_t.normalized() / static_cast<double>(n);
    }
    const Relations rel0 = relations(head.front().joints);

    // (3) Filter.
    const auto& p = params_;
    const auto& c = calibration_;
    Vector15d diag;
    diag << Eigen::Matrix<double, 6, 1>::Constant(p.sig_init_attitude * p.sig_init_attitude),
            Eigen::Matrix<double, 9, 1>::Constant(p.sig_init_bias * p.sig_init_bias);
    ekf_.emplace(tilt_yaw_rotation(mean_up_p, heading(rel0.L)), tilt_yaw_rotation(mean_up_t, heading(rel0.L * rel0.W)),
                 c.bias_p, c.bias_t, c.bias_s, R_ts_, c.gyro_density, p.sig_bias_walk, Matrix15d(diag.asDiagonal()));
    last_.reset();
    step_ = 0;
}

Eigen::Matrix4d BodyAttitudeEstimator::step(const GridSample& sample) {
    const auto& p = params_;
    const auto& c = calibration_;
    Body2EKF& f = *ekf_;
    const Relations rel = relations(sample.joints);
    if (last_) {
        f.propagate(last_->gyro_p, last_->gyro_t, last_->gyro_s, static_cast<double>(sample.t - last_->t) * 1e-9);
    }
    f.gravity(Body::Pelvis, sample.acc_p.normalized(), p.sig_acc);
    f.gravity(Body::Torso, sample.acc_t.normalized(), p.sig_acc);
    if (p.use_camera_imu) {
        f.gravity(Body::Torso, c.R_sensor_align * sensor_up(sample.acc_s), p.sig_acc);
    }
    if (step_ % p.encoder_decimation == 0) {
        f.waist(rel.W * c.C_waist, p.sig_waist);
        f.leg(rel.L * c.C_leg, p.sig_leg_tilt, p.sig_leg_yaw);
    }
    last_ = sample;
    ++step_;
    Eigen::Matrix4d T_foot_torso = rel.T_foot_torso;
    T_foot_torso.topLeftCorner<3, 3>() = f.R_t;
    return T_foot_torso;
}

SampleBuffer::SampleBuffer(double horizon): horizon_ns_(static_cast<int64_t>(horizon * 1e9)) { }

void SampleBuffer::append(int64_t t, const Eigen::VectorXd& row) {
    if (!t_.empty() && t <= t_.back()) {
        return;
    }
    t_.push_back(t);
    rows_.push_back(row);
    while (t_.back() - t_.front() > horizon_ns_) {
        t_.pop_front();
        rows_.pop_front();
    }
}

void SampleBuffer::clear() {
    t_.clear();
    rows_.clear();
}

bool SampleBuffer::covers(int64_t t, double margin) const {
    return t_.size() > 1 && t_.front() < t && t_.back() >= t + static_cast<int64_t>(margin * 1e9);
}

Eigen::VectorXd SampleBuffer::boxcar(int64_t t, double window) const {
    // (1) Recent samples and the kernel length in median sample periods.
    const double start = static_cast<double>(t) - std::max(10.0 * window, 0.05) * 1e9;
    const auto first = static_cast<std::size_t>(
        std::lower_bound(t_.begin(), t_.end(), start, [](int64_t a, double b) { return a < b; }) - t_.begin());
    const std::size_t n = t_.size() - first;
    std::vector<double> diffs;
    for (std::size_t i = first + 1; i < t_.size(); ++i) {
        diffs.push_back(static_cast<double>(t_[i] - t_[i - 1]));
    }
    if (diffs.empty()) {
        return interp(t);
    }
    const std::size_t mid = diffs.size() / 2;
    std::nth_element(diffs.begin(), diffs.begin() + mid, diffs.end());
    double median = diffs[mid];
    if (diffs.size() % 2 == 0) {
        median = 0.5 * (median + *std::max_element(diffs.begin(), diffs.begin() + mid));
    }
    const long k = std::max(1L, std::lrint(window / median * 1e9));

    // (2) Zero-padded centred average at one sample, matching numpy's convolve(..., 'same').
    auto smoothed = [&](std::size_t m) {
        Eigen::VectorXd sum = Eigen::VectorXd::Zero(rows_.front().size());
        const long lo = std::max(0L, static_cast<long>(m) - k / 2);
        const long hi = std::min(static_cast<long>(n) - 1, static_cast<long>(m) + (k - 1) / 2);
        for (long i = lo; i <= hi; ++i) {
            sum += rows_[first + i];
        }
        return Eigen::VectorXd(sum / static_cast<double>(k));
    };

    // (3) Linear interpolation at t, clamped at the ends.
    if (t <= t_[first]) {
        return smoothed(0);
    }
    if (t >= t_.back()) {
        return smoothed(n - 1);
    }
    const std::size_t j = static_cast<std::size_t>(
        std::upper_bound(t_.begin() + first, t_.end(), t) - t_.begin()) - 1 - first;
    const double a = static_cast<double>(t - t_[first + j]) / static_cast<double>(t_[first + j + 1] - t_[first + j]);
    return (1.0 - a) * smoothed(j) + a * smoothed(j + 1);
}

Eigen::VectorXd SampleBuffer::interp(int64_t t) const {
    if (t <= t_.front()) {
        return rows_.front();
    }
    if (t >= t_.back()) {
        return rows_.back();
    }
    const std::size_t j = static_cast<std::size_t>(std::upper_bound(t_.begin(), t_.end(), t) - t_.begin()) - 1;
    const double a = static_cast<double>(t - t_[j]) / static_cast<double>(t_[j + 1] - t_[j]);
    return (1.0 - a) * rows_[j] + a * rows_[j + 1];
}

StillPeriodCollector::StillPeriodCollector(
    double duration,
    double threshold,
    std::size_t smooth_samples,
    double grid_period
): required_(static_cast<std::size_t>(std::lround(duration / grid_period))),
   threshold_(threshold), smooth_samples_(smooth_samples)
{ }

bool StillPeriodCollector::add(const GridSample& sample, double joint_speed) {
    speeds_.push_back(joint_speed);
    if (speeds_.size() > smooth_samples_) {
        speeds_.pop_front();
    }
    const double mean = std::accumulate(speeds_.begin(), speeds_.end(), 0.0) / static_cast<double>(speeds_.size());
    if (mean >= threshold_) {
        rows_.clear();
        return false;
    }
    rows_.push_back(sample);
    return rows_.size() >= required_;
}

OnlineBodyEstimator::OnlineBodyEstimator(
    std::unique_ptr<BodyAttitudeEstimator> estimator,
    double calibration_duration,
    double still_threshold
): estimator_(std::move(estimator)),
   collector_(calibration_duration, still_threshold, STILL_SMOOTH_SAMPLES, estimator_->params().grid_period),
   window_(estimator_->params().resample_window)
{ }

void OnlineBodyEstimator::add_joint_state(int64_t t, const JointState& state) {
    const std::size_t n = state.name.size();
    if (state.name != joint_names_) {
        joint_names_ = state.name;
        still_joints_.assign(n, false);
        std::transform(state.name.begin(), state.name.end(), still_joints_.begin(), is_still_joint);
        joints_.clear();
    }
    Eigen::VectorXd row = Eigen::VectorXd::Zero(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        row(i) = state.position[i];
        row(n + i) = i < state.velocity.size() ? state.velocity[i] : 0.0;
    }
    joints_.append(t, row);
}

void OnlineBodyEstimator::add_imu(Body body, int64_t t, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc) {
    Eigen::VectorXd row(6);
    row << gyro, acc;
    (body == Body::Pelvis ? pelvis_ : torso_).append(t, row);
    if (body == Body::Torso && !estimator_->params().use_camera_imu) {
        queue_grid(t);
    }
}

void OnlineBodyEstimator::add_camera_imu(int64_t t, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc) {
    if (!estimator_->params().use_camera_imu) {
        return;
    }
    pending_.push_back({t, gyro, acc});
    if (pending_.size() > MAX_PENDING_IMU) {
        pending_.pop_front();
    }
}

void OnlineBodyEstimator::queue_grid(int64_t t) {
    // Regular grid up to the torso-IMU stamp t; it restarts one period after t on a clock jump.
    const auto period = static_cast<int64_t>(std::llround(estimator_->params().grid_period * 1e9));
    if (!next_grid_ || std::llabs(t - *next_grid_) > static_cast<int64_t>(MAX_PENDING_IMU) * period) {
        next_grid_ = t + period;
    }
    for (; *next_grid_ <= t; *next_grid_ += period) {
        pending_.push_back({*next_grid_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()});
    }
    while (pending_.size() > MAX_PENDING_IMU) {
        pending_.pop_front();
    }
}

OnlineBodyEstimator::Stamp OnlineBodyEstimator::classify(int64_t t, ProcessReport& report) const {
    if (joints_.covers(t, window_) && pelvis_.covers(t, window_) && torso_.covers(t, window_)) {
        return Stamp::Ready;
    }
    int64_t oldest = std::numeric_limits<int64_t>::min();
    for (const SampleBuffer* buffer : {&joints_, &pelvis_, &torso_}) {
        if (!buffer->empty()) oldest = std::max(oldest, buffer->oldest());
    }
    if (t <= oldest) {
        return Stamp::Stale;
    }
    if (!joints_.empty() && t > joints_.latest() + CAMERA_AHEAD_NS) {
        report.camera_ahead = true;
    }
    return Stamp::Wait;
}

GridSample OnlineBodyEstimator::resample(const PendingImu& imu, double& joint_speed) const {
    // (1) IMUs boxcar-averaged, joints interpolated at the camera-IMU stamp.
    const Eigen::VectorXd pelvis = pelvis_.boxcar(imu.t, window_);
    const Eigen::VectorXd torso = torso_.boxcar(imu.t, window_);
    const Eigen::VectorXd joints = joints_.interp(imu.t);
    const std::size_t n = joint_names_.size();
    GridSample sample;
    sample.t = imu.t;
    sample.gyro_p = pelvis.head<3>();
    sample.acc_p = pelvis.tail<3>();
    sample.gyro_t = torso.head<3>();
    sample.acc_t = torso.tail<3>();
    sample.gyro_s = imu.gyro;
    sample.acc_s = imu.acc;
    sample.joints.name = joint_names_;
    sample.joints.position.assign(joints.data(), joints.data() + n);

    // (2) Leg and waist speed for the still detection.
    joint_speed = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        joint_speed += still_joints_[i] ? std::abs(joints(n + i)) : 0.0;
    }
    return sample;
}

ProcessReport OnlineBodyEstimator::process() {
    ProcessReport report;
    while (!pending_.empty()) {
        const Stamp stamp = classify(pending_.front().t, report);
        if (stamp == Stamp::Wait) {
            break;
        }
        const PendingImu imu = pending_.front();
        pending_.pop_front();
        if (stamp == Stamp::Stale) {
            ++report.dropped;
            continue;
        }
        double joint_speed = 0.0;
        const GridSample sample = resample(imu, joint_speed);
        if (estimator_->initialized()) {
            report.outputs.push_back({sample.t, estimator_->step(sample)});
        } else if (collector_.add(sample, joint_speed)) {
            estimator_->initialize(collector_.samples());
            report.calibrated = true;
        }
    }
    return report;
}

} // Attitude namespace
} // HumanoidPilot namespace
