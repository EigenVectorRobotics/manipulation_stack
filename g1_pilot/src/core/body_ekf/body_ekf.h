#ifndef BODY_EKF_H
#define BODY_EKF_H

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace HumanoidPilot {
namespace Attitude {

using Matrix15d = Eigen::Matrix<double, 15, 15>;
using Vector15d = Eigen::Matrix<double, 15, 1>;

Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w);
Eigen::Vector3d so3_log(const Eigen::Matrix3d& R);
// URDF fixed-axis convention: R = Rz(yaw) Ry(pitch) Rx(roll).
Eigen::Matrix3d rpy_matrix(double roll, double pitch, double yaw);
double heading(const Eigen::Matrix3d& R);
Eigen::Matrix3d tilt_yaw_rotation(const Eigen::Vector3d& up, double yaw);

enum class Body { Pelvis, Torso, Camera };

// Error state: [pelvis rotation, torso rotation, b_pelvis, b_torso, b_sensor], rotations in body axes.
class Body2EKF {
public:
    Body2EKF(
        const Eigen::Matrix3d& R_p,
        const Eigen::Matrix3d& R_t,
        const Eigen::Vector3d& bias_p,
        const Eigen::Vector3d& bias_t,
        const Eigen::Vector3d& bias_s,
        const Eigen::Matrix3d& R_torso_sensor,
        const Eigen::Vector3d& gyro_density,
        double sig_bias_walk,
        const Matrix15d& P0
    );

    void propagate(
        const Eigen::Vector3d& gyro_p,
        const Eigen::Vector3d& gyro_t,
        const Eigen::Vector3d& gyro_s,
        double dt
    );

    void gravity(Body body, const Eigen::Vector3d& up, double sig);
    void waist(const Eigen::Matrix3d& W, double sig);
    void leg(const Eigen::Matrix3d& L, double sig_tilt, double sig_yaw);

    Eigen::Matrix3d R_p;
    Eigen::Matrix3d R_t;
    Eigen::Vector3d bias_p;
    Eigen::Vector3d bias_t;
    Eigen::Vector3d bias_s;
    Matrix15d P;

    double w_t = 0.5;
    double w_s = 0.5;

private:
    void update(const Eigen::Matrix<double, 3, 15>& H, const Eigen::Vector3d& residual, const Eigen::Matrix3d& R_meas);

    Eigen::Matrix3d R_ts_;
    double sig_gyro_p_;
    double sig_gyro_blend_;
    double sig_bias_walk_;

}; // Body2EKF class

struct BodyConfig {
    Eigen::Matrix4d T_torso_camera_link = Eigen::Matrix4d::Identity();
    std::string base_link = "pelvis";
    std::string torso_link = "torso_link";
    std::string foot_link = "left_ankle_roll_link";
    std::string other_foot_link = "right_ankle_roll_link";
};

// Camera rotation from the quaternion when it has 4 values, else from rpy (deg).
BodyConfig make_body_config(
    const std::vector<double>& camera_mount_xyz,
    const std::vector<double>& camera_mount_rpy_deg,
    const std::vector<double>& camera_mount_quat_xyzw
);

// Angles in rad, gyro densities in rad/s/sqrt(Hz); a density <= 0 is measured on the still period.
struct EKFParams {
    double sig_acc = 3.0 * M_PI / 180.0;
    double sig_waist = 10.0 * M_PI / 180.0;
    double sig_leg_tilt = 1.0 * M_PI / 180.0;
    double sig_leg_yaw = 1.0 * M_PI / 180.0;
    double sig_bias_walk = 1e-5;
    double gyro_density_pelvis = 0.0;
    double gyro_density_torso = 0.0;
    double gyro_density_sensor = 0.0;
    double fallback_density_pelvis = 2.6e-4;
    double fallback_density_torso = 1.8e-4;
    double fallback_density_sensor = 2.8e-4;
    double sig_init_attitude = 0.5 * M_PI / 180.0;
    double sig_init_bias = 0.05 * M_PI / 180.0;
    double grid_period = 5e-3;
    double resample_window = 5e-3;
    int encoder_decimation = 2;
    int init_samples = 50;
    bool use_camera_imu = false;    // off: torso gyro only
};

// Encoder kinematics at one stamp; L_other only feeds the still detection.
struct Relations {
    Eigen::Matrix3d W = Eigen::Matrix3d::Identity();             // pelvis <- torso
    Eigen::Matrix3d L = Eigen::Matrix3d::Identity();             // foot <- pelvis
    Eigen::Matrix4d T_foot_torso = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d L_other = Eigen::Matrix3d::Identity();       // other foot <- pelvis
};

Relations make_relations(
    const Eigen::Matrix4d& T_base_torso,
    const Eigen::Matrix4d& T_foot_base,
    const Eigen::Matrix4d& T_other_foot_base
);

// Sum of the angular rates (rad/s) of W, L and L_other between two stamps dt s apart.
double relation_speed(const Relations& a, const Relations& b, double dt);

// Sensor data at one grid stamp (ns); IMU vectors in their own frames.
struct GridSample {
    int64_t t = 0;
    Eigen::Vector3d gyro_p = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_t = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_s = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc_p = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d acc_t = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d acc_s = Eigen::Vector3d::UnitZ();
    Relations rel;
};

struct StillCalibration {
    Eigen::Vector3d bias_p;
    Eigen::Vector3d bias_t;
    Eigen::Vector3d bias_s;
    Eigen::Matrix3d R_sensor_align;  // aligns the sensor-accel up direction to the torso accel
    Eigen::Matrix3d C_waist;         // true pelvis <- torso = W(q) C_waist
    Eigen::Matrix3d C_leg;           // true foot <- pelvis = L(q) C_leg
    Eigen::Vector3d gyro_density;    // pelvis, torso, sensor
};

// Runs Body2EKF on grid samples; R_torso_sensor is torso <- camera IMU frame.
class BodyAttitudeEstimator {
public:
    explicit BodyAttitudeEstimator(const Eigen::Matrix3d& R_torso_sensor, const EKFParams& params = EKFParams());

    // Calibrates on `still`; the first params.init_samples of `init` (default `still`) set the initial tilt.
    void initialize(const std::vector<GridSample>& still, const std::vector<GridSample>* init = nullptr);
    bool initialized() const { return ekf_.has_value(); }
    void reset();

    // Propagates to sample.t with the previous gyros, fuses this sample and returns foot <- torso.
    Eigen::Matrix4d step(const GridSample& sample);

    const StillCalibration& calibration() const { return calibration_; }
    const EKFParams& params() const { return params_; }

private:
    Eigen::Vector3d sensor_up(const Eigen::Vector3d& acc_s) const;

    Eigen::Matrix3d R_ts_;
    EKFParams params_;
    StillCalibration calibration_;
    std::optional<Body2EKF> ekf_;
    std::optional<GridSample> last_;
    long step_ = 0;

}; // BodyAttitudeEstimator class

// Recent time-stamped rows (ns) of one stream, resampled on-line at grid stamps.
class SampleBuffer {
public:
    explicit SampleBuffer(double horizon = 0.5);

    void append(int64_t t, const Eigen::VectorXd& row);
    void clear();
    bool empty() const { return t_.empty(); }
    int64_t oldest() const { return t_.front(); }
    int64_t latest() const { return t_.back(); }

    // True when samples exist from before t to at least margin seconds after it.
    bool covers(int64_t t, double margin) const;

    // Centred boxcar average over `window` seconds, interpolated at t.
    Eigen::VectorXd boxcar(int64_t t, double window) const;
    Eigen::VectorXd interp(int64_t t) const;

private:
    int64_t horizon_ns_;
    std::deque<int64_t> t_;
    std::deque<Eigen::VectorXd> rows_;

}; // SampleBuffer class

// Collects grid samples until `duration` s of continuous stillness; any motion restarts the collection.
class StillPeriodCollector {
public:
    StillPeriodCollector(
        double duration = 2.0,
        double threshold = 0.05,
        std::size_t smooth_samples = 40,
        double grid_period = 5e-3
    );

    // `speed` is the leg and waist motion in rad/s (relation_speed); true once enough data is held.
    bool add(const GridSample& sample, double speed);
    const std::vector<GridSample>& samples() const { return rows_; }
    void clear();

private:
    std::size_t required_;
    double threshold_;
    std::size_t smooth_samples_;
    std::deque<double> speeds_;
    std::vector<GridSample> rows_;

}; // StillPeriodCollector class

struct GridOutput {
    int64_t t;
    Eigen::Matrix4d T_foot_torso;
};

struct ProcessReport {
    std::vector<GridOutput> outputs;
    bool calibrated = false;        // the still-period calibration finished in this call
    int dropped = 0;                // grid stamps older than the IMU or TF history
    bool camera_ahead = false;      // camera-IMU stamps lead the torso-IMU stamps by > 1 s
};

enum class Lookup { Found, Wait, Missing };

// Fills the relations at a grid stamp (ns); Wait until the source reaches it, Missing when it never will.
using RelationLookup = std::function<Lookup(int64_t t, Relations& relations)>;

// A regular grid on the torso-IMU stamps; IMUs are resampled and relations looked up there to feed the still-period
// calibration, then the EKF.
class OnlineBodyEstimator {
public:
    OnlineBodyEstimator(
        std::unique_ptr<BodyAttitudeEstimator> estimator,
        RelationLookup lookup,
        double calibration_duration = 2.0,
        double still_threshold = 0.05
    );

    void add_imu(Body body, int64_t t, const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc);

    ProcessReport process();
    const BodyAttitudeEstimator& estimator() const { return *estimator_; }

    // Recalibrates on the next still period without dropping the sensor streams; no output until then.
    void reset();

private:
    enum class Stamp { Ready, Wait, Stale };

    Stamp classify(int64_t t, ProcessReport& report) const;
    GridSample resample(int64_t t, const Relations& rel) const;
    void queue_grid(int64_t t);

    std::unique_ptr<BodyAttitudeEstimator> estimator_;
    RelationLookup lookup_;
    StillPeriodCollector collector_;
    SampleBuffer pelvis_;
    SampleBuffer torso_;
    SampleBuffer camera_;
    std::deque<int64_t> pending_;
    std::optional<int64_t> next_grid_;
    std::optional<GridSample> last_sample_;
    double window_;

}; // OnlineBodyEstimator class

} // Attitude namespace
} // HumanoidPilot namespace

#endif // BODY_EKF_H
