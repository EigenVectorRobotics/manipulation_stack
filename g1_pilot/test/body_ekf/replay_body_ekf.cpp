#include "core/body_ekf/body_ekf.h"

#include <g1_pilot/g1_pilot.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace HumanoidPilot;
using namespace HumanoidPilot::Attitude;

using Table = std::vector<std::vector<double>>;
using Meta = std::map<std::string, std::vector<std::string>>;

Table read_csv(const std::string& path) {
    Table rows;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        rows.push_back(row);
    }
    return rows;
}

Meta read_meta(const std::string& path) {
    Meta meta;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string key, value;
        ss >> key;
        while (ss >> value) {
            meta[key].push_back(value);
        }
    }
    return meta;
}

std::vector<double> numbers(const Meta& meta, const std::string& key) {
    std::vector<double> out;
    if (meta.count(key)) {
        for (const auto& v : meta.at(key)) {
            out.push_back(std::stod(v));
        }
    }
    return out;
}

std::vector<std::string> strings(const Meta& meta, const std::string& key) {
    return meta.count(key) ? meta.at(key) : std::vector<std::string>{};
}

Eigen::Vector3d vec3(const std::vector<double>& row, int begin) {
    return Eigen::Vector3d(row[begin], row[begin + 1], row[begin + 2]);
}

JointState joint_state(const std::vector<std::string>& names, const std::vector<double>& row, int begin) {
    JointState state;
    state.name = names;
    const std::size_t n = names.size();
    state.position.assign(row.begin() + begin, row.begin() + begin + n);
    if (row.size() >= static_cast<std::size_t>(begin) + 2 * n) {
        state.velocity.assign(row.begin() + begin + n, row.begin() + begin + 2 * n);
    }
    return state;
}

std::shared_ptr<Humanoid> make_humanoid(const std::string& asset_root, const std::string& config_file) {
    RobotConfig cfg;
    cfg.asset_root = asset_root;
    cfg.asset_file = asset_root + "/g1_29dof_with_hand_rev_1_0.urdf";
    cfg.config_file = config_file;
    return std::make_shared<Humanoid>(&cfg);
}

// The node looks these up on TF; offline they come from FK on the recorded joints.
Relations fk_relations(Humanoid& humanoid, const JointState& joints) {
    const BodyConfig robot;
    return make_relations(humanoid.frame_transform(joints, robot.base_link, robot.torso_link),
                          humanoid.frame_transform(joints, robot.foot_link, robot.base_link),
                          humanoid.frame_transform(joints, robot.other_foot_link, robot.base_link));
}

std::unique_ptr<BodyAttitudeEstimator> make_estimator(const Meta& meta) {
    EKFParams params;
    const auto densities = numbers(meta, "gyro_densities");
    params.gyro_density_pelvis = densities.at(0);
    params.gyro_density_torso = densities.at(1);
    params.gyro_density_sensor = densities.at(2);
    const auto camera = numbers(meta, "use_camera_imu");
    params.use_camera_imu = camera.empty() || camera[0] != 0.0;
    const auto r = numbers(meta, "R_torso_sensor");
    const Eigen::Matrix3d R_torso_sensor = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(r.data());
    return std::make_unique<BodyAttitudeEstimator>(R_torso_sensor, params);
}

void write_row(std::FILE* out, int64_t t, const Eigen::Matrix4d& T) {
    std::fprintf(out, "%lld", static_cast<long long>(t));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::fprintf(out, ",%.17g", T(i, j));
        }
    }
    std::fprintf(out, "\n");
}

// Grid CSV columns: t, gyro_p, gyro_t, gyro_s, acc_p, acc_t, acc_s, joint positions (meta joint_names order).
void run_grid(std::unique_ptr<BodyAttitudeEstimator> estimator, Humanoid& humanoid, const Meta& meta,
              const std::string& csv, std::FILE* out)
{
    // (1) Grid samples and the still hold used for calibration.
    const auto names = strings(meta, "joint_names");
    const auto window = numbers(meta, "hold_ns");
    std::vector<GridSample> samples, hold;
    for (const auto& row : read_csv(csv)) {
        GridSample s;
        s.t = static_cast<int64_t>(row[0]);
        s.gyro_p = vec3(row, 1);
        s.gyro_t = vec3(row, 4);
        s.gyro_s = vec3(row, 7);
        s.acc_p = vec3(row, 10);
        s.acc_t = vec3(row, 13);
        s.acc_s = vec3(row, 16);
        s.rel = fk_relations(humanoid, joint_state(names, row, 19));
        samples.push_back(s);
        if (s.t >= window.at(0) && s.t <= window.at(1)) {
            hold.push_back(s);
        }
    }

    // (2) Calibrate on the hold, start at the first sample and step through all of them.
    estimator->initialize(hold, &samples);
    for (const auto& s : samples) {
        write_row(out, s.t, estimator->step(s));
    }
}

// Message CSVs: joints (t, position, velocity) and pelvis, torso, camera IMUs (t, gyro, acc), merged by stamp.
void run_online(std::unique_ptr<BodyAttitudeEstimator> estimator, Humanoid& humanoid, const Meta& meta,
                const std::vector<std::string>& csvs, std::FILE* out)
{
    // (1) Streams, the joint history standing in for the TF buffer, and the on-line estimator.
    const auto names = strings(meta, "joint_names");
    const auto still = numbers(meta, "online_still");
    std::vector<Table> streams;
    for (const auto& csv : csvs) {
        streams.push_back(read_csv(csv));
    }
    SampleBuffer joints(10.0);
    auto lookup = [&](int64_t t, Relations& rel) {
        if (joints.empty() || t > joints.latest()) {
            return Lookup::Wait;
        }
        if (t < joints.oldest()) {
            return Lookup::Missing;
        }
        const Eigen::VectorXd q = joints.interp(t);
        rel = fk_relations(humanoid, joint_state(names, std::vector<double>(q.data(), q.data() + q.size()), 0));
        return Lookup::Found;
    };
    OnlineBodyEstimator online(std::move(estimator), lookup, still.at(0), still.at(1));

    // (2) Feed messages in stamp order (ties: joints, pelvis, torso, camera), as the node receives them.
    std::vector<std::size_t> next(streams.size(), 0);
    while (true) {
        int s = -1;
        for (int i = 0; i < static_cast<int>(streams.size()); ++i) {
            if (next[i] < streams[i].size() && (s < 0 || streams[i][next[i]][0] < streams[s][next[s]][0])) {
                s = i;
            }
        }
        if (s < 0) {
            break;
        }
        const auto& r = streams[s][next[s]++];
        const auto t = static_cast<int64_t>(r[0]);
        if (s == 0) joints.append(t, Eigen::Map<const Eigen::VectorXd>(r.data() + 1, names.size()));
        if (s == 1) online.add_imu(Body::Pelvis, t, vec3(r, 1), vec3(r, 4));
        if (s == 2) online.add_imu(Body::Torso, t, vec3(r, 1), vec3(r, 4));
        if (s == 3) online.add_imu(Body::Camera, t, vec3(r, 1), vec3(r, 4));
        for (const auto& output : online.process().outputs) {
            write_row(out, output.t, output.T_foot_torso);
        }
    }
}

int main(int argc, char** argv) {
    const std::string usage =
        "usage: body_ekf_replay grid <asset_root> <g1.yaml> <meta.txt> <grid.csv> <out.csv>\n"
        "       body_ekf_replay online <asset_root> <g1.yaml> <meta.txt> <joints.csv> <pelvis_imu.csv> "
        "<torso_imu.csv> <camera_imu.csv> <out.csv>\n";
    const std::vector<std::string> args(argv + 1, argv + argc);
    const bool grid = args.size() == 6 && args[0] == "grid";
    const bool online = args.size() == 9 && args[0] == "online";
    if (!grid && !online) {
        std::cerr << usage;
        return 1;
    }
    const Meta meta = read_meta(args[3]);
    auto estimator = make_estimator(meta);
    const auto humanoid = make_humanoid(args[1], args[2]);
    std::FILE* out = std::fopen(args.back().c_str(), "w");
    if (grid) {
        run_grid(std::move(estimator), *humanoid, meta, args[4], out);
    } else {
        run_online(std::move(estimator), *humanoid, meta, {args[4], args[5], args[6], args[7]}, out);
    }
    std::fclose(out);
    return 0;
}
