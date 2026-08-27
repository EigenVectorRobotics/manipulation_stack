#include <g1_pilot/g1_pilot.h>

#include <cmath>
#include <iostream>
#include <string>


using namespace HumanoidPilot;

int main(){

    try{
        RobotConfig cfg;
        cfg.asset_root = G1_ASSET_ROOT;
        cfg.asset_file = std::string(G1_ASSET_ROOT) + "g1_29dof_with_hand_rev_1_0.urdf";
        cfg.config_file = G1_CONFIG_FILE;
        cfg.NUM_DOF = 29;

        auto handle = Humanoid(&cfg);
        const auto names = handle.get_joint_names();

        // Measured start configuration, idle (left) arm deliberately off zero.
        JointState start;
        start.name = names;
        start.position.assign(names.size(), 0.0);
        start.velocity.assign(names.size(), 0.0);
        start.effort.assign(names.size(), 0.0);
        for (std::size_t i = 0; i < names.size(); ++i) {
            if (names[i] == "left_shoulder_pitch_joint") start.position[i] = 0.55;
            if (names[i] == "left_shoulder_roll_joint")  start.position[i] = 0.45;
            if (names[i] == "left_elbow_joint")          start.position[i] = 1.30;
            if (names[i] == "right_elbow_joint")         start.position[i] = 1.10;
        }

        Eigen::MatrixXd target = Eigen::MatrixXd::Identity(4,4);
        target.block<3,1>(0,3) = Eigen::Vector3d(0.30, -0.149, 0.095);

        std::cout << "Arm Initialized" << std::endl;
        std::cout << "target\n" << target << std::endl;

        handle.setLeftArm(false);                       // right arm is the target
        auto traj = handle.planTrajectory(start, target);

        const std::size_t n_joints = traj.position.size();
        const std::size_t n_samples = n_joints ? traj.position.front().size() : 0;

        std::cout << "samples: " << n_samples
                  << "  dt: " << traj.time_step
                  << "  duration: " << (n_samples ? (n_samples - 1) * traj.time_step : 0.0)
                  << std::endl;

        if (n_samples == 0) {
            std::cerr << "planning produced an empty trajectory" << std::endl;
            return -1;
        }

        // Per-arm travel: the idle arm should not move at all.
        double left_travel = 0.0, right_travel = 0.0;
        for (std::size_t j = 0; j < n_joints; ++j) {
            const double d = std::abs(traj.position[j].back() - traj.position[j].front());
            if (traj.name[j].rfind("left_", 0) == 0)  left_travel += d;
            if (traj.name[j].rfind("right_", 0) == 0) right_travel += d;
        }
        std::cout << "idle (left) arm travel:  " << left_travel  << " rad" << std::endl;
        std::cout << "active (right) arm travel: " << right_travel << " rad" << std::endl;

        // Start configuration must match what was handed in, joint for joint.
        double start_drift = 0.0;
        for (std::size_t j = 0; j < n_joints; ++j)
            for (std::size_t i = 0; i < names.size(); ++i)
                if (names[i] == traj.name[j])
                    start_drift = std::max(start_drift,
                        std::abs(traj.position[j].front() - start.position[i]));
        std::cout << "max start-config drift: " << start_drift << " rad" << std::endl;

        constexpr double IDLE_TOL = 1e-3;
        constexpr double START_TOL = 1e-3;
        if (left_travel > IDLE_TOL) {
            std::cerr << "FAIL: idle arm moved " << left_travel
                      << " rad (> " << IDLE_TOL << ")" << std::endl;
            return -1;
        }
        if (start_drift > START_TOL) {
            std::cerr << "FAIL: trajectory does not start at the given configuration ("
                      << start_drift << " rad > " << START_TOL << ")" << std::endl;
            return -1;
        }

        Eigen::VectorXd q(n_joints);
        for (std::size_t k = 0; k < n_samples; ++k) {
            for (std::size_t j = 0; j < n_joints; ++j) q(j) = traj.position[j][k];
            std::cout << handle.forward_kinematics(q).block<3,1>(0,3).transpose() << std::endl;
        }

        std::cout << "PASS" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception during trajectory generation: " << e.what() << std::endl;
        return -1;
    }
}
