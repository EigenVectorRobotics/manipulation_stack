#include <g1_pilot/g1_pilot.h>

#include <iostream>


using namespace HumanoidPilot;
int main(){

    try{
        Eigen::Matrix4d target = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d initial = Eigen::Matrix4d::Identity();
        target.block<3,1>(0,3) = Eigen::Vector3d(0.3,-0.149,0.095);
        initial.block<3,1>(0,3) = Eigen::Vector3d(0.21, -0.149, 0.095);

        auto handle = Humanoid();

        std::cout <<"Arm Initialized" <<std::endl;
        std::cout <<"Initial" << initial <<std::endl;
        std::cout <<"target" << target  <<std::endl;

        Eigen::MatrixXd goal_pose = target;
        Eigen::MatrixXd start_pose = initial;

        auto traj = handle.planTrajectory(&goal_pose, &start_pose);

        const std::size_t n_joints = traj.position.size();
        const std::size_t n_samples = n_joints ? traj.position.front().size() : 0;

        std::cout << "samples: " << n_samples
                  << "  dt: " << traj.time_step
                  << "  duration: " << (n_samples ? (n_samples - 1) * traj.time_step : 0.0)
                  << std::endl;

        for (std::size_t k = 0; k < n_samples; ++k) {
            Eigen::VectorXd q(n_joints);
            for (std::size_t j = 0; j < n_joints; ++j) q(j) = traj.position[j][k];
            std::cout << handle.forward_kinematics(q).block<3,1>(0,3).transpose() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception during IK solving: " << e.what() << std::endl;
        return -1;
    }
}