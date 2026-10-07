#include "core/arm_mp/joint_space_planner.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>

using HumanoidPilot::JointSpacePlanner;
using HumanoidPilot::JointState;

namespace {

using Clock = std::chrono::steady_clock;
constexpr int ITERATIONS = 20000;

JointState benchmarkState(std::size_t joints) {
    JointState state;
    for (std::size_t j = 0; j < joints; ++j) {
        state.name.push_back("joint_" + std::to_string(j));
        state.position.push_back(0.1 * j);
        state.velocity.push_back((j % 2 == 0 ? 1.0 : -1.0) * (1.0 - 0.02 * j));
    }
    return state;
}

void benchmark(std::size_t joints, double dt) {
    // (1) Warm allocation paths before collecting per-call generation latency.
    const auto state = benchmarkState(joints);
    for (int i = 0; i < 200; ++i) (void) JointSpacePlanner::brake(state, dt);
    std::vector<double> times;
    times.reserve(ITERATIONS);
    double checksum = 0.0;
    std::size_t samples = 0;

    // (2) Include allocation and sampling; exclude trajectory destruction.
    for (int i = 0; i < ITERATIONS; ++i) {
        const auto begin = Clock::now();
        const auto trajectory = JointSpacePlanner::brake(state, dt);
        const auto end = Clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
        samples = trajectory.position[0].size();
        checksum += trajectory.position[i % joints].back();
    }

    // (3) Report latency separately from the duration of the generated motion.
    std::sort(times.begin(), times.end());
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << joints << ',' << dt << ',' << samples << ',' << (samples - 1) * dt << ','
              << mean << ',' << times[times.size() / 2] << ',' << times[times.size() * 95 / 100]
              << ',' << times.back() << ',' << checksum << '\n';
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "joints,dt_s,samples,motion_duration_s,mean_us,median_us,p95_us,max_us,checksum\n";
    benchmark(7, 0.01);
    benchmark(14, 0.01);
    benchmark(29, 0.01);
    benchmark(29, 0.001);
}
