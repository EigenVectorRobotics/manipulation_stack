#ifndef G1_PILOT_MOVE_TO_POSE_GRIPPER_ERROR_HISTORY_H
#define G1_PILOT_MOVE_TO_POSE_GRIPPER_ERROR_HISTORY_H

#include <Eigen/Dense>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <limits>
#include <vector>

namespace g1_pilot {

class GripperErrorHistory {
public:
    struct Correction {
        Eigen::Vector3d error = Eigen::Vector3d::Zero();
        std::size_t fits = 0;
        double newest_age = std::numeric_limits<double>::infinity();
    };

    explicit GripperErrorHistory(std::size_t capacity = 5) : capacity_(capacity) {}

    bool add(double stamp, const Eigen::Vector3d & error)
    {
        if (!samples_.empty() && stamp <= samples_.back().stamp) return false;
        samples_.push_back({stamp, error});
        while (samples_.size() > capacity_) samples_.pop_front();
        return true;
    }

    double newest_stamp() const
    {
        return samples_.empty() ? -std::numeric_limits<double>::infinity() : samples_.back().stamp;
    }

    Correction correction(double now, double max_age, double hold_time) const
    {
        // A fresh fit is used alone; otherwise the per-axis median of the fits held within hold_time rejects a bad one
        Correction out;
        if (samples_.empty()) return out;
        out.newest_age = now - samples_.back().stamp;
        if (out.newest_age <= max_age) {
            out.error = samples_.back().error;
            out.fits = 1;
            return out;
        }
        std::vector<Eigen::Vector3d> held;
        for (const auto & s : samples_) {
            if (now - s.stamp <= hold_time) held.push_back(s.error);
        }
        out.fits = held.size();
        for (int axis = 0; axis < 3 && !held.empty(); ++axis) {
            std::vector<double> values;
            for (const auto & e : held) values.push_back(e(axis));
            std::sort(values.begin(), values.end());
            const std::size_t mid = values.size() / 2;
            out.error(axis) = values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
        }
        return out;
    }

private:
    struct Sample {
        double stamp;
        Eigen::Vector3d error;
    };
    std::size_t capacity_;
    std::deque<Sample> samples_;
};

} // namespace g1_pilot

#endif // G1_PILOT_MOVE_TO_POSE_GRIPPER_ERROR_HISTORY_H
