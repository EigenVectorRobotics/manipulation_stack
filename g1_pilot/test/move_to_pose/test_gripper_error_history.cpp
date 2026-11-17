#include "move_to_pose/gripper_error_history.h"

#include <gtest/gtest.h>

using g1_pilot::GripperErrorHistory;

const Eigen::Vector3d MM(double x, double y, double z) { return Eigen::Vector3d(x, y, z) * 1e-3; }

TEST(GripperErrorHistory, EmptyGivesNoFits) {
    GripperErrorHistory history;
    EXPECT_EQ(history.correction(10.0, 0.5, 60.0).fits, 0u);
}

TEST(GripperErrorHistory, FreshFitIsUsedAlone) {
    GripperErrorHistory history;
    history.add(1.0, MM(10, -2, 0));
    history.add(2.0, MM(12, 1, 1));
    const auto c = history.correction(2.3, 0.5, 60.0);
    EXPECT_EQ(c.fits, 1u);
    EXPECT_TRUE(c.error.isApprox(MM(12, 1, 1)));
    EXPECT_NEAR(c.newest_age, 0.3, 1e-9);
}

TEST(GripperErrorHistory, StaleFitsGiveTheMedianAndRejectAnOutlier) {
    // Fits from bag full_test_20261108_212948 in the pelvis, with one 40 mm bad fit that passed the gate
    GripperErrorHistory history;
    history.add(40.45, MM(10.4, -4.7, 0.5));
    history.add(40.62, MM(8.6, -2.8, -1.1));
    history.add(40.78, MM(29.0, -13.8, -20.0));
    history.add(40.95, MM(12.4, -0.8, 1.5));
    const auto c = history.correction(50.0, 0.5, 60.0);
    EXPECT_EQ(c.fits, 4u);
    EXPECT_TRUE(c.error.isApprox(MM(11.4, -3.75, 0.5 * (0.5 - 1.1)), 1e-9));
}

TEST(GripperErrorHistory, FitsOlderThanTheHoldTimeAreDropped) {
    GripperErrorHistory history;
    history.add(0.0, MM(40, 0, 0));
    history.add(100.0, MM(11, -1, 0));
    const auto c = history.correction(120.0, 0.5, 60.0);
    EXPECT_EQ(c.fits, 1u);
    EXPECT_TRUE(c.error.isApprox(MM(11, -1, 0)));
    EXPECT_EQ(history.correction(200.0, 0.5, 60.0).fits, 0u);
}

TEST(GripperErrorHistory, KeepsOnlyNewerFitsUpToCapacity) {
    GripperErrorHistory history(3);
    EXPECT_TRUE(history.add(1.0, MM(1, 0, 0)));
    EXPECT_FALSE(history.add(1.0, MM(9, 0, 0)));
    EXPECT_FALSE(history.add(0.5, MM(9, 0, 0)));
    for (double t : {2.0, 3.0, 4.0}) history.add(t, MM(t, 0, 0));
    const auto c = history.correction(10.0, 0.5, 60.0);
    EXPECT_EQ(c.fits, 3u);
    EXPECT_NEAR(c.error(0), 3e-3, 1e-12);
    EXPECT_DOUBLE_EQ(history.newest_stamp(), 4.0);
}
