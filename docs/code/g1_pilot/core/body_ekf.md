# g1_pilot/src/core/body_ekf

The body attitude estimator behind the `body_ekf` node. [Body attitude EKF](../../../g1_pilot/src/core/body_ekf.md) has the design.

- **`body_ekf.h`**: Declares `Body2EKF`, `BodyAttitudeEstimator`, `OnlineBodyEstimator`, `SampleBuffer`, `StillPeriodCollector` and the config and sample structs.
- **`body_ekf.cpp`**: Implements the 15-state filter, the still-period calibration and the IMU resampling on a fixed grid.
