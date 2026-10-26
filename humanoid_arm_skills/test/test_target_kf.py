import numpy as np
import pytest

from humanoid_arm_skills.target_kf import CHI2_3DOF_999, TargetKF

RATE = 15.0
TRUTH = np.array([0.45, 0.12, 0.85])
NORMAL = np.array([-1.0, 0.0, 0.0])


def camera_rotation():
    # Optical frame looking forward and down at the target: z along the ray, x to the right.
    ray = np.array([1.0, -0.2, -0.6])
    z = ray / np.linalg.norm(ray)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def measurements(rng, count, C):
    noise = rng.normal(0.0, 1.0, (count, 3)) * [4e-3, 4e-3, 1.5e-2]
    return TRUTH + noise @ C.T + rng.normal(0.0, 2e-3, (count, 3))


def noisy_normal(rng, sigma_deg=3.0):
    n = NORMAL + rng.normal(0.0, np.radians(sigma_deg), 3)
    return n / np.linalg.norm(n)


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def converged_filter(rng, C, count=150):
    kf = TargetKF()
    for k, z in enumerate(measurements(rng, count, C)):
        kf.update(k / RATE, z, noisy_normal(rng), C)
    return kf


def test_static_target_converges_and_beats_ema():
    rng = np.random.default_rng(0)
    C = camera_rotation()
    z = measurements(rng, 450, C)
    kf, ema = TargetKF(), z[0].copy()
    kf_track, ema_track, nees = [], [], []
    for k in range(len(z)):
        kf.update(k / RATE, z[k], NORMAL, C)
        ema += 0.2 * (z[k] - ema)
        kf_track.append(kf.p.copy())
        ema_track.append(ema.copy())
        nees.append((kf.p - TRUTH) @ np.linalg.inv(kf.P) @ (kf.p - TRUTH))
    kf_error = np.linalg.norm(np.array(kf_track[30:]) - TRUTH, axis=1)
    ema_error = np.linalg.norm(np.array(ema_track[30:]) - TRUTH, axis=1)
    # The random walk holds the depth sigma near 3.4 mm, about 0.6x the EMA error, so bounds are set from P.
    assert np.mean(nees[30:]) < 3.0 and nees[-1] < CHI2_3DOF_999
    assert np.sqrt(np.mean(kf_error ** 2)) < 6e-3
    assert np.sqrt(np.mean(kf_error ** 2)) < 0.75 * np.sqrt(np.mean(ema_error ** 2))
    assert np.trace(kf.P) < np.trace(kf.measurement_covariance(C))


def test_measurement_covariance_puts_depth_noise_along_the_ray():
    C = camera_rotation()
    R = TargetKF().measurement_covariance(C)
    ray, lateral = C[:, 2], C[:, 0]
    assert ray @ R @ ray == pytest.approx(1.5e-2 ** 2 + 2e-3 ** 2)
    assert lateral @ R @ lateral == pytest.approx(4e-3 ** 2 + 2e-3 ** 2)


def test_gate_rejects_one_frame_outlier():
    rng = np.random.default_rng(1)
    C = camera_rotation()
    kf = converged_filter(rng, C)
    p, P = kf.p.copy(), kf.P.copy()
    nis, accepted = kf.update(150 / RATE, TRUTH + 0.04 * C[:, 0], NORMAL, C)
    assert not accepted and nis > kf.gate
    np.testing.assert_array_equal(kf.p, p)
    assert np.trace(kf.P) > np.trace(P)
    _, accepted = kf.update(151 / RATE, TRUTH, NORMAL, C)
    assert accepted


def test_normal_stays_unit_and_converges():
    rng = np.random.default_rng(2)
    C = camera_rotation()
    kf = TargetKF()
    initial = np.array([-0.9, 0.3, 0.3])
    kf.update(0.0, TRUTH, initial / np.linalg.norm(initial), C)
    for k in range(1, 300):
        kf.update(k / RATE, TRUTH, noisy_normal(rng), C)
        assert np.linalg.norm(kf.n) == pytest.approx(1.0)
    assert angle_deg(kf.n, NORMAL) < 1.0


def test_reset_reinitialises_from_next_measurement():
    rng = np.random.default_rng(3)
    C = camera_rotation()
    kf = converged_filter(rng, C)
    assert kf.initialized
    kf.reset()
    assert not kf.initialized and kf.P is None and kf.n is None
    moved = TRUTH + np.array([0.2, 0.0, 0.0])
    nis, accepted = kf.update(20.0, moved, 2.0 * NORMAL, C)
    assert accepted and nis == 0.0
    np.testing.assert_array_equal(kf.p, moved)
    np.testing.assert_allclose(kf.P, kf.measurement_covariance(C))
    np.testing.assert_allclose(kf.n, NORMAL)
