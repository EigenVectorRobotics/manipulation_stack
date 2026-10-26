import numpy as np

CHI2_3DOF_999 = 16.27


def tangent_basis(normal: np.ndarray) -> np.ndarray:
    reference = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)
    return np.column_stack([u, np.cross(normal, u)])


class TargetKF:
    def __init__(self, q_pos: float = 3e-3, sig_lat: float = 4e-3, sig_depth: float = 1.5e-2,
                 sig_kin: float = 2e-3, q_ang: float = np.radians(0.5), sig_ang: float = np.radians(3.0),
                 gate: float = CHI2_3DOF_999):
        self.q_pos, self.sig_lat, self.sig_depth, self.sig_kin = q_pos, sig_lat, sig_depth, sig_kin
        self.q_ang, self.sig_ang, self.gate = q_ang, sig_ang, gate
        self.reset()

    def reset(self) -> None:
        self.p = None
        self.P = None
        self.n = None
        self.Pn = None
        self.t = None

    @property
    def initialized(self) -> bool:
        return self.p is not None

    def measurement_covariance(self, R_frame_optical: np.ndarray) -> np.ndarray:
        camera = np.diag([self.sig_lat ** 2, self.sig_lat ** 2, self.sig_depth ** 2])
        return R_frame_optical @ camera @ R_frame_optical.T + self.sig_kin ** 2 * np.eye(3)

    def update(self, t: float, z_p: np.ndarray, z_n: np.ndarray,
               R_frame_optical: np.ndarray) -> tuple[float, bool]:
        # (1) Initialise from the first measurement.
        R_meas = self.measurement_covariance(R_frame_optical)
        z_n = z_n / np.linalg.norm(z_n)
        if self.p is None:
            self.p, self.P = np.array(z_p, dtype=float), R_meas
            self.n, self.Pn = z_n, np.eye(2) * self.sig_ang ** 2
            self.t = t
            return 0.0, True

        # (2) Random-walk prediction and NIS gate on the position innovation.
        dt = max(t - self.t, 0.0)
        self.t = t
        self.P = self.P + np.eye(3) * self.q_pos ** 2 * dt
        self.Pn = self.Pn + np.eye(2) * self.q_ang ** 2 * dt
        y = z_p - self.p
        S_inv = np.linalg.inv(self.P + R_meas)
        nis = float(y @ S_inv @ y)
        if nis > self.gate:
            return nis, False

        # (3) Position update, then the normal as a small-angle error in the estimate's tangent plane.
        K = self.P @ S_inv
        self.p = self.p + K @ y
        self.P = (np.eye(3) - K) @ self.P
        basis = tangent_basis(self.n)
        K_n = self.Pn @ np.linalg.inv(self.Pn + np.eye(2) * self.sig_ang ** 2)
        self.n = self.n + basis @ (K_n @ (basis.T @ z_n))
        self.n /= np.linalg.norm(self.n)
        self.Pn = (np.eye(2) - K_n) @ self.Pn
        return nis, True
