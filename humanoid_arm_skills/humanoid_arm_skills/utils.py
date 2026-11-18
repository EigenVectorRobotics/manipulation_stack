"""Local replacements for the few vector_utils.transform_utils helpers used here."""

import math

import numpy as np

from geometry_msgs.msg import Pose, Quaternion


def normalize_angle(angle):
    """Normalize angle (rad) to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def quaternion_angular_distance(q1: Quaternion, q2: Quaternion) -> float:
    """Angular distance in radians between two quaternions, in [0, pi]."""
    dot = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w
    return 2.0 * np.arccos(np.clip(abs(dot), 0.0, 1.0))


def pose_distance(p1: Pose, p2: Pose) -> tuple[float, float]:
    """Position (metres) and orientation (degrees) distance between two poses."""
    dp = math.sqrt(
        (p1.position.x - p2.position.x) ** 2
        + (p1.position.y - p2.position.y) ** 2
        + (p1.position.z - p2.position.z) ** 2)
    d_ori_deg = math.degrees(quaternion_angular_distance(
        p1.orientation, p2.orientation))
    return dp, d_ori_deg


def average_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Average quaternion via the eigenvector method (Markley et al.).

    Input/output use [x, y, z, w] order. Handles q/-q sign ambiguity.
    """
    Q = np.asarray(quaternions, dtype=np.float64)
    if Q.ndim == 1:
        return Q / np.linalg.norm(Q)

    for i in range(1, len(Q)):
        if np.dot(Q[i], Q[0]) < 0.0:
            Q[i] = -Q[i]

    M = Q.T @ Q
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    avg = eigenvectors[:, eigenvalues.argmax()]
    return avg / np.linalg.norm(avg)


def quaternion_ema(previous: np.ndarray, current: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate unit quaternions along their shortest arc by the new-sample weight."""
    dot = float(np.dot(previous, current))
    if dot < 0.0:
        current = -current
        dot = -dot
    dot = float(np.clip(dot, 0.0, 1.0))
    if dot > 0.9995:
        result = (1.0 - alpha) * previous + alpha * current
    else:
        angle = np.arccos(dot)
        result = (np.sin((1.0 - alpha) * angle) * previous
                  + np.sin(alpha * angle) * current) / np.sin(angle)
    return result / np.linalg.norm(result)


def surface_normal(quaternion: np.ndarray) -> np.ndarray:
    """Unit local X axis of a quaternion [x, y, z, w]."""
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)])


def normal_quaternion(normal: np.ndarray) -> np.ndarray:
    """Quaternion with local X along the normal and local Z closest to the reference frame's +Z."""
    normal_x, normal_y, normal_z = normal
    horizontal = np.hypot(normal_x, normal_y)
    yaw = np.arctan2(normal_y, normal_x) if horizontal > 1e-12 else 0.0
    pitch = np.arctan2(-normal_z, horizontal)
    sy, cy = np.sin(yaw / 2.0), np.cos(yaw / 2.0)
    sp, cp = np.sin(pitch / 2.0), np.cos(pitch / 2.0)
    return np.array([-sy * sp, cy * sp, sy * cp, cy * cp])


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Rotation matrix of a quaternion [x, y, z, w]."""
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]])


def matrix_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] of a rotation matrix."""
    # The branch with the largest pivot, so near 180 deg nothing divides by a tiny w
    if np.trace(rotation) > 0.0:
        w = np.sqrt(1.0 + np.trace(rotation)) / 2.0
        x = (rotation[2, 1] - rotation[1, 2]) / (4.0 * w)
        y = (rotation[0, 2] - rotation[2, 0]) / (4.0 * w)
        z = (rotation[1, 0] - rotation[0, 1]) / (4.0 * w)
        return np.array([x, y, z, w]) / np.linalg.norm([x, y, z, w])
    k = int(np.argmax(np.diag(rotation)))
    i, j = (k + 1) % 3, (k + 2) % 3
    q = np.zeros(4)
    q[k] = np.sqrt(max(0.0, 1.0 + rotation[k, k] - rotation[i, i] - rotation[j, j])) / 2.0
    q[i] = (rotation[i, k] + rotation[k, i]) / (4.0 * q[k])
    q[j] = (rotation[j, k] + rotation[k, j]) / (4.0 * q[k])
    q[3] = (rotation[j, i] - rotation[i, j]) / (4.0 * q[k])
    return q / np.linalg.norm(q)


def matrix_rpy(rotation: np.ndarray) -> np.ndarray:
    """Roll, pitch and yaw [rad] of a rotation matrix, as fixed-axis x, y, z rotations."""
    return np.array([np.arctan2(rotation[2, 1], rotation[2, 2]),
                     np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)),
                     np.arctan2(rotation[1, 0], rotation[0, 0])])


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """Rotation matrix of roll, pitch and yaw [rad], the inverse of matrix_rpy."""
    (cr, cp, cy), (sr, sp, sy) = np.cos(rpy), np.sin(rpy)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def align_surface_z(quaternion: np.ndarray) -> np.ndarray:
    """Preserve local X and align local Z closest to the reference frame's +Z."""
    return normal_quaternion(surface_normal(quaternion))


def approach_pose(pose: Pose) -> Pose:
    """Reverse the surface normal about local Z while preserving its full tilt."""
    q = pose.orientation
    quaternion = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError('Surface orientation must be a finite, nonzero quaternion')
    x, y, z, w = quaternion / norm
    result = Pose()
    result.position.x = pose.position.x
    result.position.y = pose.position.y
    result.position.z = pose.position.z
    result.orientation = Quaternion(x=float(y), y=float(-x), z=float(w), w=float(-z))
    return result


def upright_pose(pose: Pose, flip: bool = False) -> Pose:
    """Return a pose with the same position but a yaw-only (Z-up) orientation.

    The yaw is taken from the projection of the input pose's +X axis onto the
    XY plane. If ``flip`` is True, yaw is rotated by 180°.
    """
    x = pose.orientation.x
    y = pose.orientation.y
    z = pose.orientation.z
    w = pose.orientation.w
    if (x * x + y * y + z * z + w * w) == 0.0:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    forward_x = 1.0 - 2.0 * (y * y + z * z)
    forward_y = 2.0 * (x * y + w * z)
    yaw = float(np.arctan2(forward_y, forward_x))
    if flip:
        yaw += np.pi

    result = Pose()
    result.position = pose.position
    result.orientation = Quaternion(
        x=0.0, y=0.0,
        z=float(np.sin(yaw / 2)),
        w=float(np.cos(yaw / 2)),
    )
    return result
