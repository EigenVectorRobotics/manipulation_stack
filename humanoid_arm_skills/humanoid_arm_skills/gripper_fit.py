import struct

import numpy as np
import warp as wp

NEAR = 0.004
FIRST_GATE = 0.015
SHRINK = 0.85
BINS = 2048
ROI_PAD = 30
# Coarse levels 8 / 4 / 2 while the gate is above 10 / 7 / 5 mm, then full resolution
LEVELS = ((0.010, 8), (0.007, 4), (0.005, 2))
PASSES = 19

NEAR_C = wp.constant(NEAR)
BINS_C = wp.constant(BINS)


def read_stl(path: str) -> np.ndarray:
    data = open(path, 'rb').read()
    count = struct.unpack('<I', data[80:84])[0]
    record = np.dtype([('normal', '<3f4'), ('vertices', '<9f4'), ('attribute', '<u2')])
    return np.frombuffer(data[84:84 + count * 50], dtype=record)['vertices'].reshape(-1, 3, 3).astype(float)


def schedule(first_gate: float = FIRST_GATE, passes: int = PASSES) -> list:
    # The gate shrinks the same way every frame, so each pass's level is known without reading the GPU
    plan, gate = [], first_gate
    for _ in range(passes):
        plan.append(next((level for above, level in LEVELS if gate > above), 1))
        gate = max(NEAR, gate * SHRINK)
    return plan


@wp.kernel
def observed_kernel(depth: wp.array2d(dtype=wp.float32), u0: int, v0: int, fx: float, fy: float, cx: float, cy: float,
                    base_from_camera: wp.mat33, camera_t: wp.vec3, low: wp.vec3, high: wp.vec3,
                    points: wp.array2d(dtype=wp.vec3), valid: wp.array2d(dtype=wp.int32), count: wp.array(dtype=wp.int32)):
    i, j = wp.tid()
    z = depth[i, j]
    valid[i, j] = 0
    if z <= 0.0:
        return
    p = wp.vec3((float(u0 + j) + 0.5 - cx) * z / fx, (float(v0 + i) + 0.5 - cy) * z / fy, z)
    b = base_from_camera * (p - camera_t)
    if b[0] < low[0] or b[1] < low[1] or b[2] < low[2] or b[0] > high[0] or b[1] > high[1] or b[2] > high[2]:
        return
    points[i, j] = p
    valid[i, j] = 1
    wp.atomic_add(count, 0, 1)


@wp.kernel
def match_kernel(mesh: wp.uint64, points: wp.array2d(dtype=wp.vec3), valid: wp.array2d(dtype=wp.int32), u0: int, v0: int, level: int,
                 fx: float, fy: float, cx: float, cy: float, z_min: float, base_from_camera: wp.mat33, camera_from_base: wp.mat33,
                 camera_t: wp.vec3, state: wp.array(dtype=wp.float32), hist: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32)):
    # One ray every `level` pixels, matched to the nearest observed point in a window searched every `level` pixels
    i0, j0 = wp.tid()
    i = i0 * level
    j = j0 * level
    h = points.shape[0]
    w = points.shape[1]
    if i >= h or j >= w:
        return
    gate = state[3]
    win = int(wp.ceil(gate * fx / (z_min * float(level))))
    t = camera_t + wp.vec3(state[0], state[1], state[2])
    origin = base_from_camera * (-t)
    d = wp.normalize(base_from_camera * wp.vec3((float(u0 + j) + 0.5 - cx) / fx, (float(v0 + i) + 0.5 - cy) / fy, 1.0))
    q = wp.mesh_query_ray(mesh, origin, d, 2.0)
    if not q.result:
        return
    rendered = camera_from_base * (origin + d * q.t) + t
    wp.atomic_add(count, 2, 1)
    best = gate * gate
    match = wp.vec3(0.0)
    found = int(0)
    for di in range(-win, win + 1):
        for dj in range(-win, win + 1):
            ii = i + di * level
            jj = j + dj * level
            if ii >= 0 and ii < h and jj >= 0 and jj < w:
                if valid[ii, jj] == 1:
                    e = points[ii, jj] - rendered
                    if wp.dot(e, e) < best:
                        best = wp.dot(e, e)
                        match = points[ii, jj]
                        found = 1
    if found == 0:
        return
    e = match - rendered
    wp.atomic_add(count, 0, 1)
    if wp.length(e) < NEAR_C:
        wp.atomic_add(count, 1, 1)
    for k in range(3):
        b = wp.clamp(int((e[k] / gate * 0.5 + 0.5) * float(BINS_C)), 0, BINS_C - 1)
        wp.atomic_add(hist, k * BINS_C + b, 1)


@wp.kernel
def median_step_kernel(hist: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32), state: wp.array(dtype=wp.float32)):
    # One thread per axis: the median residual from its histogram, interpolated inside the median bin
    k = wp.tid()
    n = count[0]
    gate = state[3]
    step = float(0.0)
    if n >= 50:
        half = float(n) * 0.5
        below = float(0.0)
        for b in range(BINS_C):
            hb = float(hist[k * BINS_C + b])
            if below + hb >= half:
                step = ((float(b) + (half - below) / wp.max(hb, 1.0)) / float(BINS_C) - 0.5) * 2.0 * gate
                break
            below = below + hb
    state[k] = state[k] + step


@wp.kernel
def gate_kernel(state: wp.array(dtype=wp.float32), shrink: float):
    state[3] = wp.max(NEAR_C, state[3] * shrink)


class GripperFit:
    def __init__(self, mesh_path: str):
        triangles = read_stl(mesh_path)
        vertices, index = np.unique(np.round(triangles.reshape(-1, 3), 9), axis=0, return_inverse=True)
        self.mesh = wp.Mesh(points=wp.array(vertices.astype(np.float32), dtype=wp.vec3),
                            indices=wp.array(index.reshape(-1).astype(np.int32), dtype=wp.int32))
        self.low, self.high = vertices.min(0), vertices.max(0)
        self.plan = schedule()
        self.hist = wp.zeros(3 * BINS, dtype=wp.int32)
        self.count = wp.zeros(3, dtype=wp.int32)
        self.observed = wp.zeros(1, dtype=wp.int32)

    def roi(self, k: np.ndarray, camera_from_base: np.ndarray, width: int, height: int):
        # The closed gripper's box at the encoder pose, projected and padded for the offset
        corners = np.array([[x, y, z] for x in (self.low[0], self.high[0]) for y in (self.low[1], self.high[1])
                            for z in (self.low[2], self.high[2])]) @ camera_from_base[:3, :3].T + camera_from_base[:3, 3]
        if np.any(corners[:, 2] <= 0.05):
            return None
        u = k[0, 0] * corners[:, 0] / corners[:, 2] + k[0, 2]
        v = k[1, 1] * corners[:, 1] / corners[:, 2] + k[1, 2]
        u0, u1 = max(int(u.min()) - ROI_PAD, 0), min(int(u.max()) + ROI_PAD, width)
        v0, v1 = max(int(v.min()) - ROI_PAD, 0), min(int(v.max()) + ROI_PAD, height)
        return (u0, v0, u1, v1) if u1 > u0 and v1 > v0 else None

    def fit(self, depth_mm: np.ndarray, k: np.ndarray, camera_from_base: np.ndarray, start: np.ndarray):
        # (1) Upload the region of interest and back-project the points inside the gripper's box
        height, width = depth_mm.shape
        roi = self.roi(k, camera_from_base, width, height)
        if roi is None:
            return None
        u0, v0, u1, v1 = roi
        patch = depth_mm[v0:v1, u0:u1]
        nonzero = patch[patch > 0]
        if len(nonzero) == 0:
            return None
        z_min = max(float(nonzero.min()) / 1000.0, 0.1)
        h, w = patch.shape
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        rotation, t = camera_from_base[:3, :3], camera_from_base[:3, 3]
        base_from_camera, camera_from_base_r = wp.mat33(*rotation.T.ravel()), wp.mat33(*rotation.ravel())
        depth = wp.array(np.ascontiguousarray(patch, dtype=np.float32) / 1000.0, dtype=wp.float32)
        points, valid = wp.empty((h, w), dtype=wp.vec3), wp.empty((h, w), dtype=wp.int32)
        self.observed.zero_()
        wp.launch(observed_kernel, dim=(h, w), inputs=[depth, u0, v0, fx, fy, cx, cy, base_from_camera, wp.vec3(*t),
                                                     wp.vec3(*self.low), wp.vec3(*self.high), points, valid, self.observed])

        # (2) All passes are queued on the GPU; the offset and gate stay there until the final read
        state = wp.array(np.r_[start, FIRST_GATE].astype(np.float32), dtype=wp.float32)
        for level in self.plan:
            self.hist.zero_()
            self.count.zero_()
            wp.launch(match_kernel, dim=((h + level - 1) // level, (w + level - 1) // level),
                      inputs=[self.mesh.id, points, valid, u0, v0, level, fx, fy, cx, cy, z_min, base_from_camera,
                              camera_from_base_r, wp.vec3(*t), state, self.hist, self.count])
            wp.launch(median_step_kernel, dim=3, inputs=[self.hist, self.count, state])
            wp.launch(gate_kernel, dim=1, inputs=[state, SHRINK])

        # (3) The last pass is at full resolution, so its counts give the inlier share
        offset, count = state.numpy()[:3].astype(float), self.count.numpy()
        return dict(offset=offset, inliers=float(count[1]) / max(int(count[2]), 1), model_rays=int(count[2]),
                    points=int(self.observed.numpy()[0]))
