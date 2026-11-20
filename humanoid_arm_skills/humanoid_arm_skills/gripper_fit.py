import os
import struct
import sys

import kornia
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
import warp as wp

from humanoid_arm_skills.utils import matrix_rpy, rpy_matrix

FOUNDATIONPOSE_DIR = os.environ.get('FOUNDATIONPOSE_DIR', os.path.expanduser('~/FoundationPose'))
sys.path.insert(0, FOUNDATIONPOSE_DIR)
from estimater import FoundationPose, PoseRefinePredictor  # noqa: E402

NEAR = 0.004
FIRST_GATE = 0.015
SHRINK = 0.85
BINS = 2048
ROI_PAD = 30
# Coarse levels 8 / 4 / 2 while the gate is above 10 / 7 / 5 mm, then full resolution
LEVELS = ((0.010, 8), (0.007, 4), (0.005, 2))
PASSES = 19
# The rotation search: short refits from the full fit's offset, stepping coarse to fine about each searched axis
ROTATION_AXES = (1,)
ROTATION_STEPS_DEG = (4.0, 2.0, 1.0)
ROTATION_GATE = 0.006
ROTATION_PASSES = 6
MAX_ROTATION_DEG = 8.0
MAX_TRANSLATION = 0.03
# FoundationPose: a ray counts when its depth is not clearly in front of the gripper, and the mesh must be half in view
OCCLUDED = 0.01
MIN_IN_VIEW = 0.5
REFINE_ITERATIONS = 2

NEAR_C = wp.constant(NEAR)
ROTATION_GATE_C = wp.constant(ROTATION_GATE)
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
                    rotations: wp.array(dtype=wp.mat33), crop_t: wp.vec3, low: wp.vec3, high: wp.vec3,
                    points: wp.array3d(dtype=wp.vec3), valid: wp.array3d(dtype=wp.int32), count: wp.array(dtype=wp.int32)):
    # Candidate c keeps the points inside the gripper box turned to its own rotation
    c, i, j = wp.tid()
    z = depth[i, j]
    valid[c, i, j] = 0
    if z <= 0.0:
        return
    p = wp.vec3((float(u0 + j) + 0.5 - cx) * z / fx, (float(v0 + i) + 0.5 - cy) * z / fy, z)
    b = wp.transpose(rotations[c]) * (p - crop_t)
    if b[0] < low[0] or b[1] < low[1] or b[2] < low[2] or b[0] > high[0] or b[1] > high[1] or b[2] > high[2]:
        return
    points[c, i, j] = p
    valid[c, i, j] = 1
    wp.atomic_add(count, c, 1)


@wp.kernel
def match_kernel(mesh: wp.uint64, points: wp.array3d(dtype=wp.vec3), valid: wp.array3d(dtype=wp.int32), u0: int, v0: int, level: int,
                 fx: float, fy: float, cx: float, cy: float, z_min: float, rotations: wp.array(dtype=wp.mat33),
                 camera_t: wp.vec3, state: wp.array2d(dtype=wp.float32), hist: wp.array2d(dtype=wp.int32),
                 count: wp.array2d(dtype=wp.int32)):
    # Candidate c: one ray every `level` pixels, matched to the nearest observed point in a window searched as finely
    c, i0, j0 = wp.tid()
    i = i0 * level
    j = j0 * level
    h = points.shape[1]
    w = points.shape[2]
    if i >= h or j >= w:
        return
    gate = state[c, 3]
    win = int(wp.ceil(gate * fx / (z_min * float(level))))
    camera_from_candidate = rotations[c]
    candidate_from_camera = wp.transpose(camera_from_candidate)
    t = camera_t + wp.vec3(state[c, 0], state[c, 1], state[c, 2])
    origin = candidate_from_camera * (-t)
    d = wp.normalize(candidate_from_camera * wp.vec3((float(u0 + j) + 0.5 - cx) / fx, (float(v0 + i) + 0.5 - cy) / fy, 1.0))
    q = wp.mesh_query_ray(mesh, origin, d, 2.0)
    if not q.result:
        return
    rendered = camera_from_candidate * (origin + d * q.t) + t
    wp.atomic_add(count, c, 2, 1)
    best = gate * gate
    match = wp.vec3(0.0)
    found = int(0)
    for di in range(-win, win + 1):
        for dj in range(-win, win + 1):
            ii = i + di * level
            jj = j + dj * level
            if ii >= 0 and ii < h and jj >= 0 and jj < w:
                if valid[c, ii, jj] == 1:
                    e = points[c, ii, jj] - rendered
                    if wp.dot(e, e) < best:
                        best = wp.dot(e, e)
                        match = points[c, ii, jj]
                        found = 1
    if found == 0:
        return
    e = match - rendered
    wp.atomic_add(count, c, 0, 1)
    if wp.length(e) < NEAR_C:
        wp.atomic_add(count, c, 1, 1)
    for k in range(3):
        b = wp.clamp(int((e[k] / gate * 0.5 + 0.5) * float(BINS_C)), 0, BINS_C - 1)
        wp.atomic_add(hist, c, k * BINS_C + b, 1)


@wp.kernel
def median_step_kernel(hist: wp.array2d(dtype=wp.int32), count: wp.array2d(dtype=wp.int32), state: wp.array2d(dtype=wp.float32)):
    # One thread per candidate and axis: the median residual from its histogram, interpolated inside the median bin
    c, k = wp.tid()
    n = count[c, 0]
    gate = state[c, 3]
    step = float(0.0)
    if n >= 50:
        half = float(n) * 0.5
        below = float(0.0)
        for b in range(BINS_C):
            hb = float(hist[c, k * BINS_C + b])
            if below + hb >= half:
                step = ((float(b) + (half - below) / wp.max(hb, 1.0)) / float(BINS_C) - 0.5) * 2.0 * gate
                break
            below = below + hb
    state[c, k] = state[c, k] + step


@wp.kernel
def gate_kernel(state: wp.array2d(dtype=wp.float32), shrink: float):
    c = wp.tid()
    state[c, 3] = wp.max(NEAR_C, state[c, 3] * shrink)


@wp.func
def rpy_rotation(rpy: wp.vec3) -> wp.mat33:
    cr, sr = wp.cos(rpy[0]), wp.sin(rpy[0])
    cp, sp = wp.cos(rpy[1]), wp.sin(rpy[1])
    cy, sy = wp.cos(rpy[2]), wp.sin(rpy[2])
    return wp.mat33(cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
                    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
                    -sp, cp * sr, cp * cr)


@wp.kernel
def candidate_kernel(best_rpy: wp.array(dtype=wp.vec3), deltas: wp.array(dtype=wp.vec3), camera_from_base: wp.mat33,
                     max_rotation: float, full: wp.array2d(dtype=wp.float32), rotations: wp.array(dtype=wp.mat33),
                     rpys: wp.array(dtype=wp.vec3), allowed: wp.array(dtype=wp.int32), state: wp.array2d(dtype=wp.float32)):
    # Candidate c steps from the current best rotation and restarts from the full fit's offset
    c = wp.tid()
    rpy = best_rpy[0] + deltas[c]
    rpys[c] = rpy
    allowed[c] = int(wp.abs(rpy[0]) <= max_rotation and wp.abs(rpy[1]) <= max_rotation and wp.abs(rpy[2]) <= max_rotation)
    rotations[c] = camera_from_base * rpy_rotation(rpy)
    for k in range(3):
        state[c, k] = full[0, k]
    state[c, 3] = ROTATION_GATE_C


@wp.kernel
def select_kernel(count: wp.array2d(dtype=wp.int32), state: wp.array2d(dtype=wp.float32), rpys: wp.array(dtype=wp.vec3),
                  allowed: wp.array(dtype=wp.int32), best_rpy: wp.array(dtype=wp.vec3), best: wp.array(dtype=wp.float32)):
    # One thread, in candidate order: a candidate replaces the best only with a strictly larger inlier share
    for c in range(count.shape[0]):
        share = float(count[c, 1]) / wp.max(float(count[c, 2]), 1.0)
        if allowed[c] == 1 and share > best[0]:
            best[0] = share
            for k in range(3):
                best[1 + k] = state[c, k]
            best_rpy[0] = rpys[c]


class GripperFit:
    def __init__(self, mesh_path: str):
        triangles = read_stl(mesh_path)
        vertices, index = np.unique(np.round(triangles.reshape(-1, 3), 9), axis=0, return_inverse=True)
        self.mesh = wp.Mesh(points=wp.array(vertices.astype(np.float32), dtype=wp.vec3),
                            indices=wp.array(index.reshape(-1).astype(np.int32), dtype=wp.int32))
        self.low, self.high = vertices.min(0), vertices.max(0)
        self.plan = schedule()
        self.rotation_plan = schedule(ROTATION_GATE, ROTATION_PASSES)
        device = wp.get_device()
        wp.load_module(__name__, device=device, block_dim=1 if device.is_cpu else None)

    def roi(self, k: np.ndarray, camera_from_base: np.ndarray, width: int, height: int):
        # The gripper box at this pose, projected and padded for the offset
        corners = np.array([[x, y, z] for x in (self.low[0], self.high[0]) for y in (self.low[1], self.high[1])
                            for z in (self.low[2], self.high[2])]) @ camera_from_base[:3, :3].T + camera_from_base[:3, 3]
        if np.any(corners[:, 2] <= 0.05):
            return None
        u = k[0, 0] * corners[:, 0] / corners[:, 2] + k[0, 2]
        v = k[1, 1] * corners[:, 1] / corners[:, 2] + k[1, 2]
        u0, u1 = max(int(u.min()) - ROI_PAD, 0), min(int(u.max()) + ROI_PAD, width)
        v0, v1 = max(int(v.min()) - ROI_PAD, 0), min(int(v.max()) + ROI_PAD, height)
        return (u0, v0, u1, v1) if u1 > u0 and v1 > v0 else None

    def fit(self, depth_mm: np.ndarray, k: np.ndarray, camera_from_base: np.ndarray, start: np.ndarray,
            lock_rotation: bool = True, max_translation: float = MAX_TRANSLATION,
            max_rotation: float = np.radians(MAX_ROTATION_DEG)) -> dict | None:
        # start and the returned correction are the estimate in the encoder's gripper frame; pose is it in the camera
        free = np.isin(np.arange(3), ROTATION_AXES) & (not lock_rotation)
        seed = np.eye(4)
        seed[:3, :3] = rpy_matrix(np.clip(matrix_rpy(start[:3, :3]), -max_rotation * free, max_rotation * free))
        seed[:3, 3] = np.clip(start[:3, 3], -max_translation, max_translation)
        best = self._fit_pose(depth_mm, k, camera_from_base, seed, free, max_rotation)
        if best is None or np.linalg.norm(best['offset']) > max_translation:
            return None
        correction = np.eye(4)
        correction[:3, :3] = best['rotation']
        correction[:3, 3] = camera_from_base[:3, :3].T @ best['offset']
        return dict(correction=correction, pose=camera_from_base @ correction,
                    inliers=best['inliers'], points=best['points'])

    def _fit_pose(self, depth_mm, k, camera_from_base, seed, free, max_rotation):
        # (1) Upload one region of interest covering the gripper box at the encoder position and every searched rotation
        height, width = depth_mm.shape
        crop_t = camera_from_base[:3, 3]
        reach = [np.zeros(3)] + [sign * max_rotation * np.eye(3)[axis] for axis in np.flatnonzero(free) for sign in (-1, 1)]
        rois = [self.roi(k, np.r_[np.c_[camera_from_base[:3, :3] @ rpy_matrix(rpy), crop_t], [[0, 0, 0, 1]]], width, height)
                for rpy in reach]
        rois = [r for r in rois if r is not None]
        if not rois:
            return None
        u0, v0 = min(r[0] for r in rois), min(r[1] for r in rois)
        u1, v1 = max(r[2] for r in rois), max(r[3] for r in rois)
        patch = depth_mm[v0:v1, u0:u1]
        nonzero = patch[patch > 0]
        if len(nonzero) == 0:
            return None
        fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        frame = (wp.array(np.ascontiguousarray(patch, dtype=np.float32) / 1000.0, dtype=wp.float32), u0, v0, fx, fy, cx,
                 cy, max(float(nonzero.min()) / 1000.0, 0.1), wp.vec3(*camera_from_base[:3, 3]), wp.vec3(*crop_t))

        # (2) The full fit at the encoder rotation, from the seeded offset
        observed = wp.zeros(1, dtype=wp.int32)
        full = wp.array(np.r_[camera_from_base[:3, :3] @ seed[:3, 3], FIRST_GATE][None].astype(np.float32),
                        dtype=wp.float32)
        rotations = wp.array(camera_from_base[:3, :3][None].astype(np.float32), dtype=wp.mat33)
        full_count = self._passes(frame, rotations, full, self.plan, observed)
        if not free.any():
            count = full_count.numpy()[0]
            return dict(offset=full.numpy()[0, :3].astype(float), rotation=np.eye(3),
                        inliers=float(count[1]) / max(int(count[2]), 1), points=int(observed.numpy()[0]))

        # (3) The rotation search, entirely on the GPU: no rotation first, then each step either side of the best so far
        best_rpy, best = wp.zeros(1, dtype=wp.vec3), wp.array([-1.0, 0.0, 0.0, 0.0], dtype=wp.float32)
        groups = [[np.zeros(3)]] + [[sign * np.radians(step) * np.eye(3)[axis] for sign in (-1.0, 1.0)]
                                    for step in ROTATION_STEPS_DEG for axis in np.flatnonzero(free)]
        for deltas in groups:
            n = len(deltas)
            rotations, rpys = wp.empty(n, dtype=wp.mat33), wp.empty(n, dtype=wp.vec3)
            allowed, state = wp.empty(n, dtype=wp.int32), wp.empty((n, 4), dtype=wp.float32)
            wp.launch(candidate_kernel, dim=n, inputs=[best_rpy, wp.array(np.array(deltas, np.float32), dtype=wp.vec3),
                                                       wp.mat33(*camera_from_base[:3, :3].ravel()), float(max_rotation),
                                                       full, rotations, rpys, allowed, state])
            count = self._passes(frame, rotations, state, self.rotation_plan, wp.zeros(n, dtype=wp.int32))
            wp.launch(select_kernel, dim=1, inputs=[count, state, rpys, allowed, best_rpy, best])
        result, rpy = best.numpy(), best_rpy.numpy()[0]
        return dict(offset=result[1:].astype(float), rotation=rpy_matrix(rpy.astype(float)), inliers=float(result[0]),
                    points=int(observed.numpy()[0]))

    def _passes(self, frame, rotations, state, plan, observed):
        depth, u0, v0, fx, fy, cx, cy, z_min, t, crop_t = frame
        h, w = depth.shape
        n = rotations.shape[0]
        points, valid = wp.empty((n, h, w), dtype=wp.vec3), wp.empty((n, h, w), dtype=wp.int32)
        wp.launch(observed_kernel, dim=(n, h, w), inputs=[depth, u0, v0, fx, fy, cx, cy, rotations, crop_t,
                                                        wp.vec3(*self.low), wp.vec3(*self.high), points, valid, observed])
        hist = wp.zeros((n, 3 * BINS), dtype=wp.int32)
        count = wp.zeros((n, 3), dtype=wp.int32)
        for level in plan:
            hist.zero_()
            count.zero_()
            wp.launch(match_kernel, dim=(n, (h + level - 1) // level, (w + level - 1) // level),
                      inputs=[self.mesh.id, points, valid, u0, v0, level, fx, fy, cx, cy, z_min, rotations, t, state,
                              hist, count])
            wp.launch(median_step_kernel, dim=(n, 3), inputs=[hist, count, state])
            wp.launch(gate_kernel, dim=n, inputs=[state, SHRINK])
        return count


def register_depth(depth_m: torch.Tensor, k_depth: np.ndarray, color_from_depth: np.ndarray, k_color: np.ndarray,
                   shape: tuple) -> torch.Tensor:
    # Every depth pixel moved into the colour camera, keeping the nearest point where several land on one pixel
    v, u = torch.nonzero(depth_m > 0, as_tuple=True)
    z = depth_m[v, u]
    p = torch.stack([(u + 0.5 - k_depth[0, 2]) * z / k_depth[0, 0], (v + 0.5 - k_depth[1, 2]) * z / k_depth[1, 1], z], 1)
    p = p @ torch.as_tensor(color_from_depth[:3, :3].T, dtype=p.dtype, device=p.device) + \
        torch.as_tensor(color_from_depth[:3, 3], dtype=p.dtype, device=p.device)
    uc = torch.floor(k_color[0, 0] * p[:, 0] / p[:, 2] + k_color[0, 2]).long()
    vc = torch.floor(k_color[1, 1] * p[:, 1] / p[:, 2] + k_color[1, 2]).long()
    ok = (uc >= 0) & (uc < shape[1]) & (vc >= 0) & (vc < shape[0]) & (p[:, 2] > 0.05)
    out = torch.full((shape[0] * shape[1],), float('inf'), device=p.device)
    out.scatter_reduce_(0, vc[ok] * shape[1] + uc[ok], p[ok, 2], reduce='amin')
    out[torch.isinf(out)] = 0
    return out.reshape(shape)


def cpu_inverse(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # The Jetson torch build cannot load its CUDA linalg library, so GPU matrices are inverted on the CPU
    return TORCH_INVERSE(x.cpu(), *args, **kwargs).to(x.device) if x.is_cuda else TORCH_INVERSE(x, *args, **kwargs)


def normal_transform(height: int, width: int, like: torch.Tensor) -> torch.Tensor:
    # kornia's normal_transform_pixel: pixel coordinates to [-1, 1] over (size - 1)
    m = torch.eye(3, dtype=like.dtype, device=like.device)
    m[0, 0], m[0, 2] = 2.0 / max(width - 1, 1e-14), -1.0
    m[1, 1], m[1, 2] = 2.0 / max(height - 1, 1e-14), -1.0
    return m


def shared_warp_perspective(src: torch.Tensor, M: torch.Tensor, dsize: tuple, mode: str = 'bilinear',
                            padding_mode: str = 'zeros', align_corners: bool = True, fill_value=None) -> torch.Tensor:
    # kornia's warp_perspective math, with the sampling grid built once per crop transform: the colour and XYZ crops
    # of one refinement share M, so the second warp only samples
    h, w = src.shape[-2:]
    h_out, w_out = dsize
    key = (h, w, h_out, w_out, src.dtype, src.device)
    if WARP_CACHE['M'] is not M or WARP_CACHE['key'] != key:
        Ms = M.to(src.dtype)
        if key[:2] + key[4:] not in WARP_CACHE['src_inverse']:
            WARP_CACHE['src_inverse'][key[:2] + key[4:]] = cpu_inverse(normal_transform(h, w, Ms))
        if key[2:] not in WARP_CACHE['points']:
            ys = torch.linspace(-1, 1, h_out, device=src.device, dtype=src.dtype)
            xs = torch.linspace(-1, 1, w_out, device=src.device, dtype=src.dtype)
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')
            WARP_CACHE['points'][key[2:]] = (torch.stack([gx, gy, torch.ones_like(gx)], -1).reshape(1, -1, 3),
                                             normal_transform(h_out, w_out, Ms))
        points, dst_normal = WARP_CACHE['points'][key[2:]]
        src_from_dst = cpu_inverse(dst_normal @ Ms @ WARP_CACHE['src_inverse'][key[:2] + key[4:]])
        mapped = points @ src_from_dst.transpose(-1, -2)
        z = mapped[..., 2:3]
        scale = torch.where(z.abs() > 1e-8, 1.0 / (z + 1e-8), torch.ones_like(z))
        WARP_CACHE.update(M=M, key=key, grid=(mapped[..., :2] * scale).reshape(-1, h_out, w_out, 2))
    return torch.nn.functional.grid_sample(src, WARP_CACHE['grid'], mode=mode, padding_mode=padding_mode,
                                           align_corners=align_corners)


TORCH_INVERSE = torch.linalg.inv
WARP_CACHE = {'M': None, 'key': None, 'grid': None, 'src_inverse': {}, 'points': {}}


class GripperFoundationPose:
    def __init__(self, mesh_path: str, iterations: int = REFINE_ITERATIONS):
        # Global registration is never used: its rotation grid (and the mycpp extension) is skipped, and scorer=False stops
        # FoundationPose from loading its score network
        FoundationPose.make_rotation_grid = lambda self, *args, **kwargs: None
        # FoundationPose and kornia invert 3x3 crop transforms on the GPU and rebuild the crop grid for every warp
        torch.linalg.inv = torch.inverse = torch.Tensor.inverse = cpu_inverse
        kornia.geometry.transform.warp_perspective = shared_warp_perspective
        self.mesh = trimesh.load(mesh_path)
        self.glctx = dr.RasterizeCudaContext()
        self.estimator = FoundationPose(
            model_pts=self.mesh.vertices, 
            model_normals=self.mesh.vertex_normals, 
            mesh=self.mesh,
            scorer=False, 
            refiner=PoseRefinePredictor(), 
            glctx=self.glctx, 
            debug=0, debug_dir=os.path.join(FOUNDATIONPOSE_DIR, 'debug')
        )
        
        # Inverted on the CPU: the Jetson torch build cannot load its CUDA linalg library
        centred = np.linalg.inv(self.estimator.get_tf_to_centered_mesh().cpu().numpy())
        self.to_centred = torch.as_tensor(centred, dtype=torch.float32, device='cuda')
        self.iterations = iterations
        self.vertices = torch.as_tensor(self.mesh.vertices, dtype=torch.float32, device='cuda')
        self.faces = torch.as_tensor(self.mesh.faces, dtype=torch.int32, device='cuda')

    def in_view(self, k: np.ndarray, pose: np.ndarray, shape: tuple) -> float:
        p = self.mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
        u, v = k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2], k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]
        return float(np.mean((p[:, 2] > 0) & (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])))

    def render_depth(self, pose: np.ndarray, k: np.ndarray, shape: tuple) -> torch.Tensor:
        h, w = shape
        r = torch.as_tensor(pose[:3, :3].T, dtype=torch.float32, device='cuda')
        p = self.vertices @ r + torch.as_tensor(pose[:3, 3], dtype=torch.float32, device='cuda')
        x = (k[0, 0] * p[:, 0] / p[:, 2] + k[0, 2]) / w * 2 - 1
        y = (k[1, 1] * p[:, 1] / p[:, 2] + k[1, 2]) / h * 2 - 1
        near, far = 0.05, 3.0
        z = (far + near) / (far - near) - 2 * far * near / ((far - near) * p[:, 2])
        clip = torch.stack([x * p[:, 2], y * p[:, 2], z * p[:, 2], p[:, 2]], 1)[None]
        rast, _ = dr.rasterize(self.glctx, clip, self.faces, resolution=[h, w])
        depth, _ = dr.interpolate(p[None, :, 2:3].contiguous(), rast, self.faces)
        return depth[0, :, :, 0] * (rast[0, :, :, 3] > 0)

    def score(self, pose: np.ndarray, k: np.ndarray, depth: torch.Tensor) -> tuple:
        # Visible rays only: a ray is skipped when depth is missing or something sits clearly in front of the gripper
        rendered = self.render_depth(pose, k, depth.shape)
        diff = depth - rendered
        judged = (rendered > 0) & (depth > 0) & (diff > -OCCLUDED)
        rays = int(judged.sum())
        return float((judged & (diff.abs() < NEAR)).sum()) / max(rays, 1), rays

    def fit(self, rgb: np.ndarray, depth_mm: np.ndarray, k_depth: np.ndarray, color_from_depth: np.ndarray,
            k_color: np.ndarray, color_from_base: np.ndarray) -> dict | None:
        # rgb and k_color share one resolution; the pose is the gripper base in the colour camera, seeded from the encoder
        if self.in_view(k_color, color_from_base, rgb.shape[:2]) < MIN_IN_VIEW:
            return None
        depth = register_depth(torch.as_tensor(depth_mm.astype(np.float32) / 1000.0, device='cuda'), k_depth,
                               color_from_depth, k_color, rgb.shape[:2])
        self.estimator.pose_last = torch.as_tensor(color_from_base, dtype=torch.float32, device='cuda') @ self.to_centred
        pose = self.estimator.track_one(rgb=rgb, depth=depth.cpu().numpy(), K=k_color, iteration=self.iterations)
        inliers, rays = self.score(pose, k_color, depth)
        return dict(pose=pose, correction=np.linalg.inv(color_from_base) @ pose, inliers=inliers, points=rays)
