"""RotateValve perception (Oct 28): locate both valves from the head and front RGB-D images by registering a point model of
the valve (surface samples of the released valve asset, rv_valve_pts.npy, made by mk_valve_model.py) to the depth points.

Only the standard observations are used: depth_head / depth_front back-projected with the calibrated camera constants
CAMS (hybrid_s2b.py). Each valve has 3 unknowns (x, y, yaw): it rests on the table (known height), turned about the
vertical. The wheel itself is often hidden by the arms at the start (~15% of wheels), so the whole valve (pipe, flanges,
bonnet, wheel) is fitted. Search space = the task's spawn ranges (root x 0.30 +- 0.07; root y 0.20 +- 0.13 or -0.40 +- 0.13;
yaw 90 +- 30 deg), parametrised by the wheel centre so that a yaw error does not move the wheel estimate; coarse grid
(1 cm, 3 deg) then two finer passes. Score = depth points within 4 mm of the posed model (hidden parts cost nothing)
minus 3 x the posed model points that a camera sees through (free space; without it a valve whose wheel is hidden can slide
along its pipe). wide=True (the valves were seen moving: spawned overlapping) searches the arm's half of the table with any
yaw. Accuracy on 200 settled training seeds: median wheel error 1.0 mm, 90th percentile 2.6 mm."""
from pathlib import Path

import numpy as np

try:
    from .hybrid_s2b import CAMS
except ImportError:
    from hybrid_s2b import CAMS

WHEEL_LOCAL = np.array([0.11685, -0.23495, 0.136595])  # wheel joint origin in the valve root frame (asset)
Z_ROOT = 0.94  # valve root height at spawn; the valve (a free body) then drops onto the table: Z_ROOT_SETTLED
Z_ROOT_SETTLED = 0.95 - 0.0127  # table top 0.95 minus the base mesh's lowest point (measured settled root z 0.9370-0.9378)
ROOT_X, ROOT_DX = 0.30, 0.07
ROOT_Y, ROOT_DY = (0.20, -0.40), 0.13  # left slot (y > 0) first
YAW0, DYAW = np.pi / 2, np.radians(30.0)
VOX = 0.002
INLIER = 0.004


class ValveModel:
    def __init__(self, path=None, n_fs=500):
        from scipy.spatial import cKDTree
        path = Path(path) if path else Path(__file__).resolve().parent / "rv_valve_pts.npy"
        M = np.load(path).astype(np.float64)
        rs = np.random.default_rng(1)
        self.fs = M[rs.choice(len(M), n_fs, replace=False)]  # free-space check subsample (local frame)
        self.lo = M.min(0) - 0.02
        hi = M.max(0) + 0.02
        self.shape = np.ceil((hi - self.lo) / VOX).astype(int) + 1
        g = [self.lo[i] + VOX * np.arange(self.shape[i]) for i in range(3)]
        G = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
        d, _ = cKDTree(M).query(G, distance_upper_bound=0.05)
        self.inl = (d <= INLIER).reshape(self.shape)  # voxel centre within INLIER of the model surface

    def count(self, L):
        """L: (..., N, 3) local points -> (...,) inlier counts"""
        idx = np.rint((L - self.lo) / VOX).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < self.shape), axis=-1)
        idx = np.where(ok[..., None], idx, 0)
        v = self.inl[idx[..., 0], idx[..., 1], idx[..., 2]] & ok
        return v.sum(-1)


def cam_points(obs, n):
    c = CAMS[n]
    R, p = np.array(c["mat"]), np.array(c["pos"])
    dep = np.asarray(obs[f"depth_{n}"], np.float64)
    h, w = dep.shape
    ys, xs = np.mgrid[0:h, 0:w]
    f = (h / 2) / np.tan(np.radians(c["fovy"]) / 2)
    xc = (xs + 0.5 - w / 2) / f * dep
    yc = -(ys + 0.5 - h / 2) / f * dep
    return (np.stack([xc, yc, -dep], -1) @ R.T + p).reshape(-1, 3)


class DepthView:
    """the two depth images with their camera constants, for free-space (see-through) checks"""

    def __init__(self, obs):
        self.cams = []
        for n in ("head", "front"):
            c = CAMS[n]
            dep = np.asarray(obs[f"depth_{n}"], np.float64)
            h, w = dep.shape
            f = (h / 2) / np.tan(np.radians(c["fovy"]) / 2)
            self.cams.append((np.array(c["mat"]), np.array(c["pos"]), dep, f, h, w))

    def violations(self, W):
        """W: (..., N, 3) world points -> (...,) number of points the cameras see through (observed depth > point
        depth + 1 cm), i.e. points in space the cameras observe as empty"""
        tot = 0
        for R, p, dep, f, h, w in self.cams:
            C = (W - p) @ R  # camera frame (R columns = camera axes)
            z = -C[..., 2]
            zs = np.maximum(z, 1e-6)
            u = np.rint(C[..., 0] / zs * f + w / 2 - 0.5).astype(np.int64)
            v = np.rint(-C[..., 1] / zs * f + h / 2 - 0.5).astype(np.int64)
            ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            d = dep[np.where(ok, v, 0), np.where(ok, u, 0)]
            tot = tot + ((d > z + 0.01) & ok).sum(-1)
        return tot


def scene_points(obs, zmax=1.104):
    P = np.concatenate([cam_points(obs, n) for n in ("head", "front")])
    m = (P[:, 0] > 0.18) & (P[:, 0] < 0.86) & (np.abs(P[:, 1]) < 0.86) & (P[:, 2] > 0.957) & (P[:, 2] < zmax)
    return P[m]


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _score_grid(model, P, wx, wy, yaws, slot, view=None, lam=0.0, z_root=Z_ROOT, box=(0.006, 0.006)):
    """inlier counts (minus lam x free-space violations of the model subsample) for wheel centres (wx x wy grid) and
    yaws; candidates whose root leaves the spawn box get a very low score"""
    out = np.full((len(yaws), len(wx), len(wy)), -10 ** 9, np.float64)
    WX, WY = np.meshgrid(wx, wy, indexing="ij")
    Wc = np.stack([WX.ravel(), WY.ravel()], 1)  # (K, 2)
    for i, th in enumerate(yaws):
        R = _rot(th)
        roots = Wc - (R @ WHEEL_LOCAL[:2])[None]  # (K, 2)
        if box is None:  # free search (displaced valves): the root only has to be on the table
            okr = (roots[:, 0] > 0.0) & (roots[:, 0] < 0.95) & (np.abs(roots[:, 1]) < 0.9)
        else:
            okr = (np.abs(roots[:, 0] - ROOT_X) <= ROOT_DX + box[0]) & (np.abs(roots[:, 1] - ROOT_Y[slot]) <= ROOT_DY + box[1])
        if not okr.any():
            continue
        q = P[:, :2] @ R  # = R^T p (rows)
        sh = roots[okr] @ R  # R^T root
        L = np.empty((okr.sum(), len(P), 3))
        L[..., :2] = q[None] - sh[:, None]
        L[..., 2] = P[None, :, 2] - z_root
        cnt = np.full(len(Wc), -10 ** 9, np.float64)
        sc = model.count(L).astype(np.float64)
        if view is not None and lam > 0:
            F = model.fs
            Fw = np.empty((okr.sum(), len(F), 3))
            Fw[..., :2] = (F[:, :2] @ R.T)[None] + roots[okr][:, None]
            Fw[..., 2] = F[None, :, 2] + z_root
            sc = sc - lam * view.violations(Fw)
        cnt[okr] = sc
        out[i] = cnt.reshape(len(wx), len(wy))
    return out


def fit_valve(model, P, slot, view=None, lam=3.0, z_root=Z_ROOT, wide=False):
    """best (wheel_xy, yaw, inliers) for the valve in slot 0 (left, y > 0) or 1 (right).
    wide: valves that spawned overlapping push each other apart (about 1% of seeds): search the wheel anywhere on this
    slot's half of the table with any yaw, instead of the spawn ranges."""
    if wide:
        box = None
        wxl, wxh = 0.30, 0.85
        wyl, wyh = (0.0, 0.62) if slot == 0 else (-0.62, 0.0)
        m = (P[:, 1] > wyl - 0.45) & (P[:, 1] < wyh + 0.45)
        yaws = np.radians(np.arange(0.0, 360.0, 6.0))
        step, refine = 0.02, ((0.03, 0.005, np.radians(9.0), np.radians(2.0)), (0.012, 0.003, np.radians(4.0), np.radians(1.0)),
                              (0.004, 0.001, np.radians(1.2), np.radians(0.3)))
        ncoarse = 1000
    else:
        box = (0.006, 0.006)
        yl, yh = ROOT_Y[slot] - ROOT_DY - box[1], ROOT_Y[slot] + ROOT_DY + box[1]
        # wheel centre bounding box over the spawn box and yaw range
        off = np.array([_rot(t) @ WHEEL_LOCAL[:2] for t in np.linspace(YAW0 - DYAW, YAW0 + DYAW, 13)])
        wxl, wxh = ROOT_X - ROOT_DX - box[0] + off[:, 0].min(), ROOT_X + ROOT_DX + box[0] + off[:, 0].max()
        wyl, wyh = yl + off[:, 1].min(), yh + off[:, 1].max()
        # points that can belong to this valve (within reach of the swept model)
        m = (P[:, 1] > yl - 0.2) & (P[:, 1] < yh + 0.42)
        yaws = np.arange(YAW0 - DYAW - np.radians(2), YAW0 + DYAW + np.radians(2.1), np.radians(3.0))
        step, refine = 0.01, ((0.012, 0.003, np.radians(4.0), np.radians(1.0)), (0.004, 0.001, np.radians(1.2), np.radians(0.3)))
        ncoarse = 1500
    Q = P[m]
    if len(Q) < 30:
        return None
    rs = np.random.default_rng(0)
    Qc = Q[rs.choice(len(Q), ncoarse, replace=False)] if len(Q) > ncoarse else Q  # subsample for the coarse pass
    wx, wy = np.arange(wxl, wxh + 1e-9, step), np.arange(wyl, wyh + 1e-9, step)
    S = _score_grid(model, Qc, wx, wy, yaws, slot, view, lam * len(Qc) / max(len(Q), 1), z_root, box)
    i, j, k = np.unravel_index(np.argmax(S), S.shape)
    best = (wx[j], wy[k], yaws[i])
    # refinement passes on all candidate points
    for dxy, sxy, dth, sth in refine:
        wx = best[0] + np.arange(-dxy, dxy + 1e-9, sxy)
        wy = best[1] + np.arange(-dxy, dxy + 1e-9, sxy)
        yaws = best[2] + np.arange(-dth, dth + 1e-9, sth)
        S = _score_grid(model, Q, wx, wy, yaws, slot, view, lam, z_root, box)
        top = S.max()
        # centre of the maximal plateau (ties are common at 1 mm steps)
        ii = np.argwhere(S >= top)
        best = (float(wx[ii[:, 1]].mean()), float(wy[ii[:, 2]].mean()), float(yaws[ii[:, 0]].mean()))
    return dict(wheel=np.array([best[0], best[1], z_root + WHEEL_LOCAL[2]]), yaw=best[2], inliers=int(round(top)), n=len(Q))


def locate_valves(obs, model, lam=3.0, z_root=Z_ROOT, wide=False):
    P = scene_points(obs)
    view = DepthView(obs) if lam > 0 else None
    return [fit_valve(model, P, s, view, lam, z_root, wide) for s in (0, 1)]


def valve_region_change(obs0, obs1, thr=0.0005):
    """number of depth pixels in the valve region (table area, below the grippers' start height) that changed by more
    than thr between two observations: > 0 while a valve is still moving (the robot must be still)"""
    n = 0
    for c in ("head", "front"):
        d0, d1 = np.asarray(obs0[f"depth_{c}"]), np.asarray(obs1[f"depth_{c}"])
        P0 = cam_points(obs0, c)
        z, x = P0[:, 2].reshape(d0.shape), P0[:, 0].reshape(d0.shape)
        m = (z > 0.93) & (z < 1.104) & (x > 0.15) & (x < 0.9)
        n += int((np.abs(d1 - d0)[m] > thr).sum())
    return n
