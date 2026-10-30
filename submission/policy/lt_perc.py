"""LiftTray perception (Oct 28): the tray pose (centre xy and long-axis angle, robot frame) from the head and front depth
images, by registering the tray's visual-mesh surface (tray_mesh.npz: the released asset's visual mesh, 744 vertices and
548 triangles in the tray frame, read once during development like the camera constants) to the back-projected depth
points.

- Points: head + front depth back-projected with the calibrated constants CAMS (hybrid_s2b.py; the cameras sit on the
  robot base, so the points are in the robot frame of fk). Kept: x 0.15-1.0, |y| < 0.8, z between the table top + 1.2 cm
  and 1.06 m (the tray is 9.4 cm tall; the arms start above 1.1 m).
- Height: the tray floor (2.2 cm above the tray bottom) is the most populated height in that band; z0 = floor - 0.022.
  At the first observation the tray still hangs 5.5 cm above the table (the env's settle step leaves it there and it
  drops during the first control step); the fit uses the measured z0, so it works before and after the drop.
- Fit: score = number of points on the model surface (occupancy grid of 3 mm voxels). Coarse search with the model
  dilated by 21 mm in xy over the full 180 deg of axis angles (3 deg steps; the tray is symmetric under a half turn) and
  +-9 cm around the points' median (15 mm steps), then 9 mm dilation (5 mm, 1 deg steps) and 3 mm (2 mm and 0.25 deg over +-8 mm / +-2 deg, then 1 mm and 0.1 deg).
  Without the coarse dilation, wall points only count within ~4 mm and the floor alone cannot fix the position along
  the long axis (errors up to 5 cm on 20 seeds). Returns the centre, the axis angle, the score and the point count."""
from pathlib import Path

import numpy as np

try:
    from .hybrid_s2b import CAMS
except ImportError:
    from hybrid_s2b import CAMS

VOX = 0.003
G0 = np.array([-0.32, -0.21, -0.03])  # model grid origin (tray frame: x long axis, y across, z up from the bottom)
GN = np.array([214, 140, 54])
FLOOR = 0.022  # floor top above the tray bottom (visual mesh)
Z_TABLE = 0.870  # table top under the tray (tray bottom when resting; body z 0.873 incl. the 3 mm collision skin)
LEVELS = (7, 3, 1)  # xy dilation (voxels) of the coarse, medium and fine model; z dilation is 1 voxel at every level
_OCC = None
_PIX = {}


def _dilate(a, r, axes):
    for ax in axes:
        b = a.copy()
        for k in range(1, r + 1):
            sl_a, sl_b = [slice(None)] * 3, [slice(None)] * 3
            sl_a[ax], sl_b[ax] = slice(k, None), slice(None, -k)
            b[tuple(sl_a)] |= a[tuple(sl_b)]
            b[tuple(sl_b)] |= a[tuple(sl_a)]
        a = b
    return a


def _model():
    """occupancy grids of the tray surface at the three dilation levels (built once per process, ~0.3 s)"""
    global _OCC
    if _OCC is None:
        z = np.load(Path(__file__).with_name("tray_mesh.npz"))
        V, F = z["V"].astype(np.float64), z["F"].astype(np.int64)
        A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
        rng = np.random.default_rng(0)  # fixed: the same model in every process
        n = 200000
        i = rng.choice(len(area), n, p=area / area.sum())
        r1, r2 = rng.random(n), rng.random(n)
        s = np.sqrt(r1)
        P = (1 - s)[:, None] * A[i] + (s * (1 - r2))[:, None] * B[i] + (s * r2)[:, None] * C[i]
        idx = np.floor((P - G0) / VOX).astype(int)
        occ = np.zeros(GN, bool)
        ok = np.all((idx >= 0) & (idx < GN), axis=1)
        occ[tuple(idx[ok].T)] = True
        occ = _dilate(occ, 1, (2,))
        _OCC = [_dilate(occ, r, (0, 1)) for r in LEVELS]
    return _OCC


def cloud(obs, zlo=Z_TABLE + 0.012, zhi=1.06):
    """head + front depth points (robot frame) in the tray's height band over the table"""
    out = []
    for n in ("head", "front"):
        c = CAMS[n]
        dep = np.asarray(obs[f"depth_{n}"], np.float64)
        h, w = dep.shape
        if (n, h, w) not in _PIX:
            ys, xs = np.mgrid[0:h, 0:w]
            f = (h / 2) / np.tan(np.radians(c["fovy"]) / 2)
            _PIX[(n, h, w)] = ((xs.ravel() + 0.5 - w / 2) / f, -(ys.ravel() + 0.5 - h / 2) / f)
        ux, uy = _PIX[(n, h, w)]
        z = dep.ravel()
        P = np.stack([ux * z, uy * z, -z], 1) @ np.array(c["mat"]).T + np.array(c["pos"])
        P = P[(P[:, 0] > 0.15) & (P[:, 0] < 1.0) & (np.abs(P[:, 1]) < 0.8) & (P[:, 2] > zlo) & (P[:, 2] < zhi)]
        out.append(P)
    return np.concatenate(out)


def _score(P, cx, cy, th, z0, occ):
    """number of points on the model surface for each pose (cx, cy, th arrays of equal length)"""
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    dx, dy = P[None, :, 0] - cx[:, None], P[None, :, 1] - cy[:, None]
    ix = np.floor(((dx * c + dy * s) - G0[0]) / VOX).astype(np.int32)
    iy = np.floor(((-dx * s + dy * c) - G0[1]) / VOX).astype(np.int32)
    iz = np.broadcast_to(np.floor((P[None, :, 2] - z0 - G0[2]) / VOX).astype(np.int32), ix.shape)
    ok = (ix >= 0) & (ix < GN[0]) & (iy >= 0) & (iy < GN[1]) & (iz >= 0) & (iz < GN[2])
    hit = np.zeros(ix.shape, bool)
    hit[ok] = occ[ix[ok], iy[ok], iz[ok]]
    return hit.sum(1)


def _search(P, z0, occ, cx0, cy0, th0, dxy, nxy, dth, nth):
    best = None
    ths = th0 + dth * np.arange(-nth, nth + 1)
    offs = dxy * np.arange(-nxy, nxy + 1)
    X, Y = np.meshgrid(cx0 + offs, cy0 + offs, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    for th in ths:
        sc = _score(P, X, Y, np.full(len(X), th), z0, occ)
        j = int(np.argmax(sc))
        if best is None or sc[j] > best[0]:
            best = (int(sc[j]), float(X[j]), float(Y[j]), float(th))
    return best


def locate_tray(obs, rng_seed=0, mask_xy=()):
    """tray estimate {c (xy), th (long-axis angle in [0, pi)), z0 (bottom height), score, n} or None. mask_xy: xy points
    (grippers low over the tray) whose surroundings (12 cm) are left out"""
    occ_c, occ_m, occ_f = _model()
    P = cloud(obs)
    for m in mask_xy:
        P = P[np.linalg.norm(P[:, :2] - np.asarray(m)[None, :2], axis=1) > 0.12]
    if len(P) < 300:
        return None
    hz, be = np.histogram(P[:, 2], bins=np.arange(Z_TABLE, 1.06, 0.002))
    zf = be[int(np.argmax(hz))] + 0.001  # floor top
    z0 = zf - FLOOR
    P = P[(P[:, 2] > z0 + 0.012) & (P[:, 2] < z0 + 0.105)]  # the tray's own height band
    if len(P) < 300:
        return None
    rng = np.random.default_rng(rng_seed)
    Ps = P[rng.choice(len(P), min(len(P), 1500), replace=False)]
    cx0, cy0 = (float(v) for v in np.median(P[:, :2], 0))
    s, cx, cy, th = _search(Ps, z0, occ_c, cx0, cy0, np.pi / 2, 0.015, 6, np.radians(3), 30)  # 180 deg, +-9 cm
    s, cx, cy, th = _search(Ps, z0, occ_m, cx, cy, th, 0.005, 3, np.radians(1), 3)
    Pr = P[rng.choice(len(P), min(len(P), 6000), replace=False)]
    s, cx, cy, th = _search(Pr, z0, occ_f, cx, cy, th, 0.002, 4, np.radians(0.25), 8)
    s, cx, cy, th = _search(Pr, z0, occ_f, cx, cy, th, 0.001, 2, np.radians(0.1), 3)
    return {"c": np.array([cx, cy]), "th": float(th % np.pi), "z0": float(z0), "score": s / len(Pr), "n": len(P)}
