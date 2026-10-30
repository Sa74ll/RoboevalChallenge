"""Book localisation for the StackSingleBookShelf hybrid controller (Oct 28): head-camera depth only (the front camera looks
at the back of the bookshelf), calibrated constants CAMS from hybrid_s2b (same cameras, robot frame).

The book lies flat on the counter with its spine toward the robot (task yaw within +-30 deg). Its top face is the dominant
height cluster 2-4 cm above the counter. The pose (x, y, yaw of the body origin) is registered to the top-face outline of
the book (main block 9.18 x 13.80 cm plus the ribbon bookmark, 6.5 mm wide, sticking 2.6 cm out of the +y end near the
spine, from the asset's hull). Score = top-face points inside the posed outline (1.5 mm margin) + 2 x spine-face points
(6-20 mm above the counter, facing the head camera) within 2.5 mm of the posed spine line - 2 x points of that band deep
inside the outline - 3 x outline points that the camera sees through (all four neighbouring pixels more than 1 cm beyond
the outline point). The spine-face term pins the near edge, which the push and the pinch depend on (the top face alone lets
the fit slide toward the camera, as the PickBook work found). Search: yaw from the minimum-area rectangle of the top
points, folded into +-45 deg (the task's yaw range is +-30 deg, so the short axis is the body x axis), +-4 deg and +-8 mm
around it, then 2 mm / 0.5 deg, 1 mm / 0.25 deg and 0.5 mm / 0.1 deg passes; the estimate is the centre of the best
plateau. The idea of the template score with see-through and side-face terms follows ctrl/pk/pk_perception.py."""
import numpy as np

try:
    from .hybrid_s2b import CAMS
except ImportError:
    from hybrid_s2b import CAMS

Z_COUNTER = 0.7901
TOP_X = (-0.0447, 0.0471)  # top-face outline of the main block (body frame)
TOP_Y = (-0.0775, 0.0605)
RIB_X, RIB_Y = (-0.0260, -0.0195), (0.0605, 0.0866)  # ribbon on the top-face level
SPINE_X = -0.0467  # spine surface (mid height), body frame
SIDE_Y = (-0.0750, 0.0580)  # spine-face band used by the side term (ends excluded)
REGION = dict(x=(0.38, 0.78), y=0.36)


def _template(step=0.002):
    xs = np.arange(TOP_X[0] + step / 2, TOP_X[1], step)
    ys = np.arange(TOP_Y[0] + step / 2, TOP_Y[1], step)
    M = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    rx = np.arange(RIB_X[0] + step / 2, RIB_X[1], step)
    ry = np.arange(RIB_Y[0] + step / 2, RIB_Y[1], step)
    Rb = np.stack(np.meshgrid(rx, ry), -1).reshape(-1, 2)
    return np.r_[M[::3], Rb]


_TM = _template()


def cloud(obs, n="head"):
    """(H*W, 3) robot-frame points of a depth camera"""
    c = CAMS[n]
    R, p = np.array(c["mat"]), np.array(c["pos"])
    dep = np.asarray(obs[f"depth_{n}"], np.float64)
    h, w = dep.shape
    ys, xs = np.mgrid[0:h, 0:w]
    f = (h / 2) / np.tan(np.radians(c["fovy"]) / 2)
    return (np.stack([(xs + 0.5 - w / 2) / f * dep, -(ys + 0.5 - h / 2) / f * dep, -dep], -1) @ R.T + p).reshape(-1, 3)


def project(P, n="head", shape=(256, 256)):
    """pixel coordinates (col, row) and depth of robot-frame points"""
    c = CAMS[n]
    R, p = np.array(c["mat"]), np.array(c["pos"])
    h, w = shape
    f = (h / 2) / np.tan(np.radians(c["fovy"]) / 2)
    C = (P - p) @ R
    z = -C[:, 2]
    return C[:, 0] * f / z + w / 2 - 0.5, -C[:, 1] * f / z + h / 2 - 0.5, z


def _body_xy(xy, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rel = xy - pose[:2]
    return rel @ np.array([c, s]), rel @ np.array([-s, c])


def _inside(xy, pose, margin):
    bx, by = _body_xy(xy, pose)
    main = (bx > TOP_X[0] - margin) & (bx < TOP_X[1] + margin) & (by > TOP_Y[0] - margin) & (by < TOP_Y[1] + margin)
    rib = (bx > RIB_X[0] - margin) & (bx < RIB_X[1] + margin) & (by >= TOP_Y[1] - margin) & (by < RIB_Y[1] + margin)
    return main | rib


class _Scorer:
    def __init__(self, T, S, depth, h_top, thr=0.01):
        self.T, self.S, self.depth, self.h_top, self.thr = T, S, depth, h_top, thr

    def __call__(self, pose):
        n_in = int(np.count_nonzero(_inside(self.T[:, :2], pose, 0.0015)))
        c, s = np.cos(pose[2]), np.sin(pose[2])
        W = _TM @ np.array([[c, s], [-s, c]]) + pose[:2]
        col, row, z = project(np.c_[W, np.full(len(W), self.h_top)], "head", self.depth.shape)
        h, w = self.depth.shape
        c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
        ok = (c0 >= 0) & (c0 < w - 1) & (r0 >= 0) & (r0 < h - 1)
        cc, rr = c0[ok], r0[ok]
        od = np.minimum(np.minimum(self.depth[rr, cc], self.depth[rr, cc + 1]), np.minimum(self.depth[rr + 1, cc], self.depth[rr + 1, cc + 1]))
        through = int(np.count_nonzero(od > z[ok] + self.thr))
        n_on = n_bad = 0
        if len(self.S):
            bx, by = _body_xy(self.S[:, :2], pose)
            band = (by > SIDE_Y[0]) & (by < SIDE_Y[1])
            n_on = int(np.count_nonzero(band & (np.abs(bx - SPINE_X) < 0.0025)))
            n_bad = int(np.count_nonzero(band & (bx > SPINE_X + 0.006) & (bx < TOP_X[1] - 0.004)))
        return n_in + 2 * n_on - 3 * through - 2 * n_bad, n_in, through, n_on, n_bad


def _minrect_yaw(xy):
    c = xy - xy.mean(0)
    best = None
    for t in np.radians(np.arange(0.0, 90.0, 1.0)):
        u = c @ np.array([np.cos(t), np.sin(t)])
        v = c @ np.array([-np.sin(t), np.cos(t)])
        a = (np.quantile(u, 0.98) - np.quantile(u, 0.02)) * (np.quantile(v, 0.98) - np.quantile(v, 0.02))
        if best is None or a < best[0]:
            best = (a, t)
    t = best[1]
    return t if t <= np.pi / 4 else t - np.pi / 2  # body x axis within +-45 deg of world x


def hand_boxes(p, R, opening):
    """boxes (gripper frame lo, hi) covering a hand at flange pose (p, R) with 1 cm margin: palm, the two fingers at the
    given opening (0 closed .. 1 open, as in proprioception_grippers), and the wrist links above the flange (asset AABBs)"""
    q = 0.04 * float(np.clip(opening, 0.0, 1.0))
    yi, yo = 0.0025 + q - 0.01, 0.0263 + q + 0.01
    return [(p, R, np.array([-0.042, -0.115, -0.04]), np.array([0.042, 0.115, 0.076])),
            (p, R, np.array([-0.021, yi, 0.048]), np.array([0.021, yo, 0.126])),
            (p, R, np.array([-0.021, -yo, 0.048]), np.array([0.021, -yi, 0.126])),
            (p, R, np.array([-0.055, -0.10, -0.20]), np.array([0.066, 0.146, 0.0]))]


def locate_book(obs, exclude=None, boxes=None, prior=None):
    """book body pose {ok, xy, yaw, ...} (robot frame) from the head depth image. exclude: (centre, radius) balls whose
    points are ignored; boxes: (p, R, lo, hi) boxes in a hand's gripper frame whose points are ignored (the robot's hands).
    prior: {xy, yaw} expected pose (searched as well as the top face's minimum-area rectangle)."""
    P = cloud(obs, "head")
    depth = np.asarray(obs["depth_head"], np.float64)
    reg = (P[:, 0] > REGION["x"][0]) & (P[:, 0] < REGION["x"][1]) & (np.abs(P[:, 1]) < REGION["y"])
    if exclude:
        for ctr, r in exclude:
            reg &= np.linalg.norm(P - np.asarray(ctr), axis=1) > r
    if boxes:
        for bp, bR, lo, hi in boxes:
            Lc = (P - bp) @ bR
            reg &= ~np.all((Lc > lo) & (Lc < hi), axis=1)
    m = reg & (P[:, 2] > Z_COUNTER + 0.012) & (P[:, 2] < Z_COUNTER + 0.045)
    Q = P[m]
    if len(Q) < 40:
        return {"ok": False, "n": len(Q)}
    hz = Q[:, 2]
    hist, edges = np.histogram(hz, bins=np.arange(Z_COUNTER + 0.018, Z_COUNTER + 0.042, 0.002))
    i = int(np.argmax(hist))
    h0 = 0.5 * (edges[i] + edges[i + 1])
    h_top = float(np.median(hz[np.abs(hz - h0) < 0.003]))
    T = Q[np.abs(hz - h_top) < 0.003]
    if len(T) < 40:
        return {"ok": False, "n": len(T)}
    S = P[reg & (P[:, 2] > h_top - 0.021) & (P[:, 2] < h_top - 0.007)]
    yaw0 = _minrect_yaw(T[:, :2])
    c, s = np.cos(yaw0), np.sin(yaw0)
    ctr = T[:, :2].mean(0)
    # body origin from the outline centre (main block centre at body (0.0012, -0.0085); the ribbon shifts the mean a little)
    org = ctr - np.array([[c, -s], [s, c]]) @ np.array([0.0012, -0.0070])
    sc = _Scorer(T, S, depth, h_top)
    best = None
    starts = [(org, yaw0, 4.0, 0.008)]
    if prior is not None:  # also around the expected pose, wider in yaw (a partly hidden top face can mislead the rectangle)
        starts.append((np.asarray(prior["xy"], float), float(prior["yaw"]), 8.0, 0.012))
    for o, y0, dy_deg, dxy in starts:
        for dyaw in np.radians(np.arange(-dy_deg, dy_deg + 0.01, 1.0)):
            for dx in np.arange(-dxy, dxy + 1e-4, 0.004):
                for dy in np.arange(-dxy, dxy + 1e-4, 0.004):
                    pose = np.array([o[0] + dx, o[1] + dy, y0 + dyaw])
                    v = sc(pose)
                    if best is None or v[0] > best[0][0]:
                        best = (v, pose)
    pose = best[1]
    for step_xy, step_yaw, span in ((0.002, np.radians(0.5), 2), (0.001, np.radians(0.25), 2), (0.0005, np.radians(0.1), 2)):
        cand = []
        for dyaw in np.arange(-span, span + 1) * step_yaw:
            for dx in np.arange(-span, span + 1) * step_xy:
                for dy in np.arange(-span, span + 1) * step_xy:
                    p = pose + np.array([dx, dy, dyaw])
                    cand.append((sc(p)[0], p))
        top = max(c_[0] for c_ in cand)
        pose = np.array([c_[1] for c_ in cand if c_[0] >= top]).mean(0)
    v = sc(pose)
    return {"ok": True, "xy": pose[:2].copy(), "yaw": float(pose[2]), "score": v, "n_top": len(T), "n_side": len(S), "h_top": h_top}
