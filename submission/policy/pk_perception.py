"""Book localisation for the PickSingleBookFromTable hybrid controller (Oct 28): head-camera depth only (the front camera
looks at the back of the bookshelf), calibrated constants CAMS from hybrid_s2b (same cameras, robot frame).

The book lies flat on the counter (top at robot z 0.7901). Its top face is the dominant height cluster 2-4 cm above the
counter (it is 4 mm higher at step 0, while the book is still dropping onto the counter). The top face is registered to a
template of the book's top outline in its body frame: the 9.39 x 13.82 cm main block plus the ribbon bookmark (7 mm wide,
sticking 2.6 cm out of the +y end next to the spine), which makes the outline asymmetric and resolves the 180 deg
ambiguity of a rectangle. Unknowns: x, y, yaw of the body origin. Score = observed top points inside the posed outline
(1.5 mm margin) + 2 x side-face points (6-20 mm above the bottom, visible on the faces toward the camera) within 2.5 mm
of the posed side outline (the page block, recessed 3-5 mm from the covers except at the spine) - 2 x side points deep
inside it - 3 x template points that the camera sees through (all four neighbouring pixels see more than 1 cm beyond the
template point). Without the side term the fit slid 4.7 mm toward the camera: rays just in front of the near edge hit
the book's side face within 1 cm, so nothing penalised a template overhanging the near edge.. Search: yaw from the minimum-area rectangle of the top
points (two candidates 180 deg apart, +-4 deg), then local refinements of x, y, yaw; the estimate is the centre of the
best plateau. Returns the body pose in the robot frame and quality numbers."""
import numpy as np

try:
    from .hybrid_s2b import CAMS
except ImportError:
    from hybrid_s2b import CAMS

Z_COUNTER = 0.7901
BOOK_X = (-0.0467, 0.0472)
BOOK_Y = (-0.0777, 0.0605)
RIB_X, RIB_Y = (-0.0255, -0.0160), (0.0605, 0.0860)  # ribbon strip on the top face level (body frame)
SIDE_X, SIDE_Y = (-0.0467, 0.0423), (-0.0745, 0.0574)  # side faces 6-20 mm above the bottom: the page block is recessed
# 3-5 mm from the covers' edges at the fore-edge and both ends (the spine is flush)
REGION = dict(x=(0.39, 0.74), y=0.34)


def _template(step=0.002):
    xs = np.arange(BOOK_X[0] + step / 2, BOOK_X[1], step)
    ys = np.arange(BOOK_Y[0] + step / 2, BOOK_Y[1], step)
    M = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    rx = np.arange(RIB_X[0] + step / 2, RIB_X[1], step)
    ry = np.arange(RIB_Y[0] + step / 2, RIB_Y[1], step)
    Rb = np.stack(np.meshgrid(rx, ry), -1).reshape(-1, 2)
    return M, Rb


_TM, _TR = _template()


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
    col = C[:, 0] * f / z + w / 2 - 0.5
    row = -C[:, 1] * f / z + h / 2 - 0.5
    return col, row, z


def _inside(xy, pose, margin):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rel = xy - pose[:2]
    bx = rel @ np.array([c, s])
    by = rel @ np.array([-s, c])
    main = (bx > BOOK_X[0] - margin) & (bx < BOOK_X[1] + margin) & (by > BOOK_Y[0] - margin) & (by < BOOK_Y[1] + margin)
    rib = (bx > RIB_X[0] - margin) & (bx < RIB_X[1] + margin) & (by >= BOOK_Y[1] - margin) & (by < RIB_Y[1] + margin)
    return main | rib


class _Scorer:
    def __init__(self, T, S, depth, h_top, sub=3, thr=0.01):
        self.T, self.S, self.depth, self.h_top, self.thr = T, S, depth, h_top, thr
        self.TM = np.r_[_TM[::sub], _TR]  # template points for the see-through term

    def __call__(self, pose):
        n_in = int(np.count_nonzero(_inside(self.T[:, :2], pose, 0.0015)))
        c, s = np.cos(pose[2]), np.sin(pose[2])
        W = self.TM @ np.array([[c, s], [-s, c]]) + pose[:2]
        P = np.c_[W, np.full(len(W), self.h_top)]
        col, row, z = project(P, "head", self.depth.shape)
        h, w = self.depth.shape
        c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
        ok = (c0 >= 0) & (c0 < w - 1) & (r0 >= 0) & (r0 < h - 1)
        obs_d = np.full(len(W), np.nan)
        cc, rr = c0[ok], r0[ok]
        # conservative: seen through only if all four neighbouring pixels see beyond the template point
        obs_d[ok] = np.minimum(np.minimum(self.depth[rr, cc], self.depth[rr, cc + 1]),
                               np.minimum(self.depth[rr + 1, cc], self.depth[rr + 1, cc + 1]))
        through = int(np.count_nonzero(obs_d > z + self.thr))
        n_on, n_bad = 0, 0
        if len(self.S):
            rel = self.S[:, :2] - pose[:2]
            bx, by = rel @ np.array([c, s]), rel @ np.array([-s, c])
            d = np.maximum.reduce([SIDE_X[0] - bx, bx - SIDE_X[1], SIDE_Y[0] - by, by - SIDE_Y[1]])
            n_on = int(np.count_nonzero(np.abs(d) < 0.0025))
            n_bad = int(np.count_nonzero(d < -0.004))
        return n_in + 2 * n_on - 3 * through - 2 * n_bad, n_in, through, n_on, n_bad


def _minrect_yaw(xy):
    c = xy - xy.mean(0)
    best = None
    for t in np.radians(np.arange(0.0, 180.0, 1.0)):
        u = c @ np.array([np.cos(t), np.sin(t)])
        v = c @ np.array([-np.sin(t), np.cos(t)])
        a = (np.quantile(u, 0.99) - np.quantile(u, 0.01)) * (np.quantile(v, 0.99) - np.quantile(v, 0.01))
        if best is None or a < best[0]:
            best = (a, t)
    return best[1]


def locate_book(obs, exclude=None):
    """book body pose {xy, yaw, ok, ...} (robot frame) from the head depth image. exclude: list of (centre, radius) balls
    whose points are ignored (the robot's hands)"""
    P = cloud(obs, "head")
    depth = np.asarray(obs["depth_head"], np.float64)
    m = ((P[:, 0] > REGION["x"][0]) & (P[:, 0] < REGION["x"][1]) & (np.abs(P[:, 1]) < REGION["y"])
         & (P[:, 2] > Z_COUNTER + 0.012) & (P[:, 2] < Z_COUNTER + 0.045))
    if exclude:
        for ctr, r in exclude:
            m &= np.linalg.norm(P - np.asarray(ctr), axis=1) > r
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
    Sd = P[m0 := ((P[:, 0] > REGION["x"][0]) & (P[:, 0] < REGION["x"][1]) & (np.abs(P[:, 1]) < REGION["y"])
                  & (P[:, 2] > h_top - 0.021) & (P[:, 2] < h_top - 0.007))]
    if exclude:
        for c_, r_ in exclude:
            Sd = Sd[np.linalg.norm(Sd - np.asarray(c_), axis=1) > r_]
    t0 = _minrect_yaw(T[:, :2])
    ctr = T[:, :2].mean(0)
    sc = _Scorer(T, Sd, depth, h_top)
    best = [None] * 4
    # the main block's long axis is body y; the min-area rectangle gives the axes mod 90 deg: try all four
    for k in range(4):
        yaw0 = t0 + k * np.pi / 2
        # body origin from the outline centre (the main block's centre is at body (0.0002, -0.0086))
        c, s = np.cos(yaw0), np.sin(yaw0)
        org = ctr - np.array([[c, -s], [s, c]]) @ np.array([0.0002, -0.0086])
        for dyaw in np.radians(np.arange(-4.0, 4.01, 1.0)):
            for dx in np.arange(-0.008, 0.0081, 0.004):
                for dy in np.arange(-0.008, 0.0081, 0.004):
                    pose = np.array([org[0] + dx, org[1] + dy, yaw0 + dyaw])
                    v = sc(pose)
                    if best[k] is None or v[0] > best[k][0][0]:
                        best[k] = (v, pose)
    k1 = int(np.argmax([b[0][0] for b in best]))
    fams = []
    for k in (k1, (k1 + 2) % 4):  # the best axis assignment and its 180-deg flip, each refined
        v, pose = best[k]
        for step_xy, step_yaw, span in ((0.002, np.radians(0.5), 2), (0.001, np.radians(0.25), 2), (0.0005, np.radians(0.1), 2)):
            cand = []
            for dyaw in np.arange(-span, span + 1) * step_yaw:
                for dx in np.arange(-span, span + 1) * step_xy:
                    for dy in np.arange(-span, span + 1) * step_xy:
                        p = pose + np.array([dx, dy, dyaw])
                        cand.append((sc(p), p))
            top = max(c_[0][0] for c_ in cand)
            plateau = np.array([c_[1] for c_ in cand if c_[0][0] >= top])
            pose = plateau.mean(0)
        fams.append((sc(pose), pose))
    (best_v, pose), (alt, _) = (fams[0], fams[1]) if fams[0][0][0] >= fams[1][0][0] else (fams[1], fams[0])
    best = (best_v, pose)
    yaw = float((pose[2] + np.pi) % (2 * np.pi) - np.pi)
    return {"ok": True, "xy": pose[:2].copy(), "yaw": yaw, "score": best[0], "flip_score": alt, "n_top": len(T), "n_side": len(Sd),
            "h_top": h_top, "margin": int(best[0][0] - alt[0])}
