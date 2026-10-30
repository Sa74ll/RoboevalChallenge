"""CubeHandover rod perception (Oct 28): the "cube" of this task is a red box 0.04 x 0.20 x 0.04 m (roboeval Rod prop,
props/cube/rod.xml), lying on the table at a random yaw. Red pixels of the fixed head and front RGB-D cameras are
back-projected with the calibrated camera constants of hybrid_s2b.CAMS. In this task's scene (cube_handover.yaml) the whole
robot, and with it these cameras, sits 0.1 m further along world x than in StackTwoBlocks, so the same constants give
points in the ROBOT frame (world x - 0.1), the frame of hybrid_s2b.fk as well. Everything here is in that frame.

locate_rod(obs) -> {c: centre (3,), u: unit long axis in xy with u_y >= 0, yaw: atan2(u_y, u_x) in [0, pi), z_top,
                    len: visible length along u, wid: visible width, n: points, ntop: top-face points} or None"""
import numpy as np

try:
    from .hybrid_s2b import CAMS, _points
except ImportError:
    from hybrid_s2b import CAMS, _points

ROD_HALF = (0.02, 0.10, 0.02)  # half extents (x, y = long axis, z) of the rod in its own frame
Z_REST = 0.970  # rod centre resting on the table (robot frame z = world z)


def _top(z):
    """height of the rod's top face: the most populated 2 mm height bin (the top face is flat; side faces spread over 4 cm),
    refined to the 99.5th percentile of the points within 3 mm of it. A plain high percentile of all points fails when a
    few silhouette pixels back-project above the rod ("flying pixels": up to 4 cm above it in the head camera, seed 816960)"""
    b = np.floor(z / 0.002).astype(np.int64)
    vals, cnt = np.unique(b, return_counts=True)
    zc = (vals[np.argmax(cnt)] + 0.5) * 0.002
    near = z[np.abs(z - zc) < 0.003]
    return float(np.percentile(near, 99.5))


def rod_points(obs, zmin=0.93, zmax=1.6):
    P = np.concatenate([_points(obs, n) for n in CAMS])
    return P[(P[:, 2] > zmin) & (P[:, 2] < zmax)]


def _axis(xy):
    c = xy - xy.mean(0)
    w, v = np.linalg.eigh(c.T @ c)
    u = v[:, 1]
    return u if u[1] > 0 or (u[1] == 0 and u[0] > 0) else -u


def locate_rod(obs, P=None):
    """rod estimate from its top face (the rod lying flat, on the table or held level)"""
    if P is None:
        P = rod_points(obs)
    if len(P) < 30:
        return None
    top = _top(P[:, 2])
    T = P[(P[:, 2] > top - 0.006) & (P[:, 2] < top + 0.003)]
    if len(T) < 20:
        return None
    u = _axis(T[:, :2])
    n = np.array([-u[1], u[0]])
    a, b = T[:, :2] @ u, T[:, :2] @ n
    a0, a1 = np.percentile(a, [0.5, 99.5])
    b0, b1 = np.percentile(b, [0.5, 99.5])
    xy = u * (a0 + a1) / 2 + n * (b0 + b1) / 2
    return {"c": np.r_[xy, top - ROD_HALF[2]], "u": u, "yaw": float(np.arctan2(u[1], u[0])), "z_top": top,
            "len": float(a1 - a0), "wid": float(b1 - b0), "n": len(P), "ntop": len(T), "xy_mean": T[:, :2].mean(0)}


def locate_held(obs, hand, w):
    """the rod held level by a gripper at `hand` (3,) and pointing from it along -w (unit xy) towards its free end: the free
    end's top-edge centre {end_xy (2,), z_c: rod centre height there, n: points} from the red points above the table, or
    None. Only the free half is used (the holding gripper hides the rest); the lateral position comes from top-face points."""
    P = rod_points(obs, zmin=Z_REST + 0.025)
    if len(P) < 30:
        return None
    w = np.asarray(w, float)
    nn = np.array([-w[1], w[0]])
    d = P[:, :2] - np.asarray(hand)[:2]
    s, lat = d @ w, d @ nn
    keep = (s < -0.03) & (s > -0.25) & (np.abs(lat) < 0.05)
    if keep.sum() < 30:
        return None
    Q, sq, lq = P[keep], s[keep], lat[keep]
    s_end = float(np.percentile(sq, 1))
    near = sq < s_end + 0.06
    top = _top(Q[near, 2])
    tf = near & (Q[:, 2] > top - 0.006) & (Q[:, 2] < top + 0.003)
    if tf.sum() < 10:
        return None
    lat0 = float(np.median(lq[tf]))
    end_xy = np.asarray(hand)[:2] + s_end * w + lat0 * nn
    return {"end_xy": end_xy, "z_c": top - ROD_HALF[2], "n": int(keep.sum()), "ntop": int(tf.sum())}
