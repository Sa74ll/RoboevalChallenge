"""Panda arm kinematics for planning (Oct 28): forward kinematics from hybrid_s2b's DH model (robot frame, flange = the
gripper body origin used by the action mode) with an analytic geometric Jacobian, and a damped least-squares IK that
follows a straight-line / geodesic path from the current joint angles, like the environment's local solver
(dm_control qpos_from_site_pose) does step by step. Used only to check at plan time that a candidate hand pose is reached
without running a joint into its limit; no simulator state is involved."""
import numpy as np

try:
    from .hybrid_s2b import DH, BASE_Y, TOOL_YAW, _mdh
except ImportError:
    from hybrid_s2b import DH, BASE_Y, TOOL_YAW, _mdh

LIM = np.array([(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698), (-2.8973, 2.8973),
                (-0.0175, 3.7525), (-2.8973, 2.8973)])
_RT = np.array([[np.cos(TOOL_YAW), -np.sin(TOOL_YAW), 0], [np.sin(TOOL_YAW), np.cos(TOOL_YAW), 0], [0, 0, 1.0]])


def fk_full(q, arm):
    """flange position, rotation (gripper frame as in hybrid_s2b.fk) and the 6x7 geometric Jacobian"""
    T = np.eye(4)
    T[:3, 3] = (0.0, BASE_Y[arm], 0.6)
    Z, O = [], []
    for (a, d, al), th in zip(DH, q):
        T = T @ _mdh(a, d, al, th)
        Z.append(T[:3, 2].copy())
        O.append(T[:3, 3].copy())
    T = T @ _mdh(0, 0.107, 0, 0)
    p = T[:3, 3].copy()
    R = T[:3, :3] @ _RT
    J = np.zeros((6, 7))
    for i in range(7):
        J[:3, i] = np.cross(Z[i], p - O[i])
        J[3:, i] = Z[i]
    return p, R, J


def _rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    a = np.arccos(c)
    if a < 1e-9:
        return np.zeros(3)
    if np.pi - a < 1e-4:
        w, V = np.linalg.eigh((R + R.T) / 2)
        return V[:, -1] * a
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (a / (2 * np.sin(a)))


def _expm(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def ik_solve(q0, arm, p_t, R_t, iters=60, lam=0.03, clamp=True):
    """damped least squares from q0 to the pose (p_t, R_t); joints clamped to their limits (the actuators' ranges)"""
    q = np.array(q0, float)
    for _ in range(iters):
        p, R, J = fk_full(q, arm)
        e = np.r_[p_t - p, _rotvec(R_t @ R.T)]
        if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.3:
            dq *= 0.3 / n
        q = q + dq
        if clamp:
            q = np.clip(q, LIM[:, 0], LIM[:, 1])
    p, R, _ = fk_full(q, arm)
    return q, float(np.linalg.norm(p_t - p)), float(np.linalg.norm(_rotvec(R_t @ R.T)))


def link_points(q, arm):
    """origins of the frames of joints 4-7 and the flange (robot frame), plus mid points, for clearance checks"""
    T = np.eye(4)
    T[:3, 3] = (0.0, BASE_Y[arm], 0.6)
    pts = []
    for (a, d, al), th in zip(DH, q):
        T = T @ _mdh(a, d, al, th)
        pts.append(T[:3, 3].copy())
    T = T @ _mdh(0, 0.107, 0, 0)
    pts.append(T[:3, 3].copy())
    P = np.array(pts[3:])
    return np.r_[P, 0.5 * (P[:-1] + P[1:])]


# the bookshelf's lower shelf board above the counter (robot frame): x 0.666-0.872, z 1.109-1.126; links are ~6 cm thick
SHELF_BOX = (np.array([0.666 - 0.07, -0.44, 1.109 - 0.07]), np.array([0.90, 0.57, 1.126 + 0.07]))


def shelf_intrusion(q, arm):
    P = link_points(q, arm)
    lo, hi = SHELF_BOX
    inside = np.all((P > lo) & (P < hi), axis=1)
    return bool(inside.any())


def path_check(q0, arm, poses, n_sub=8):
    """follow a list of (p, R) waypoints from q0 (straight line + geodesic between them, n_sub IK solves each); return
    (final q, worst position error m, worst rotation error rad, smallest joint-limit margin rad; -1 if a link enters the
    space in front of / under the bookshelf's lower shelf board)"""
    q = np.array(q0, float)
    p_prev, R_prev, _ = fk_full(q, arm)
    worst_p, worst_r, margin = 0.0, 0.0, np.inf
    for p_t, R_t in poses:
        rv = _rotvec(R_t @ R_prev.T)
        for j in range(1, n_sub + 1):
            s = j / n_sub
            q, ep, er = ik_solve(q, arm, p_prev + s * (p_t - p_prev), _expm(s * rv) @ R_prev, iters=25)
            margin = min(margin, float(np.min(np.minimum(q - LIM[:, 0], LIM[:, 1] - q))))
            if shelf_intrusion(q, arm):
                margin = -1.0
        worst_p, worst_r = max(worst_p, ep), max(worst_r, er)
        p_prev, R_prev = p_t, R_t
    return q, worst_p, worst_r, margin
