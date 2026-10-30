"""Panda arm kinematics for planning (Oct 28): forward kinematics of hybrid_s2b's DH model with the full gripper rotation
and the geometric Jacobian, and a damped least-squares IK that follows a straight-line / geodesic path from given joint
angles, like the environment's local solver does step by step. Used at plan time to pick hand orientations that keep
every joint inside its range with margin. Pure kinematics; nothing is read from the simulator."""
import numpy as np

try:
    from .hybrid_s2b import DH, BASE_Y, TOOL_YAW, _mdh
    from .ss_lib import rotvec, expm
except ImportError:
    from hybrid_s2b import DH, BASE_Y, TOOL_YAW, _mdh
    from ss_lib import rotvec, expm

# joint ranges (actuator ctrlrange of the Panda model used by the environment)
LIM = np.array([(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698), (-2.8973, 2.8973),
                (-0.0175, 3.7525), (-2.8973, 2.8973)])
_RT = np.array([[np.cos(TOOL_YAW), -np.sin(TOOL_YAW), 0], [np.sin(TOOL_YAW), np.cos(TOOL_YAW), 0], [0, 0, 1.0]])


def fk_full(q, arm):
    """flange position, gripper rotation (as hybrid_s2b.fk) and the 6x7 geometric Jacobian (robot frame)"""
    T = np.eye(4)
    T[:3, 3] = (0.0, BASE_Y[arm], 0.6)
    Z, O = [], []
    for (a, d, al), th in zip(DH, q):
        T = T @ _mdh(a, d, al, th)
        Z.append(T[:3, 2].copy())
        O.append(T[:3, 3].copy())
    T = T @ _mdh(0, 0.107, 0, 0)
    p = T[:3, 3].copy()
    J = np.zeros((6, 7))
    for i in range(7):
        J[:3, i] = np.cross(Z[i], p - O[i])
        J[3:, i] = Z[i]
    return p, T[:3, :3] @ _RT, J


def joint_points(q, arm):
    """origins of the frames of joints 2-7 and the flange (robot frame): for clearance checks"""
    T = np.eye(4)
    T[:3, 3] = (0.0, BASE_Y[arm], 0.6)
    pts = []
    for (a, d, al), th in zip(DH, q):
        T = T @ _mdh(a, d, al, th)
        pts.append(T[:3, 3].copy())
    T = T @ _mdh(0, 0.107, 0, 0)
    pts.append(T[:3, 3].copy())
    return np.array(pts)


def ik_solve(q0, arm, p_t, R_t, iters=40, lam=0.03, clamp=True):
    """damped least squares from q0 to the pose (p_t, R_t), joints clamped to their ranges"""
    q = np.array(q0, float)
    for _ in range(iters):
        p, R, J = fk_full(q, arm)
        e = np.r_[p_t - p, rotvec(R_t @ R.T)]
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
    return q, float(np.linalg.norm(p_t - p)), float(np.linalg.norm(rotvec(R_t @ R.T)))


def path_check(q0, arm, poses, n_sub=6, clamp=False):
    """follow (p, R) waypoints from q0 (straight line + geodesic between them, n_sub IK solves each). Returns the final
    joints, the worst position error (m) and rotation error (rad), and the smallest joint-range margin (rad; negative when
    the solution leaves a joint's range). Unclamped by default, like the environment's IK (dm_control qpos_from_site_pose):
    the actuators clamp the commanded joints to their ranges, so a solution outside them is not reached."""
    q = np.array(q0, float)
    p_prev, R_prev, _ = fk_full(q, arm)
    worst_p, worst_r, margin = 0.0, 0.0, np.inf
    for p_t, R_t in poses:
        rv = rotvec(R_t @ R_prev.T)
        for j in range(1, n_sub + 1):
            s = j / n_sub
            q, ep, er = ik_solve(q, arm, p_prev + s * (p_t - p_prev), expm(s * rv) @ R_prev, iters=25, clamp=clamp)
            margin = min(margin, float(np.min(np.minimum(q - LIM[:, 0], LIM[:, 1] - q))))
        worst_p, worst_r = max(worst_p, ep), max(worst_r, er)
        p_prev, R_prev = p_t, R_t
    return q, worst_p, worst_r, margin


def path_plan(q0, arm, poses, n_sub=6, rate=0.07):
    """like path_check, and for each waypoint segment the joint displacement: returns (final q, worst position error,
    worst rotation error, smallest joint margin, [minimum segment duration in control steps]) where the minimum duration
    keeps the peak joint rate of a minimum-jerk profile (1.875 x the mean) under `rate` rad/step (the environment moves a
    joint target by at most 0.109 rad per step)"""
    q = np.array(q0, float)
    p_prev, R_prev, _ = fk_full(q, arm)
    worst_p, worst_r, margin = 0.0, 0.0, np.inf
    n_min = []
    for p_t, R_t in poses:
        rv = rotvec(R_t @ R_prev.T)
        qs = q.copy()
        peak = 0.0
        q_last = q.copy()
        for j in range(1, n_sub + 1):
            s = j / n_sub
            q, ep, er = ik_solve(q, arm, p_prev + s * (p_t - p_prev), expm(s * rv) @ R_prev, iters=25, clamp=False)
            margin = min(margin, float(np.min(np.minimum(q - LIM[:, 0], LIM[:, 1] - q))))
            peak = max(peak, float(np.max(np.abs(q - q_last))) * n_sub)  # displacement rate over the segment
            q_last = q.copy()
        worst_p, worst_r = max(worst_p, ep), max(worst_r, er)
        n_min.append(int(np.ceil(1.875 * peak / rate)))
        p_prev, R_prev = p_t, R_t
    return q, worst_p, worst_r, margin, n_min
