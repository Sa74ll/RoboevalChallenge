"""Hybrid StackTwoBlocks policy (Oct 24, prototype): classical perception + a scripted pick-and-place controller, using
only the challenge's standard observations (the rules permit "classical or hybrid controllers"; nothing is read from the
simulator at test time).

Perception: the fixed head and front cameras' RGB-D images (depth: true in the bundle's metadata). Red pixels are
back-projected with calibrated camera constants (pose and field of view, measured once during development like a real
robot's camera calibration) into 3D points, split into two blocks, and each block's centre and yaw (mod 90 deg) come from
its top-face points (centre of the top face; minimum-area rectangle). Accuracy on 40 training seeds: 1.1 mm, 0.6 deg.
Gripper poses: forward kinematics of the Panda arms from the joint angles in proprioception (DH parameters, arm bases at
(0, +-0.3, 0.6), tool yaw 135 deg; exact). Holding: the finger opening in proprioception_grippers.

Controller: the scripted expert v2 (work/xtask/s2b_expert2.py) on these estimates: one arm, align yaw, descend, close,
lift, carry the held block over the other, lower, open; re-grasp after a miss or a drop. Plan: the cheapest (arm, block)
whose pick and place both lie within the arm's comfortable reach; if none (blocks far apart on opposite sides: ~7% of
seeds, where a single arm stalls at full extension), a relay: one arm sets its block down mid-table and moves away, then
the other arm stacks its block onto it.
Actions: 14-D end-effector deltas [dp_L dr_L | dp_R dr_R | g_L g_R] (world frame), grippers 0 open / 1 closed."""
import numpy as np

# ---- calibrated constants -------------------------------------------------------------------------------------------
CAMS = {
    "head": {"pos": [-0.7, 0.0, 1.65], "fovy": 45.0,
             "mat": [[0.0, 0.49993488868845326, -0.8660629925542734], [-1.0, 0.0, 0.0], [0.0, 0.8660629925542734, 0.49993488868845326]]},
    "front": {"pos": [1.4, 0.0, 1.6], "fovy": 45.0,
              "mat": [[0.0, -0.49993488868845326, 0.8660629925542734], [1.0, 0.0, 0.0], [0.0, 0.8660629925542734, 0.49993488868845326]]},
}
DH = [(0, 0.333, 0), (0, 0, -np.pi / 2), (0, 0.316, np.pi / 2), (0.0825, 0, np.pi / 2), (-0.0825, 0.384, -np.pi / 2),
      (0, 0, np.pi / 2), (0.088, 0, np.pi / 2)]
BASE_Y = (0.3, -0.3)  # left, right arm base (x 0, z 0.6, no rotation)
TOOL_YAW = 2.3562  # gripper body = flange rotated about z
BLOCK_HALF = 0.0215  # top-face z minus block centre z (measured)

H_GRASP, H_APPROACH, H_CARRY = 0.100, 0.17, 0.21
VMAX, KP, RMAX, KR = 0.012, 0.6, 0.06, 0.8
NTOP_MIN = 90  # top-face points needed to trust a fresh estimate (unoccluded: ~130)
REACH = 0.62  # max horizontal distance from an arm's base for a pick or place (carries beyond ~0.65 stall at the arm's limit)
HOME = (np.array([0.42, 0.30, 1.18]), np.array([0.42, -0.30, 1.18]))  # where an arm waits after its half of a relay
Z_TABLE = 0.97  # block centre resting on the table


def _mdh(a, d, al, th):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1.0]])


def fk(q, arm):
    """gripper body position and yaw (world) from the arm's 7 joint angles"""
    T = np.eye(4)
    T[:3, 3] = (0.0, BASE_Y[arm], 0.6)
    for (a, d, al), th in zip(DH, q):
        T = T @ _mdh(a, d, al, th)
    T = T @ _mdh(0, 0.107, 0, 0)
    c, s = np.cos(TOOL_YAW), np.sin(TOOL_YAW)
    R = T[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return T[:3, 3].copy(), float(np.arctan2(R[1, 0], R[0, 0])), R


def wrap(a, period):
    return (a + period / 2) % period - period / 2


def _points(obs, n):
    c = CAMS[n]
    R, p = np.array(c["mat"]), np.array(c["pos"])
    rgb = np.asarray(obs[f"rgb_{n}"]).astype(np.int16)
    dep = np.asarray(obs[f"depth_{n}"])
    h, w = dep.shape
    m = (rgb[0] > 90) & (rgb[1] < 70) & (rgb[2] < 70)
    ys, xs = np.nonzero(m)
    z = dep[ys, xs]
    f = (h / 2) / np.tan(np.radians(c["fovy"]) / 2)
    xc = (xs + 0.5 - w / 2) / f * z
    yc = -(ys + 0.5 - h / 2) / f * z
    return np.stack([xc, yc, -z], 1) @ R.T + p


def _yaw_minrect(xy):
    if len(xy) < 4:
        return np.nan
    c = xy - xy.mean(0)
    best, bt = None, 0.0
    for t in np.radians(np.arange(0, 90, 1.0)):
        r = c @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        a = np.ptp(r[:, 0]) * np.ptp(r[:, 1])
        if best is None or a < best:
            best, bt = a, t
    return float(bt)


def _block(Q):
    top = Q[:, 2].max()
    T = Q[Q[:, 2] > top - 0.006]
    return {"xy": T[:, :2].mean(0), "z": float(top) - BLOCK_HALF, "yaw": _yaw_minrect(T[:, :2]), "ntop": len(T), "n": len(Q)}


def locate(obs):
    """up to two block estimates {xy, z (centre), yaw (mod 90), ntop, n} from the head and front RGB-D images"""
    P = np.concatenate([_points(obs, n) for n in CAMS])
    P = P[(P[:, 2] > 0.9) & (P[:, 2] < 1.4)]
    if len(P) < 10:
        return []
    a = P[np.argmax(np.linalg.norm(P[:, :2] - P[:, :2].mean(0), axis=1))]
    b = P[np.argmax(np.linalg.norm(P[:, :2] - a[:2], axis=1))]
    if np.linalg.norm(a[:2] - b[:2]) < 0.03:  # one block visible
        return [_block(P)]
    for _ in range(10):
        lab = np.linalg.norm(P[:, :2] - a[:2], axis=1) > np.linalg.norm(P[:, :2] - b[:2], axis=1)
        if lab.all() or (~lab).all():
            return [_block(P)]
        a, b = P[~lab].mean(0), P[lab].mean(0)
    return [_block(P[~lab]), _block(P[lab])]


def _reach(arm, xy):
    return float(np.hypot(xy[0], xy[1] - BASE_Y[arm]))


class HybridS2B:
    carry_queue = False  # the bundle's policy.py resets every episode

    def __init__(self, log=False):
        self.log = log
        self.reset()

    def reset(self, episode=None):
        self.phase, self.count = "start", 0
        self.plan, self.step = [], 0
        self.blk = [None, None]
        self.off = None  # held block centre minus gripper position (world), set when a grasp holds
        self.recoveries, self.t, self.relocate = 0, 0, False
        self.z_rest = Z_TABLE  # the picked block's resting centre height as perceived (table placements use it)
        self.zhist = []
        self.trace = []

    # -- state --------------------------------------------------------------------------------------------------------
    @property
    def arm(self):
        return self.plan[self.step]["arm"] if self.plan else None

    def _grippers(self, obs):
        q = np.asarray(obs["proprioception"], np.float64)
        return [fk(q[0:7], 0), fk(q[9:16], 1)]

    def _holding(self, obs):
        f = float(np.asarray(obs["proprioception_grippers"])[self.arm])
        return 0.15 < f < 0.8

    def _perceive(self, obs, G):
        est = locate(obs)
        if not est:
            return
        pick = self.plan[self.step]["pick"]
        if self.relocate:  # after a miss or a drop, with the gripper raised: re-assign both blocks
            gp = G[self.arm][0]
            if gp[2] > 1.15 and len(est) == 2 and min(e["ntop"] for e in est) >= NTOP_MIN:
                other = 1 - pick
                j = int(np.argmin([np.linalg.norm(e["xy"] - self.blk[other]["xy"]) for e in est]))
                self.blk[other], self.blk[pick] = est[j], est[1 - j]
                self.relocate = False
            return
        for i in (0, 1):
            if self.off is not None and i == pick:
                continue  # held
            b = self.blk[i]
            e = min(est, key=lambda e: np.linalg.norm(e["xy"] - b["xy"]))
            if np.linalg.norm(e["xy"] - b["xy"]) > 0.03 or e["ntop"] < NTOP_MIN:
                continue
            if any(np.linalg.norm(g[0][:2] - b["xy"]) < 0.10 and g[0][2] < b["z"] + 0.16 for g in G):
                continue  # a gripper is low over it: possibly occluded (at approach height its top face stays visible)
            self.blk[i] = e

    def _make_plan(self, G):
        best = None
        for k in (0, 1):
            for i in (0, 1):
                j = 1 - i
                if _reach(k, self.blk[i]["xy"]) > REACH or _reach(k, self.blk[j]["xy"]) > REACH:
                    continue
                cost = np.linalg.norm(G[k][0][:2] - self.blk[i]["xy"]) + np.linalg.norm(self.blk[i]["xy"] - self.blk[j]["xy"])
                if best is None or cost < best[0]:
                    best = (cost, k, i, j)
        if best is not None:
            _, k, i, j = best
            return [dict(arm=k, pick=i, dest=("block", j))]
        # relay: each block goes with the arm it is nearer to; the arm nearer its block goes first
        own = [int(np.argmin([_reach(k, self.blk[i]["xy"]) for k in (0, 1)])) for i in (0, 1)]
        if own[0] == own[1]:  # both nearer the same arm (should not happen when no single arm can do it)
            own[1] = 1 - own[0]
        i = int(np.argmin([np.linalg.norm(G[own[i]][0][:2] - self.blk[i]["xy"]) for i in (0, 1)]))
        k, j = own[i], 1 - i
        R = np.array([np.clip((self.blk[0]["xy"][0] + self.blk[1]["xy"][0]) / 2, 0.45, 0.55), 0.0])
        if np.linalg.norm(R - self.blk[j]["xy"]) < 0.09:  # keep clear of the other block
            R[1] += 0.06 * np.sign(BASE_Y[k])
        return [dict(arm=k, pick=i, dest=("table", R)), dict(arm=1 - k, pick=j, dest=("block", i))]

    def _recover(self):
        self.recoveries += 1
        self.off = None
        self.phase, self.count, self.relocate = "reopen", 0, True

    # -- policy -------------------------------------------------------------------------------------------------------
    def act(self, obs):
        self.t += 1
        a = np.zeros(14, np.float32)
        G = self._grippers(obs)
        if self.phase == "start":  # no-op step; perceive both blocks and plan
            est = locate(obs)
            if len(est) < 2:
                est = (est * 2)[:2] if est else [{"xy": np.array([0.5, 0.1]), "z": Z_TABLE, "yaw": 0.0, "ntop": 0, "n": 0},
                                                 {"xy": np.array([0.5, -0.1]), "z": Z_TABLE, "yaw": 0.0, "ntop": 0, "n": 0}]
            self.blk = list(est)
            self.plan, self.step = self._make_plan(G), 0
            self.phase = "approach"
            return a
        self._perceive(obs, G)
        st = self.plan[self.step]
        k, off = st["arm"], 6 * st["arm"]
        gp, gy, gR = G[k]
        P = self.blk[st["pick"]]
        if st["dest"][0] == "block":
            B = self.blk[st["dest"][1]]
            bp, z_place = np.r_[B["xy"], B["z"]], B["z"] + 0.042
        else:
            bp, z_place = np.r_[st["dest"][1], self.z_rest], self.z_rest + 0.002  # same estimator as pp: biases cancel
        pp = gp + self.off if self.off is not None else np.r_[P["xy"], P["z"]]
        py = P["yaw"]
        grip = 0.0
        holding = self._holding(obs)
        if self.phase == "approach":
            target, yaw_ref = np.r_[pp[:2], pp[2] + H_APPROACH], py
            if np.linalg.norm(target - gp) < 0.012 and abs(wrap(py - gy, np.pi / 2)) < np.radians(6):
                self.phase = "descend"
        elif self.phase == "descend":
            target, yaw_ref = np.r_[pp[:2], pp[2] + H_GRASP], py
            if abs(wrap(py - gy, np.pi / 2)) > np.radians(12) and gp[2] < pp[2] + H_GRASP + 0.02:
                self.phase = "approach"
            elif abs(gp[2] - target[2]) < 0.006 and np.linalg.norm(gp[:2] - pp[:2]) < 0.01:
                self.phase, self.count = "grasp", 0
        elif self.phase == "grasp":
            target, yaw_ref, grip = gp, None, 1.0
            self.count += 1
            if self.count >= 10:
                if holding:
                    rel = gR.T @ (pp - gp)  # block centre in the gripper frame; the fingers centre it along y
                    rel[1] = 0.0
                    self.off = gR @ rel
                    self.z_rest = float(pp[2])
                    self.phase = "lift"
                else:
                    self._recover()
        elif self.phase == "reopen":  # open and rise to approach height before re-approaching
            target, yaw_ref, grip = np.r_[gp[:2], pp[2] + H_APPROACH], None, 0.0
            self.count += 1
            if self.count >= 6 and gp[2] > pp[2] + H_APPROACH - 0.03 and not self.relocate:
                self.phase = "approach"
            elif self.count >= 40:
                self.relocate = False
                self.phase = "approach"
        elif self.phase in ("lift", "carry", "lower") and not holding:
            self._recover()
            target, yaw_ref, grip = gp, None, 0.0
        elif self.phase == "lift":
            target, yaw_ref, grip = np.r_[gp[:2], Z_TABLE + H_CARRY], None, 1.0
            if gp[2] > Z_TABLE + H_CARRY - 0.012:
                self.phase = "carry"
        elif self.phase == "carry":
            target, yaw_ref, grip = np.r_[gp[:2] + (bp[:2] - pp[:2]), Z_TABLE + H_CARRY], None, 1.0
            if np.linalg.norm(pp[:2] - bp[:2]) < 0.008:
                self.phase = "lower"
        elif self.phase == "lower":
            target, yaw_ref, grip = np.r_[gp[:2] + (bp[:2] - pp[:2]), gp[2] - (pp[2] - z_place)], None, 1.0
            self.zhist.append(gp[2])
            blocked = len(self.zhist) > 6 and self.zhist[-7] - gp[2] < 0.002 and pp[2] < z_place + 0.03  # in contact
            if pp[2] < z_place + 0.004 or blocked:
                self.phase, self.count, self.zhist = "release", 0, []
        elif self.phase == "release":
            target, yaw_ref, grip = gp, None, 0.0
            self.count += 1
            if self.count >= 10:
                if self.step + 1 < len(self.plan):  # relay: leave the block, clear the way
                    i = st["pick"]
                    self.blk[i] = {"xy": pp[:2].copy(), "z": Z_TABLE, "yaw": py, "ntop": 0, "n": 0}
                    self.off = None
                    self.phase, self.count = "clear", 0
                else:
                    self.phase = "retreat"
        elif self.phase == "clear":  # up, then home; the next step starts once the arm is out of the way
            target, yaw_ref, grip = (np.r_[gp[:2], Z_TABLE + H_CARRY] if gp[2] < Z_TABLE + H_CARRY - 0.03 else HOME[k]), None, 0.0
            self.count += 1
            if np.linalg.norm(gp - HOME[k]) < 0.04 or self.count > 70:
                self.step += 1
                self.phase = "approach"
        else:  # retreat
            target, yaw_ref, grip = np.r_[gp[:2], Z_TABLE + H_CARRY], None, 0.0
        a[off:off + 3] = np.clip(KP * (target - gp), -VMAX, VMAX)
        if yaw_ref is not None and np.isfinite(yaw_ref):
            a[off + 5] = np.clip(KR * wrap(yaw_ref - gy, np.pi / 2), -RMAX, RMAX)
        a[12 + k] = grip
        if self.log:
            self.trace.append((self.t, self.step, self.phase, gp.round(3).tolist(), pp.round(3).tolist(), bp.round(3).tolist(),
                               float(np.asarray(obs["proprioception_grippers"])[k])))
        return a
