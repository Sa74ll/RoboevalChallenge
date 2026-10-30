"""Scripted LiftPot expert (Oct 24) for synthetic demonstrations, like s2b_expert.py: reads the pot's and grippers' poses in
simulation and acts through the policy's own interface (14-D end-effector deltas, grippers 0 = open, 1 = closed). Both arms
at once: move above the handle, descend, close, lift together. Handle grasp poses in the pot's frame come from 48 grasps of
the incumbent's successful TRAINING-seed episodes replayed exactly (lp_calib.py): left (-3.3, +23.7, +12.3) cm, right
(-2.4, -25.4, +11.9) cm (gripper body above the pot origin). The incumbent's grasp yaws do not follow the pot's yaw (left
-113 to -168 deg in the world whatever the pot's yaw), and a yaw past the arm's reach stalls the controller (left arm stuck
at -132 deg), so the expert holds a fixed world yaw per arm: the medians of those grasps, left -129, right -50 deg.
Success: pot 10 cm up, tilt < 20 deg, clear of the cabinets.
Usage (test): python lp_expert.py FIRST_SEED N   -> success rate, steps"""
import json
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

# Pot geometry (collision meshes, pot frame): each handle is a bar along the pot's x axis (x -6..+6 cm, y +23.4 / -24.6 cm,
# z +3..+5 cm) held by two support arms along y (x +-7.5 cm, |y| 15..23 cm, z +1..+4.5 cm). The fingers close along the
# gripper's body y axis (world x at the start yaw of -90 deg). Two grasps work (both seen in the incumbent's successes):
# across the bar (fingers along the pot's y; rel. yaw 0 mod 180, up to ~45 deg off) and across a support arm (fingers along
# the pot's x; rel. yaw 90 mod 180, e.g. seed 330039). Per arm, the expert takes the grasp needing the least wrist rotation
# from the start yaw (a yaw past the arm's reach stalls the controller).
Y_BAR = {0: 0.234, 1: -0.246}
Y_ARM = {0: 0.205, 1: -0.205}
Z_GRASP = 0.122  # gripper body above the pot origin at a grasp (calibration median)
X_ARM = 0.075
Y_START = np.radians(-90)
H_ABOVE, H_LIFT, X_REACH = 0.06, 0.16, 0.70
LIB = json.load(open(Path(__file__).resolve().parent / "lp_grasps.json"))
VMAX, KP, RMAX, KR, VMAX_DESCEND = 0.010, 0.6, 0.05, 0.8, 0.006


def yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rotvec(q_target, q_now):
    """World-frame rotation vector taking q_now to q_target (shortest way)."""
    e = q_target * q_now.inverse
    if e.w < 0:
        e = -e
    v = np.array([e.x, e.y, e.z])
    n = np.linalg.norm(v)
    return np.zeros(3) if n < 1e-9 else v / n * 2 * np.arctan2(n, e.w)


def wrap(a, period):
    return (a + period / 2) % period - period / 2


class Expert:
    def __init__(self, env):
        self.u = getattr(env, "unwrapped", env)
        self.sides = list(self.u.robot.grippers)
        self.arm = None  # both arms act; kept for s2b_gen-style noise code

    def reset(self):
        self.phase, self.count, self.hold, self.rel = "start", 0, None, None

    def g(self, k):
        b = self.u.robot.grippers[self.sides[k]].body
        return np.array(b.get_position()), Quaternion(b.get_quaternion())

    def plan(self):
        """Grasp per arm = the median of the incumbent's 15 library grasps (lp_grasps.json: 520 grasps from 300 successful
        TRAINING-seed episodes replayed exactly) with the nearest pot yaw: position in the pot's frame and world yaw (the
        left wrist sits near -136 deg whatever the pot's yaw, the right one follows it: -46 + 0.79 x pot yaw)."""
        pot = self.u.kitchenpot.body
        py = np.degrees(yaw(np.array(pot.get_quaternion())))
        self.rel, self.rq, self.kind = {}, {}, {}
        for k in (0, 1):
            g = [x for x in LIB if x["side"] == k]
            # one real grasp (a median would mix the bar and support-arm grasps into a pose that fits neither)
            x = min(g, key=lambda x: abs(x["pot_yaw_deg"] - py))
            self.rel[k] = np.array(x["rel"])
            self.rq[k] = Quaternion(x["rel_quat"])
            self.kind[k] = f"{x['seed']}"

    def targets(self):
        if self.rel is None:
            self.plan()
        pot = self.u.kitchenpot.body
        pp, pq = np.array(pot.get_position()), Quaternion(pot.get_quaternion())
        return {k: (pp + pq.rotate(self.rel[k]), pq * self.rq[k]) for k in (0, 1)}

    def act(self):
        a = np.zeros(14, np.float32)
        if self.phase == "start":  # the start no-op
            self.phase = "approach"
            return a
        tg = self.targets()
        gs = {k: self.g(k) for k in (0, 1)}
        grip, tgt, yref = 0.0, {}, {}
        if self.phase == "approach":  # travel at or above the pre-grasp height, so the fingers clear the rim on the way
            for k in (0, 1):
                tgt[k], yref[k] = tg[k][0] + np.r_[0, 0, H_ABOVE], tg[k][1]
                if np.linalg.norm(tgt[k][:2] - gs[k][0][:2]) > 0.02:
                    tgt[k] = np.r_[tgt[k][:2], max(tgt[k][2], gs[k][0][2])]
            if all(np.linalg.norm(tgt[k] - gs[k][0]) < 0.012 and np.linalg.norm(rotvec(yref[k], gs[k][1])) < np.radians(6) for k in (0, 1)):
                self.phase = "descend"
        elif self.phase == "descend":
            for k in (0, 1):
                tgt[k], yref[k] = tg[k][0], tg[k][1]
            if all(np.linalg.norm(tgt[k] - gs[k][0]) < 0.008 for k in (0, 1)):
                self.phase, self.count = "grasp", 0
                self.hold = {k: gs[k][0].copy() for k in (0, 1)}
        elif self.phase == "grasp":
            grip = 1.0
            for k in (0, 1):
                tgt[k], yref[k] = self.hold[k], None
            self.count += 1
            if self.count >= 10:
                self.phase = "lift"
        else:  # lift both together
            grip = 1.0
            for k in (0, 1):
                tgt[k], yref[k] = self.hold[k] + np.r_[0, 0, H_LIFT], None
        vmax = VMAX_DESCEND if self.phase == "descend" else VMAX
        for k in (0, 1):
            off = 6 * k
            gp, gy = gs[k]
            a[off:off + 3] = np.clip(KP * (tgt[k] - gp), -vmax, vmax)
            if yref[k] is not None:  # rotation deltas act in the world frame (rot_conv test)
                a[off + 3:off + 6] = np.clip(KR * rotvec(yref[k], gy), -RMAX, RMAX)
            a[12 + k] = grip
        return a


if __name__ == "__main__":
    from roboeval.challenge.submission import Interface, make_env
    first, n = int(sys.argv[1]), int(sys.argv[2])
    env = make_env("LiftPot", "PositionAndOrientation", Interface(absolute=False, ee=True, control_frequency=20))
    ex = Expert(env)
    res = []
    for s in range(first, first + n):
        env.reset(seed=s)
        ex.reset()
        term = "budget"
        for t in range(200):
            _, _, terminated, truncated, _ = env.step(ex.act())
            if env.success:
                term = "success"; break
            if terminated or truncated:
                term = "fail"; break
        pot = ex.u.kitchenpot.body
        res.append((term == "success", t + 1, ex.phase))
        print(f"seed {s}: {term} at {t + 1}, last phase {ex.phase}, pot z {pot.get_position()[2]:.3f}, grasps {ex.kind}", flush=True)
    ok = [r for r in res if r[0]]
    print(f"success {len(ok)}/{len(res)}; steps median {np.median([r[1] for r in ok]) if ok else -1:.0f}; "
          f"failures end in {dict(zip(*np.unique([r[2] for r in res if not r[0]], return_counts=True))) if len(ok) < len(res) else {}}")
