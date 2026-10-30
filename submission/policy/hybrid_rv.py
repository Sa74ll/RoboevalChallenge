"""Hybrid RotateValve controller (Oct 28): classical RGB-D perception + scripted bimanual control, using only the
challenge's standard observations (nothing is read from the simulator at test time).

Settle: the valves are free bodies (250 kg). They drop ~2.7 mm onto the table in the first step and, in ~1-2% of seeds
where they spawn overlapping, shove each other apart for several steps. The robot stays still (grippers closed) until
the valve region of the head and front depth images stops changing (typically 8-13 steps).
Perception (rv_perception.py): a point model of the valve (surface samples of the released valve asset) is registered to
the head and front depth points: 3 unknowns per valve (x, y, yaw) at the settled height; search space = the task's spawn
ranges (or, if the valves were seen moving, the arm's whole half of the table and any yaw); score = depth points on the
posed model minus model points the cameras see through. Output: each wheel's centre (median error ~1 mm).
Gripper poses: forward kinematics of the Panda arms from proprioception (hybrid_s2b.fk).

Control (per arm; the left arm takes the wheel with y > 0, the right arm the other; both arms work at the same time):
  reach    open the gripper and move the flange above the wheel centre, pads 2 cm above the wheel (minimum-jerk, both
           arms with the same duration, so heights and speeds stay matched);
  descend  once the measured flange is within 4 mm (xy) of the wheel axis and the fingers are open: lower the pads
           around the rim (minimum-jerk); wait until the measured height is within 1.5 mm;
  close    close the gripper; check the finger opening (a rim grasp reads ~0.7, a miss ~0): on a miss re-open, rise,
           descend again (at most twice);
  turn     rotate the gripper about the vertical through the wheel axis by 36 deg (counterclockwise seen from above, the
           valve's positive direction; success needs state > 0.10 = 27 deg of the 270 deg joint range);
  hold     keep the grasp; the episode ends at success. If it has not ended 1.2 s later (a slipped grasp), turn 20 deg
           further (up to 3 times).
Commands: feedforward increments of the reference trajectory; a bounded position correction (gravity sag) only while the
reference is stationary (integrating the servo lag during motion overshoots and jams the pads on the valve base).
Actions: 14-D end-effector deltas [dp_L dr_L | dp_R dr_R | g_L g_R] (world frame), grippers 0 open / 1 closed.
"""
import numpy as np

try:
    from .hybrid_s2b import fk
    from . import rv_perception as rvp
except ImportError:
    from hybrid_s2b import fk
    import rv_perception as rvp

# ---- geometry (valve asset handwheel_valve.xml; Panda hand) ---------------------------------------------------------
Z_JOINT = rvp.Z_ROOT_SETTLED + 0.136595  # wheel joint height once the valve rests on the table (spawned at root z 0.94,
# it drops 2.7 mm in the first step); the joint is 0.1366 above the valve root
PAD_DOWN = 0.1034  # finger pad centre below the flange (attachment site) with the gripper pointing down
Z_PAD_REL = 0.015  # pad centre height relative to the wheel joint (rim spans +0.005..+0.015; pad +-0.012, so its bottom
# stays 5.5 mm above the valve base's convex collision hull, whose top is at most -0.0025 at the rim radius)
Z_GRASP = Z_JOINT + Z_PAD_REL + PAD_DOWN  # flange height at the grasp
DT = 0.05  # control period (20 Hz)
G_RIM = (0.45, 0.92)  # finger opening (proprioception_grippers) that means the rim is between the pads
SETTLE_MAX = 40  # steps: start anyway if the valves are still moving then


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def _rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    a = np.arccos(c)
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (a / (2 * np.sin(a)))


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class _Seg:
    """a smooth segment: xy, z and yaw angle each go from a start to an end value with a minimum-jerk profile over their
    own step windows (start, duration), relative to step i0"""

    def __init__(self, i0, p0, p1, a0, a1, xy=(0, 1), z=(0, 1), ang=(0, 1)):
        self.i0, self.p0, self.p1, self.a0, self.a1 = i0, np.array(p0, float), np.array(p1, float), a0, a1
        self.xy, self.z, self.ang = xy, z, ang

    @staticmethod
    def _s(j, w):
        return minjerk((j - w[0]) / max(w[1], 1e-9))

    def at(self, i):
        j = i - self.i0
        p = self.p0.copy()
        p[:2] += self._s(j, self.xy) * (self.p1[:2] - self.p0[:2])
        p[2] += self._s(j, self.z) * (self.p1[2] - self.p0[2])
        return p, self.a0 + self._s(j, self.ang) * (self.a1 - self.a0)

    def end(self):
        return self.i0 + max(self.xy[0] + self.xy[1], self.z[0] + self.z[1], self.ang[0] + self.ang[1])


class HybridRV:
    carry_queue = False  # the bundle's policy.py resets every episode

    def __init__(self, log=False, ta=0.5, tb=5.0, t_desc=0.8, t_close=10, t_turn=1.6, turn_deg=36.0, extra_deg=20.0,
                 t_wait=1.2, vmax=0.03, kp=0.25, fb_max=0.015, z_pre=0.035, xy_tol=0.004, z_tol=0.0015):
        self.log = log
        self.ta, self.tb, self.t_desc, self.xy_tol, self.z_tol = ta, tb, t_desc, xy_tol, z_tol
        self.t_close, self.t_turn, self.t_wait = t_close, t_turn, t_wait
        self.turn, self.extra = np.radians(turn_deg), np.radians(extra_deg)
        self.vmax, self.kp, self.fb_max, self.z_pre = vmax, kp, fb_max, z_pre
        self.wheels_override = None  # analysis only: a test harness may set ground-truth wheel centres here
        self.model = None  # valve point model, loaded on first use
        self.reset()

    def reset(self, episode=None):
        self.t = 0
        self.wheels, self.est = None, None
        self.arms = None
        self.fb = np.zeros((2, 3))  # accumulated position correction per arm (bounded)
        self.prev, self.changes, self.t_start = None, [], None
        self.trace = []

    # -- perception -------------------------------------------------------------------------------------------------
    def _perceive(self, obs, wide=False):
        if self.model is None:
            self.model = rvp.ValveModel()
        self.est = rvp.locate_valves(obs, self.model, z_root=rvp.Z_ROOT_SETTLED, wide=wide)
        W = []
        for s, e in enumerate(self.est):
            if e is None:  # nothing seen in that slot: the middle of its spawn range
                W.append(np.array([0.50, 0.30 if s == 0 else -0.30, Z_JOINT]))
            else:
                W.append(np.asarray(e["wheel"], float))
        return W

    # -- planning ---------------------------------------------------------------------------------------------------
    def _start(self, G, i):
        W = sorted(self.wheels, key=lambda w: -w[1])  # slot 0 = left (y > 0) -> left arm
        D = max(float(np.linalg.norm(W[k][:2] - G[k][0][:2])) for k in (0, 1))
        T = max(0.7, self.ta + self.tb * D)  # same reach duration for both arms (synchronised heights and speeds)
        n_xy = int(round(T / DT))
        self.arms = []
        for k in (0, 1):
            p0 = G[k][0]
            tgt = np.array([W[k][0], W[k][1], Z_GRASP])
            app = np.array([W[k][0], W[k][1], Z_GRASP + self.z_pre])  # above the wheel: pads clear its top
            seg = _Seg(i, p0, app, 0.0, 0.0, xy=(0, n_xy), z=(0, n_xy), ang=(0, 1))
            self.arms.append(dict(phase="reach", seg=seg, tgt=tgt, grip=0.0, i_phase=i, ang=0.0, tries=0, extra=0))

    def _phase(self, k, i, g, gp):
        """advance arm k's phase machine at step i (g: its finger opening, gp: its flange position)"""
        A = self.arms[k]
        seg = A["seg"]
        exy = np.linalg.norm(gp[:2] - A["tgt"][:2])
        if A["phase"] == "reach" and i >= seg.end() - 1 and g > 0.9 and (
                exy < self.xy_tol or (i - seg.end() > 40 and exy < 0.010)):  # (a target at the edge of the reach)
            # above the wheel with the fingers open: descend around the rim
            p, a = seg.at(i)
            A.update(phase="descend", i_phase=i, seg=_Seg(i, p, A["tgt"], a, a, xy=(0, 1), z=(0, int(round(self.t_desc / DT)))))
        elif A["phase"] == "descend" and i >= seg.end() and (
                (abs(gp[2] - A["tgt"][2]) < self.z_tol and np.linalg.norm(gp[:2] - A["tgt"][:2]) < self.xy_tol)
                or i - seg.end() >= 20):
            A.update(phase="close", grip=1.0, i_phase=i)  # at the grasp height (the lag has settled): close
        elif A["phase"] == "close" and i - A["i_phase"] >= self.t_close:
            p, a = seg.at(i)
            if G_RIM[0] <= g <= G_RIM[1] or A["tries"] >= 2:
                A.update(phase="turn", i_phase=i,
                         seg=_Seg(i, p, p, A["ang"], A["ang"] + self.turn, ang=(0, int(round(self.t_turn / DT)))))
            else:  # missed the rim: open, rise, descend again
                A["tries"] += 1
                up = A["tgt"] + np.array([0, 0, self.z_pre])
                A.update(phase="retry", grip=0.0, i_phase=i, seg=_Seg(i, p, up, a, a, xy=(0, 10), z=(0, 10)))
        elif A["phase"] == "retry" and i >= seg.end() and g > 0.9:
            A.update(phase="reach", grip=0.0, i_phase=i)  # the retry segment ends above the wheel: descend again
        elif A["phase"] == "turn" and i >= seg.end():
            A.update(phase="hold", i_phase=i, ang=seg.a1)
        elif A["phase"] == "hold" and i - A["i_phase"] >= int(round(self.t_wait / DT)) and A["extra"] < 3:
            # still running: some valve has not passed 0.10 (slipped grasp): turn further
            A["extra"] += 1
            p, a = seg.at(i)
            A.update(phase="turn", i_phase=i,
                     seg=_Seg(i, p, p, A["ang"], A["ang"] + self.extra, ang=(0, int(round(0.8 / DT)))))

    # -- policy -------------------------------------------------------------------------------------------------------
    def act(self, obs):
        q = np.asarray(obs["proprioception"], np.float64)
        gq = np.asarray(obs["proprioception_grippers"], np.float64)
        G = [fk(q[0:7], 0), fk(q[9:16], 1)]
        a = np.zeros(14, np.float32)
        i = self.t
        if self.arms is None:
            # settle: the valves are free bodies; they drop onto the table in the first step and, if they spawned
            # overlapping, push each other apart. Keep the robot still (grippers closed) until the valve region of the
            # depth images stops changing, then localise them and plan.
            if self.prev is not None:
                self.changes.append(rvp.valve_region_change(self.prev, obs))
            self.prev = {c: np.array(obs[c]) for c in ("depth_head", "depth_front")}
            # after the first-step drop only a few edge pixels flicker (0-2 per step); colliding valves change thousands
            settled = len(self.changes) >= 3 and self.changes[-1] <= 5 and self.changes[-2] <= 5
            if not (settled or i >= SETTLE_MAX):
                a[12:] = 1.0
                self.t += 1
                return a
            if self.wheels_override is not None:
                self.wheels = [np.asarray(w, float) for w in self.wheels_override]
            else:
                moved = any(c > 300 for c in self.changes[2:])  # still moving after the first-step drop
                self.wheels = self._perceive(obs, wide=moved)
            self.t_start = i
            self._start(G, i)
        for k in (0, 1):
            self._phase(k, i, float(gq[k]), G[k][0])
            A = self.arms[k]
            gp = G[k][0]
            p0, a0 = A["seg"].at(i)
            p1, a1 = A["seg"].at(i + 1)
            if np.linalg.norm(p1 - p0) < 2e-4:  # stationary reference: correct the steady error (gravity sag)
                fb = np.clip(self.fb[k] + self.kp * (p0 - gp), -self.fb_max, self.fb_max) - self.fb[k]
                self.fb[k] += fb
            else:  # moving: feedforward only (integrating the servo lag would overshoot)
                fb = np.zeros(3)
            dp = (p1 - p0) + fb
            dr = _rotvec(_rz(a1) @ _rz(a0).T)  # yaw increment about the world vertical
            a[6 * k:6 * k + 3] = np.clip(dp, -self.vmax, self.vmax)
            a[6 * k + 3:6 * k + 6] = np.clip(dr, -0.15, 0.15)
            a[12 + k] = A["grip"]
        if self.log:
            self.trace.append((i, [A["phase"] for A in self.arms], [G[k][0].round(4).tolist() for k in (0, 1)],
                               gq.round(2).tolist()))
        self.t += 1
        return a

    @property
    def phases(self):
        return [A["phase"] for A in self.arms] if self.arms else None

    @property
    def tries(self):
        return [A["tries"] for A in self.arms] if self.arms else None
