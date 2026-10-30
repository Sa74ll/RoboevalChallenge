"""Hybrid LiftTray controller (Oct 28): classical RGB-D perception + a scripted, synchronized two-arm grasp and lift, using
only the challenge's standard observations (nothing is read from the simulator at test time).

Task (roboeval/envs/lift_tray.py, LiftTrayPositionAndOrientation): a 34 kg wooden tray (0.55 x 0.33 m, end walls 9.4 cm
tall, 2.4 cm thick) lies on the table with its centre at robot-frame x 0.55, y in {-0.175, -0.025, 0.125} (+-5 mm), its
long axis within 30 deg of the y axis. Success: both grippers hold the tray (both finger pads of each in contact), the
tray does not touch the table and its body is at least 5 cm above init_z_pos. The env's settle step leaves the tray 5.5 cm
up at reset (it drops in the first control step) and init_z_pos is taken there, so the tray must rise ~7.8 cm from rest.
Budget 200 steps (10 s).

Frames: everything is in the robot frame of hybrid_s2b.fk (the scene puts the robot base at world (0.1, -0.3, 0); the
head and front cameras are mounted on it, so hybrid_s2b.CAMS holds unchanged).

Perception (lt_perc.py): the tray's visual mesh registered to the head + front depth points at the first step (centre
xy, long-axis angle). Accuracy on 160 training seeds (first observation): handle points within 4.2 mm (median 2.7),
axis within 0.65 deg.

Plan: each arm takes the end wall on its side (left arm: the +y end) at the wall's middle, top-down, fingers across the
wall (gripper y axis along the tray's long axis), pads 2.1-4.5 cm below the wall top (the fingers open 8 cm around the
2.4 cm wall: 2.8 cm of clearance per side).

Control (per step, both arms): position delta = KP x (target - fk position), capped at VMAX per step; in joint phases the
arm with less to go is capped proportionally lower, so both arrive together and stay at the same height (the scorer
compares the arms' joint speeds, bavd, and wrist heights, bgvd). Yaw: KR x error, capped at RMAX. The env adds each delta
to its commanded pose, so in the saturated lift the command runs 0.2-0.4 m ahead of the hands; that lead is what moves
the arms into a configuration that lifts (capping it at 8 or 12 cm: 2/20 and 12/20 successes on training seeds).
Phases (joint): approach (above the walls, yaw aligned, fingers open) -> descend -> grasp (close; both holding = finger
opening 0.15-0.8 for 2 steps) -> lift: the hands pull the tray toward the robot while lifting (to tray centre x X_PULL,
+LIFT in z). Pulling is what makes the lift possible: holding 17 kg per hand at the handles' initial distance saturates
both shoulders (joint 2, 87 N m) and the wrists (12 N m) and the tray only tilts; ~20 cm closer the shoulders can lift
it. The episode ends at success during this move (median 70 steps, 3.5 s).
Recovery (always both arms, so the lift starts from a simultaneous grasp): a blocked descent (finger on the wall top), a
missed grasp (closed on nothing, or no hold after 20 steps) or a tray lost during the lift (set down first) -> both
open and rise, the tray is perceived again (12 cm around each low gripper masked) and both grasp again; a lift with no
success after STALL steps is lowered back onto the table (grasp kept) and repeated from there, pulling to X_PULL2.
Actions: 14-D end-effector deltas [dp_L dr_L | dp_R dr_R | g_L g_R] (world-frame axes = robot-frame axes), grippers 0 open."""
import numpy as np

try:
    from .hybrid_s2b import fk, wrap
    from .lt_perc import locate_tray
except ImportError:
    from hybrid_s2b import fk, wrap
    from lt_perc import locate_tray

HANDLE = 0.266  # end-wall centre distance from the tray centre along the long axis (collision walls at 0.254-0.278)
WALL_TOP = 0.970  # end-wall top height with the tray resting (collision hull; visual 0.967)
GRASP_DZ = 0.075  # gripper origin above the wall top when grasping: pads (0.091-0.115 below the origin) at 2.1-4.5 cm
# below the wall top, the palm 1.7 cm above it
APP_DZ = 0.06  # approach point above the grasp height, then straight down
LIFT = 0.12  # hand lift after the grasp
X_PULL = 0.345  # ... while the hands pull the tray centre toward the robot to this x (an absolute target: the hob's near
# edge is at x 0.343, so a tray that is set down again never ends beyond it; from the spawn x 0.55 this is a 20 cm pull)
X_PULL2 = 0.30  # if no success STALL steps into a lift: the tray is lowered back onto the table (grasp kept) and pulled
# again from there, closer (a stalled lift with the arms stretched and saturated does not recover by pulling harder)
STALL = 55  # (lift steps; normal lifts end after 30-46 steps)
VMAX, KP, RMAX, KR = 0.012, 0.6, 0.06, 0.8
DEFAULTS = {k: globals()[k] for k in ("HANDLE", "WALL_TOP", "GRASP_DZ", "APP_DZ", "LIFT", "X_PULL", "X_PULL2", "STALL",
                                      "VMAX", "KP", "RMAX", "KR")}
FALLBACK = {"c": np.array([0.55, -0.025]), "th": np.pi / 2}  # spawn-grid centre, if the tray is never found


def _plausible(c):
    """a tray centre on the table in front of the robot (spawn: x 0.55, y -0.175..0.125, +-5 mm)"""
    return 0.35 < c[0] < 0.75 and -0.45 < c[1] < 0.35


def _opening(obs, k):
    return float(np.asarray(obs["proprioception_grippers"])[k])


class HybridLT:
    carry_queue = False  # the bundle's policy.py resets every episode

    def __init__(self, log=False, **cfg):
        """cfg: overrides of the module constants in DEFAULTS (from checkpoint/LiftTray/hybrid.json)"""
        self.log = log
        bad = set(cfg) - set(DEFAULTS)
        assert not bad, f"unknown hybrid_lt options {sorted(bad)}"
        for k, v in {**DEFAULTS, **cfg}.items():
            setattr(self, k, float(v))
        self.oracle = None  # development only (analysis harness): callable -> (centre xy, axis angle); never set in a bundle
        self.reset()

    def reset(self, episode=None):
        self.t = 0
        self.tray = None
        self.est = []
        self.phase = ["start", "start"]  # per arm: approach, descend, grasp, ready, lift (relower), retry (setdown)
        self.count = [0, 0]
        self.hold_n = [0, 0]
        self.retries = [0, 0]
        self.why = []  # (step, arm, reason) of each retry
        self.zh = [[], []]  # heights during the descent (blocked-descent check)
        self.lost_n = [0, 0]  # consecutive lift steps with the fingers closed on nothing
        self.setdown = False  # a hand lost the tray during the lift: lower it back onto the table before reopening
        self.lift_t = None
        self.lifts = 0  # lift attempts in this grasp (a stalled lift is lowered and repeated, at most 3 in all)
        self.lift_c = None  # tray centre at the start of the current lift
        self.g0 = None
        self.trace = []

    # -- perception and plan -----------------------------------------------------------------------------------------
    def _plan(self, c, th):
        u = np.array([np.cos(th), np.sin(th)])
        if u[1] < 0:
            u = -u
        phi = float(np.arctan2(u[1], u[0])) - np.pi / 2  # gripper yaw: its y axis (finger closing) along the tray axis
        self.tray = {"c": np.asarray(c, float), "u": u, "h": [c + self.HANDLE * u, c - self.HANDLE * u], "phi": phi}

    def _perceive(self, obs, G=None):
        if self.oracle is not None:
            c, th = self.oracle()
            return {"c": np.asarray(c, float), "th": float(th), "score": 1.0}
        # grippers low over the tray (retries) would add finger points to the tray's height band: mask them
        mask = [g[0][:2] for g in G if g[0][2] < 1.12] if G is not None else []
        return locate_tray(obs, mask_xy=mask)

    def _at(self, k, G, target, yaw_tol, pos_tol, need_open, op):
        return (np.linalg.norm(target - G[k][0]) < pos_tol and abs(wrap(self.tray["phi"] - G[k][1], np.pi)) < yaw_tol
                and (op[k] > 0.7 or not need_open))

    def _retry(self, k, why):
        self.phase[k], self.count[k] = "retry", 0
        self.retries[k] += 1
        self.why.append((self.t, k, why))

    # -- policy -------------------------------------------------------------------------------------------------------
    def act(self, obs):
        self.t += 1
        a = np.zeros(14, np.float32)
        q = np.asarray(obs["proprioception"], np.float64)
        G = [fk(q[9 * k:9 * k + 7], k) for k in (0, 1)]
        op = [_opening(obs, k) for k in (0, 1)]
        if self.tray is None:
            e = self._perceive(obs)
            if e is not None:
                self.est.append(e)
            # plan on the first plausible estimate (score 0.91-1.0 on 160 training seeds); else look again for up to 3
            # more steps (the robot waits), then take the best-scoring one
            if e is None or e["score"] < 0.8 or not _plausible(e["c"]):
                if self.t < 4:
                    return a
                e = max(self.est, key=lambda x: x["score"]) if self.est else FALLBACK
            self._plan(e["c"], e["th"])
            self.phase = ["approach", "approach"]
        z_grasp = self.WALL_TOP + self.GRASP_DZ
        P_app = [np.r_[self.tray["h"][k], z_grasp + self.APP_DZ] for k in (0, 1)]
        P_gr = [np.r_[self.tray["h"][k], z_grasp] for k in (0, 1)]
        T, grip, yaw = [None, None], [0.0, 0.0], [self.tray["phi"]] * 2
        for k in (0, 1):
            ph, gp = self.phase[k], G[k][0]
            self.count[k] += 1
            if ph == "approach":
                T[k] = P_app[k]
            elif ph == "descend":
                T[k] = P_gr[k]
                self.zh[k].append(gp[2])
                stuck = len(self.zh[k]) > 8 and self.zh[k][-7] - gp[2] < 0.002
                if stuck and gp[2] > z_grasp + 0.015:  # a finger on the wall top (or the tray moved): look again
                    self._retry(k, "blocked")
            elif ph in ("grasp", "ready"):
                T[k], grip[k] = P_gr[k], 1.0
                self.hold_n[k] = self.hold_n[k] + 1 if 0.15 < op[k] < 0.8 else 0
                if ph == "grasp" and self.hold_n[k] >= 2 and self.count[k] >= 4:
                    self.phase[k] = "ready"
                elif ph == "grasp" and ((self.count[k] >= 12 and op[k] <= 0.15) or self.count[k] >= 20):
                    self._retry(k, "missed")  # closed on nothing, or never got a hold
                elif ph == "ready" and op[k] <= 0.15:
                    self._retry(k, "lost")
            elif ph == "setdown":
                T[k] = gp  # (set below)
            elif ph == "retry":  # open, back up above the wall (re-perceive), approach again
                T[k], grip[k] = np.r_[gp[:2], z_grasp + self.APP_DZ + 0.03], 0.0
            elif ph in ("lift", "relower"):
                if ph == "lift":
                    xc = self.X_PULL if self.lifts == 1 else self.X_PULL2
                    dx = max(0.0, self.lift_c[0] - xc)
                    T[k], grip[k] = self.g0[k] + np.array([-dx, 0.0, self.LIFT]), 1.0
                    if self.t - self.lift_t >= self.STALL and self.lifts < 3:  # no success: lower the tray, pull again
                        self.phase, self.count = ["relower", "relower"], [0, 0]
                        self.why.append((self.t, k, "stalled"))
                else:  # relower: keep the grasp, put the tray back on the table (hands to the grasp height)
                    T[k], grip[k] = np.r_[gp[:2], self.g0[k][2]], 1.0
                self.lost_n[k] = self.lost_n[k] + 1 if op[k] <= 0.15 else 0
                if self.lost_n[k] >= 3:  # the wall slipped out of this hand: set the tray down, look again, start over
                    self.why.append((self.t, k, "dropped"))
                    self.retries[k] += 1
                    self.setdown = True
        if self.setdown:  # both hands back down to the grasp height (one still holding lowers its end), then a joint retry
            if self.phase[0] != "setdown":
                self.phase, self.count = ["setdown", "setdown"], [0, 0]
            for k in (0, 1):
                T[k], grip[k] = np.r_[G[k][0][:2], z_grasp], (1.0 if 0.15 < op[k] < 0.8 else 0.0)
            if (self.count[0] > 3 and all(abs(G[j][0][2] - z_grasp) < 0.012 for j in (0, 1))) or self.count[0] > 25:
                self.setdown = False
                self.phase, self.count = ["retry", "retry"], [0, 0]
                for k in (0, 1):
                    T[k], grip[k] = np.r_[G[k][0][:2], z_grasp + self.APP_DZ + 0.03], 0.0
        if "retry" in self.phase:
            # a retry is always joint: both arms open, rise, the tray is perceived again and both grasp together (a lift
            # started from one hand that held the tray alone for a while tilted that wrist to 57 deg and stalled)
            for k in (0, 1):
                if self.phase[k] != "retry":
                    self.phase[k], self.count[k] = "retry", 0
                    T[k], grip[k] = np.r_[G[k][0][:2], z_grasp + self.APP_DZ + 0.03], 0.0
            up = all(self.count[k] >= 6 and op[k] > 0.7 and abs(G[k][0][2] - T[k][2]) < 0.02 for k in (0, 1))
            if up or min(self.count) > 40:
                e = self._perceive(obs, G)
                if e is not None and e["score"] > 0.6:
                    self.est.append(e)
                    self._plan(e["c"], e["th"])
                self.phase, self.count, self.lost_n = ["approach", "approach"], [0, 0], [0, 0]
                self.lift_t, self.lifts = None, 0
        # transitions: both arms together (the same phase, the same heights)
        if self.phase[0] == self.phase[1] == "approach":
            slow = self.count[0] > 60  # (an arm that cannot settle at the approach point goes on with relaxed tolerance)
            if all(self._at(j, G, P_app[j], np.radians(8 if slow else 4), 0.02 if slow else 0.01, True, op) for j in (0, 1)):
                self.phase, self.count, self.zh = ["descend", "descend"], [0, 0], [[], []]
        elif self.phase[0] == self.phase[1] == "descend":
            tol = 0.012 if self.count[0] > 25 else 0.005
            if all(np.linalg.norm(P_gr[j] - G[j][0]) < tol for j in (0, 1)):
                self.phase, self.count, self.hold_n = ["grasp", "grasp"], [0, 0], [0, 0]
        elif self.phase[0] == self.phase[1] == "relower" and (
                (self.count[0] > 3 and all(abs(G[j][0][2] - self.g0[j][2]) < 0.012 for j in (0, 1))) or self.count[0] > 25):
            self.phase = ["ready", "ready"]  # (lift again from here, pulling to X_PULL2)
        if self.phase == ["ready", "ready"]:
            self.phase, self.lift_t = ["lift", "lift"], self.t
            self.g0 = [G[k][0].copy() for k in (0, 1)]
            self.lifts += 1
            # tray centre now: midway between the hands (they hold the two walls' middles); first lift: the estimate
            self.lift_c = self.tray["c"].copy() if self.lifts == 1 else 0.5 * (self.g0[0][:2] + self.g0[1][:2])
        # synchronized P control: in a joint phase the arm with less to go moves proportionally slower (both arrive
        # together, at the same heights)
        E = [T[k] - G[k][0] for k in (0, 1)]
        nrm = [float(np.linalg.norm(e)) for e in E]
        nmax = max(max(nrm), 1e-9)
        for k in (0, 1):
            v = self.KP * E[k]
            cap = self.VMAX * (nrm[k] / nmax if self.phase[0] == self.phase[1] else 1.0)
            nv = float(np.linalg.norm(v))
            if nv > cap > 0:
                v = v * cap / nv
            a[6 * k:6 * k + 3] = v
            if yaw[k] is not None:
                a[6 * k + 5] = np.clip(self.KR * wrap(yaw[k] - G[k][1], np.pi), -self.RMAX, self.RMAX)
            a[12 + k] = grip[k]
        if self.log:
            tilt = [round(float(np.degrees(np.arccos(np.clip(-G[k][2][2, 2], -1, 1)))), 1) for k in (0, 1)]
            self.trace.append((self.t, "/".join(self.phase), np.round(G[0][0], 3).tolist(), np.round(G[1][0], 3).tolist(), tilt,
                               [round(float(np.degrees(G[k][1])), 1) for k in (0, 1)], np.round(op, 2).tolist()))
        return a
