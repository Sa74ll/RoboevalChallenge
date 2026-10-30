"""Hybrid CubeHandover controller (Oct 28): classical RGB-D perception + a scripted two-arm handover, using only the
challenge's standard observations (nothing is read from the simulator at test time).

Task (roboeval/envs/manipulation.py, CubeHandoverPositionAndOrientation): the "cube" is a red rod 0.04 x 0.20 x 0.04 m,
3 kg, lying on the table at world (0.5, y, 0.97), y in {-0.15, 0, 0.15} +- 0.05, yaw uniform in +-180 deg. Success: the
gripper that held it first no longer holds it, the other gripper does, and the rod does not touch the table.

Frames: everything is in the ROBOT frame (world x - 0.1: this task's scene places the robot at x = 0.1), the frame of
hybrid_s2b.fk and of the camera constants hybrid_s2b.CAMS (the cameras are mounted on the robot base).
Perception (rod_perc.py): red pixels of the head and front RGB-D cameras -> 3D points. On the table, the top face gives the
rod centre and axis (locate_rod); once lifted, the free end of the held rod (locate_held). Gripper poses: forward kinematics
from proprioception; holding: the finger opening in proprioception_grippers.

Plan: the giver G (the arm on the rod's side) grasps the rod top-down D_GRASP from its centre on its own end, fingers across
the rod; lifts it, carries it towards the handover pose (rod centre at X_H, |y| <= Y_H, LIFT_DZ up) and turns it so the rod
lies along the y axis with the giver's end on the giver's side, stopping MEET short on its own side. The receiver R waits
WAIT_OUT beyond its grasp point on the other end, out along the rod axis (its arm stays on its own side), then slides in
along the rod with open fingers while the giver brings the rod the last MEET; R closes, then G opens (success).
Recovery: missed grasp, blocked descent or drop -> G opens, looks again from above and aside, re-plans, retries; a blocked
or missed insert -> R backs out and retries, aiming at the perceived free end.
Actions: 14-D end-effector deltas [dp_L dr_L | dp_R dr_R | g_L g_R] (world frame = robot frame axes), grippers 0 open."""
import numpy as np

try:
    from .hybrid_s2b import fk, wrap
    from .rod_perc import locate_rod, locate_held, Z_REST, ROD_HALF
except ImportError:
    from hybrid_s2b import fk, wrap
    from rod_perc import locate_rod, locate_held, Z_REST, ROD_HALF

GRASP_DZ = 0.110  # gripper origin above the rod centre when grasping (pads 0.103 below; fingertips ~1.5 cm above the table)
APP_DZ = 0.040  # approach point above the grasp height (fingertips clear the rod top), then straight down
D_GRASP = 0.065  # grasp points this far from the rod centre along the rod (hands 0.13 apart; at 0.055 the right giver's
# forearm touched the left receiver's wrist on rods lying along x, after the giver's ~80 deg turn)
LIFT_DZ = 0.060  # rod centre above its resting height at the handover
WAIT_OUT = 0.06  # receiver waits this far beyond its grasp point, out along the rod axis, until the rod is in place
X_H, Y_H = 0.40, 0.08  # handover: rod centre x, and |y| at most Y_H
TAU = 0.0  # tolerated rod yaw deviation from the y axis at the handover (rad); > 0 brought the arms into conflict
MEET = 0.05  # the giver stops the rod this far short (on its own side) and brings it in while the receiver slides in, so
# both arms move during the insert (bavd 0.342 -> 0.301, ct 4.34 -> 4.09 s on 24 seeds vs MEET 0 with WAIT_OUT 0.10)
CLEAR = 0.10  # min distance between the two hands' finger-axis segments (xy) while the receiver waits; pushes its waiting
# point out along its side while the giver works near it (rod along x near the centre: hands touched, seed 815345)
VIS_GATE = 0.025  # use the perceived free end of the held rod if within this of the kinematic prediction (5 cm after a miss)
# speeds (per 50 ms step): half of StackTwoBlocks' controller. The scorer's arm-velocity difference (bavd, mean over steps of
# | |qvel_L| - |qvel_R| |) scales about inversely with speed for the same paths, and time stays below the human median:
# full speed 2.6 s / bavd 0.53, 2/3 speed 3.3 s / 0.39, half speed 4.2 s / 0.30 (16 seeds; human 5.55 s / 0.357)
VMAX, VMAX_FAST, KP = 0.006, 0.008, 0.6
RMAX, KR = 0.045, 0.8
DEFAULTS = {k: globals()[k] for k in ("GRASP_DZ", "APP_DZ", "D_GRASP", "LIFT_DZ", "WAIT_OUT", "X_H", "Y_H", "TAU", "MEET", "CLEAR", "VIS_GATE", "VMAX", "VMAX_FAST", "KP", "RMAX", "KR")}
BASE_Y = (0.3, -0.3)


def _holding(obs, k):
    f = float(np.asarray(obs["proprioception_grippers"])[k])
    return 0.15 < f < 0.8


def _seg_dist(p1, d1, p2, d2, n=9):
    """min distance between the 2D segments p1 +- d1 and p2 +- d2 (sampled)"""
    s = np.linspace(-1.0, 1.0, n)[:, None]
    A, B = p1 + s * d1, p2 + s * d2
    return float(np.min(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)))


def _fingers(yaw):
    """half finger-axis segment (xy) of a top-down hand with gripper yaw `yaw` (the fingers close along its y axis)"""
    return 0.10 * np.array([-np.sin(yaw), np.cos(yaw)])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class _Arm:
    def __init__(self, k):
        self.k, self.phase, self.count, self.hold_n = k, "idle", 0, 0
        self.recov, self.t0, self.zh, self.dh, self.z_hold = 0, 0, [], [], None


class HybridCH:
    carry_queue = False  # the bundle's policy.py resets every episode

    def __init__(self, log=False, **cfg):
        """cfg: overrides of the module constants in DEFAULTS (from checkpoint/CubeHandover/hybrid.json)"""
        self.log = log
        bad = set(cfg) - set(DEFAULTS)
        assert not bad, f"unknown hybrid_ch options {sorted(bad)}"
        for k, v in {**DEFAULTS, **cfg}.items():
            setattr(self, k, float(v))
        self.reset()

    def reset(self, episode=None):
        self.t = 0
        self.rod = None  # latest rod estimate on the table: {c, u, yaw}
        self.rod_t = 0  # step of the latest accepted estimate
        self.G = self.R = None
        self.arms = [_Arm(0), _Arm(1)]
        self.rel = None  # rod centre in the giver's gripper frame, set at the grasp
        self.gy_grasp = self.R_grasp = None
        self.phi = 0.0  # giver's planned yaw turn after the grasp
        self.C_h = None  # rod centre at the handover
        self.recv_tgt = None  # receiver's grasp target, frozen when it starts sliding in
        self.pull = True  # (MEET) giver still holds the rod short of the handover pose
        self.giver_final = False  # giver settled at the final handover pose
        self.hold_t = None  # step at which the giver last reached its handover pose
        self.use_vision = False  # after a failed insert: accept the perceived free end of the held rod up to 5 cm away
        self.trace = []

    # -- geometry -----------------------------------------------------------------------------------------------------
    def _plan(self, G):
        c, u = self.rod["c"], self.rod["u"]
        # the giver: the arm on the rod's side; near the middle, the one closer to its grasp point
        if abs(c[1]) > 0.03:
            g = 0 if c[1] > 0 else 1
        else:
            d = [np.linalg.norm(G[k][0][:2] - self._grasp_xy(k)) for k in (0, 1)]
            g = int(np.argmin(d))
        self.G, self.R = g, 1 - g
        self.arms[g].phase = "approach"
        self.arms[1 - g].phase = "approach"
        self._plan_geometry()

    def _plan_geometry(self):
        """handover pose from the resting-rod estimate (recomputed whenever the estimate changes before a grasp; the roles
        stay: the scorer fixes the giver at its first grasp)"""
        c, u = self.rod["c"], self.rod["u"]
        th = float(np.arctan2(u[1], u[0]))  # in [0, pi)
        dev = th - np.pi / 2  # deviation of the rod axis from +y, in [-pi/2, pi/2)
        self.phi = -np.sign(dev) * max(0.0, abs(dev) - self.TAU)  # turn of the rod (and the giver's hand) after the grasp
        yc = float(np.clip(c[1], -self.Y_H, self.Y_H))
        self.C_h = np.array([self.X_H, yc, Z_REST + self.LIFT_DZ])

    def _grasp_xy(self, k):
        """grasp point of arm k on the resting rod: left takes the +y end (u_y >= 0), right the other"""
        c, u = self.rod["c"], self.rod["u"]
        return c[:2] + (self.D_GRASP if k == 0 else -self.D_GRASP) * u

    def _rod_live(self, G):
        """rod centre (3,) and unit direction (xy) from the centre towards the giver's grasp, from the giver's kinematics"""
        gp, gy, gR = G[self.G]
        c = gp + gR @ self.rel
        w = gp[:2] - c[:2]
        return c, w / max(np.linalg.norm(w), 1e-6)

    def _handover_targets(self, G):
        """giver hand target (3,), its yaw target, and the receiver's grasp point (3,) and yaw (rod axis)"""
        gp, gy, gR = G[self.G]
        Rt = _rz(self.phi) @ self.R_grasp
        hand = self.C_h - Rt @ self.rel
        yaw_g = self.gy_grasp + self.phi
        c = self.C_h
        w = hand[:2] - c[:2]
        w = w / max(np.linalg.norm(w), 1e-6)
        recv = np.r_[c[:2] - self.D_GRASP * w, c[2] + self.GRASP_DZ]
        return hand, yaw_g, recv, float(np.arctan2(w[1], w[0]))

    def _giver_restart(self):
        Ag = self.arms[self.G]
        Ag.recov += 1
        self.rel = None
        self.pull, self.giver_final = True, False
        Ag.phase, Ag.count, Ag.t0 = "reopen", 0, self.t

    @staticmethod
    def _valid(e):
        """a plausible resting-rod estimate: enough top-face points, the rod's visible length and width"""
        return e is not None and e["ntop"] >= 300 and abs(e["len"] - 0.2) <= 0.015 and abs(e["wid"] - 0.04) <= 0.008

    def _perceive(self, obs, G):
        """update the resting-rod estimate while no gripper is low over it (only before the giver's grasp)"""
        e = locate_rod(obs)
        if not self._valid(e):
            return
        if any(g[0][2] < Z_REST + 0.20 and np.linalg.norm(g[0][:2] - e["c"][:2]) < 0.16 for g in G):
            return
        u = e["u"]
        if self.rod is not None and float(u @ self.rod["u"]) < 0:  # keep the axis sign (rods along x: u_y ~ 0 may flip)
            u = -u
        self.rod = {"c": np.r_[e["c"][:2], Z_REST], "u": u, "yaw": float(np.arctan2(u[1], u[0]))}
        self.rod_t = self.t
        if self.G is not None and self.rel is None:
            self._plan_geometry()

    # -- policy -------------------------------------------------------------------------------------------------------
    def act(self, obs):
        self.t += 1
        a = np.zeros(14, np.float32)
        q = np.asarray(obs["proprioception"], np.float64)
        G = [fk(q[9 * k:9 * k + 7], k) for k in (0, 1)]
        if self.rod is None:  # plan on the first plausible estimate (wait up to 10 steps for one)
            e = locate_rod(obs)
            if not self._valid(e) and self.t < 10:
                return a
            self.rod = ({"c": np.r_[e["c"][:2], Z_REST], "u": e["u"], "yaw": e["yaw"]} if e is not None else
                        {"c": np.array([0.4, 0.0, Z_REST]), "u": np.array([0.0, 1.0]), "yaw": np.pi / 2})
            self.rod_t = self.t
            self._plan(G)
        Ag, Ar = self.arms[self.G], self.arms[self.R]
        if Ag.phase in ("approach", "descend", "reopen"):
            self._perceive(obs, G)
        hold_g, hold_r = _holding(obs, self.G), _holding(obs, self.R)

        # ---- giver ----
        gp, gy, gR = G[self.G]
        grip_g, yaw_g, ymod = 0.0, None, np.pi
        if Ag.phase in ("approach", "descend", "reopen"):
            P = self._grasp_xy(self.G)
            th = self.rod["yaw"]
            z_grasp = self.rod["c"][2] + self.GRASP_DZ
            if Ag.phase == "approach":
                tg, yaw_g = np.r_[P, z_grasp + self.APP_DZ], th
                if np.linalg.norm(tg - gp) < 0.012 and abs(wrap(th - gy, np.pi)) < np.radians(5):
                    Ag.phase, Ag.count, Ag.zh = "descend", 0, []
            elif Ag.phase == "descend":
                tg, yaw_g = np.r_[P, z_grasp], th
                Ag.count += 1
                Ag.zh.append(gp[2])
                blocked = Ag.count > 8 and Ag.zh[-7] - gp[2] < 0.002 and gp[2] > z_grasp + 0.008  # a finger on the rod
                if abs(gp[2] - z_grasp) < 0.006 and np.linalg.norm(gp[:2] - P) < 0.006:
                    Ag.phase, Ag.count, Ag.hold_n = "grasp", 0, 0
                elif blocked or Ag.count > 45:
                    self._giver_restart()
            else:  # reopen (missed grasp or drop): open, up and aside so both cameras see the rod, wait for a fresh estimate
                side = np.array([0.0, 1.0 if self.G == 0 else -1.0])
                tg = np.r_[self.rod["c"][:2] + 0.10 * side, Z_REST + 0.22]
                Ag.count += 1
                if Ag.count >= 8 and gp[2] > Z_REST + 0.19 and (self.rod_t > Ag.t0 + 2 or Ag.count >= 60):
                    Ag.phase = "approach"
        elif Ag.phase == "grasp":
            tg, grip_g = gp, 1.0
            Ag.count += 1
            Ag.hold_n = Ag.hold_n + 1 if hold_g else 0
            if Ag.hold_n >= 2 and Ag.count >= 3:
                c = self.rod["c"]
                rel = gR.T @ (c - gp)
                rel[1] = 0.0  # the fingers centre the rod across the gripper's y axis
                self.rel, self.R_grasp, self.gy_grasp = rel, gR.copy(), gy
                Ag.phase, Ag.count = "lift", 0
            elif Ag.count >= 15:
                self._giver_restart()
        elif Ag.phase in ("lift", "hold") and not hold_g:  # dropped: open, re-perceive, start over
            self._giver_restart()
            tg = gp
        elif Ag.phase in ("lift", "hold"):
            hand, yg, _, _ = self._handover_targets(G)
            wf = hand[:2] - self.C_h[:2]
            wf = wf / max(np.linalg.norm(wf), 1e-6)
            hand_final = hand
            if self.MEET and self.pull:
                hand = np.r_[hand[:2] + self.MEET * wf, hand[2]]
            self.giver_final = (not (self.MEET and self.pull)) and np.linalg.norm(hand_final - gp) < 0.010
            grip_g, ymod = 1.0, 2 * np.pi
            z_grasp = self.rod["c"][2] + self.GRASP_DZ
            if gp[2] < z_grasp + 0.025:  # straight up first, clear of the table
                tg, yaw_g = np.r_[gp[:2], hand[2]], None
            else:
                tg, yaw_g = hand, yg
            Ag.count += 1
            close = np.linalg.norm(hand - gp) < 0.010 and abs(wrap(yg - gy, 2 * np.pi)) < np.radians(4)
            near = np.linalg.norm(hand - gp) < 0.025 and abs(wrap(yg - gy, 2 * np.pi)) < np.radians(10)
            if Ag.phase == "lift" and (close or (near and Ag.count > 80)):  # (a stalled arm settles for near)
                Ag.phase, self.hold_t = "hold", self.t
            # both hold: the giver opens (Oct 28 16:22: the earlier rule that timed this step to the scorer's slip-sampling
            # frames was removed, so nothing here depends on how the scorer samples)
            if Ag.phase == "hold" and Ar.phase == "hold":
                Ag.phase, Ag.count = "release", 0
        else:  # release; the episode ends on success at the contact loss. If it has not ended after 4 steps, back off up
            tg, grip_g = gp, 0.0
            Ag.count += 1
            if Ag.count > 4:
                tg = np.r_[gp[:2], min(gp[2] + 0.01, Z_REST + 0.30)]
            if Ag.count > 15 and not hold_r:  # nobody holds the rod (it fell back to the table): start over
                self._giver_restart()
                Ar.phase = "approach"
        self._write(a, self.G, gp, gy, tg, yaw_g, grip_g, ymod)

        # ---- receiver ----
        gp, gy, gR = G[self.R]
        grip_r = 0.0
        recv_p, yaw_p, w_p = self._plan_recv()  # planned handover: receiver grasp point, yaw, rod direction
        wait = recv_p - np.r_[self.WAIT_OUT * w_p, 0.0]  # on the receiver's side, out along the rod axis (clear of the giver)
        yaw_r = yaw_p
        if self.rel is not None and Ag.phase in ("hold", "release"):
            c, w = self._rod_live(G)
            yaw_r = float(np.arctan2(w[1], w[0]))
        if Ag.phase in ("approach", "descend", "grasp", "reopen", "lift"):  # keep the hands CLEAR apart
            gpg, gyg = G[self.G][0], G[self.G][1]
            for _ in range(8):
                if _seg_dist(gpg[:2], _fingers(gyg), wait[:2], _fingers(yaw_p)) >= self.CLEAR:
                    break
                wait = wait - np.r_[0.015 * w_p, 0.0]
        if Ag.phase == "reopen":  # keep out of the cameras' way while the giver looks for the rod again
            wait = np.r_[wait[:2], max(wait[2], Z_REST + 0.22)]
        if Ar.phase == "approach":
            tr = wait
            dw, dyaw = np.linalg.norm(tr - gp), abs(wrap(yaw_r - gy, np.pi))
            late = self.hold_t is not None and self.t - self.hold_t > 60  # cannot settle at the waiting point: go anyway
            if Ag.phase == "hold" and ((dw < 0.015 and dyaw < np.radians(5)) or (late and dw < 0.03 and dyaw < np.radians(10))):
                tgt = self._recv_target(obs, G)  # frozen: never chase the giver if the rod is bumped
                if self.MEET and self.pull:  # the giver now brings the rod the rest of the way
                    _, w = self._rod_live(G)
                    tgt = np.r_[tgt[:2] - self.MEET * w, tgt[2]]
                    self.pull = False
                self.recv_tgt = tgt
                Ar.phase, Ar.count, Ar.dh = "insert", 0, []
        elif Ar.phase == "insert":  # slide in along the rod axis, fingers open around the rod
            tr = self.recv_tgt
            Ar.count += 1
            dist = float(np.linalg.norm(tr - gp))
            Ar.dh.append(dist)
            blocked = Ar.count > 10 and Ar.dh[-9] - dist < 0.003 and dist > 0.008  # stuck against the rod end
            if Ag.phase not in ("hold", "release"):
                Ar.phase = "approach"
            elif dist < 0.006 and (self.giver_final or not self.MEET):
                Ar.phase, Ar.count, Ar.hold_n = "grasp", 0, 0
            elif blocked or Ar.count > 60:
                Ar.recov += 1
                self.use_vision = True
                Ar.phase, Ar.count = "reopen", 0
        elif Ar.phase == "grasp":
            tr, grip_r = gp, 1.0
            Ar.count += 1
            Ar.hold_n = Ar.hold_n + 1 if hold_r else 0
            if Ar.hold_n >= 2 and Ar.count >= 3:
                Ar.phase = "hold"
            elif Ar.count >= 15:
                Ar.recov += 1
                self.use_vision = True
                Ar.phase, Ar.count = "reopen", 0
        elif Ar.phase == "hold":
            tr, grip_r = gp, 1.0
            if Ag.phase == "release" and Ag.count > 8:  # no success yet (rod on the table?): lift it 3 cm
                if Ar.z_hold is None:
                    Ar.z_hold = gp[2]
                tr = np.r_[gp[:2], Ar.z_hold + 0.03]
            if not hold_r:
                Ar.recov += 1
                Ar.phase, Ar.count = "reopen", 0
        else:  # reopen: open and back out along the rod axis to the waiting point, then try again
            tr = wait
            Ar.count += 1
            if Ar.count >= 6 and np.linalg.norm(wait - gp) < 0.02:
                Ar.phase = "approach"
        self._write(a, self.R, gp, gy, tr, yaw_r, grip_r, np.pi)
        if self.log:
            self.trace.append((self.t, Ag.phase, Ar.phase, np.round(G[self.G][0], 3).tolist(), np.round(G[self.R][0], 3).tolist(),
                               int(hold_g), int(hold_r)))
        return a

    def _recv_target(self, obs, G):
        """receiver grasp point (3,): xy from the perceived free end of the held rod when that is plausible (within VIS_GATE of
        the kinematic prediction; 2.1 mm median error on 20 training seeds vs 3.6 mm for the kinematics), height from the giver's
        kinematics and the grasp offset (0.6 mm; the perceived top face reads 1.7 mm high)"""
        c, w = self._rod_live(G)
        kin = np.r_[c[:2] - self.D_GRASP * w, c[2] + self.GRASP_DZ]
        e = locate_held(obs, G[self.G][0], w)
        if e is not None and e["ntop"] >= 15:
            vis = e["end_xy"] + (ROD_HALF[1] - self.D_GRASP) * w
            if np.linalg.norm(vis - kin[:2]) < (0.05 if self.use_vision else self.VIS_GATE):
                return np.r_[vis, kin[2]]
        return kin

    def _plan_recv(self):
        """planned handover (from the resting-rod estimate): receiver grasp point (3,), its yaw, and the unit rod direction
        (xy) from the rod centre towards the giver's grasp"""
        c, u = self.rod["c"], self.rod["u"]
        w0 = self._grasp_xy(self.G) - c[:2]
        w0 = w0 / max(np.linalg.norm(w0), 1e-6)
        w = _rz(self.phi)[:2, :2] @ w0
        recv = np.r_[self.C_h[:2] - self.D_GRASP * w, self.C_h[2] + self.GRASP_DZ]
        return recv, float(np.arctan2(w[1], w[0])), w

    def _write(self, a, k, gp, gy, target, yaw_ref, grip, ymod):
        off = 6 * k
        e = np.asarray(target) - gp
        vmax = self.VMAX_FAST if np.linalg.norm(e) > 0.05 else self.VMAX
        a[off:off + 3] = np.clip(self.KP * e, -vmax, vmax)
        if yaw_ref is not None and np.isfinite(yaw_ref):
            a[off + 5] = np.clip(self.KR * wrap(yaw_ref - gy, ymod), -self.RMAX, self.RMAX)
        a[12 + k] = grip
