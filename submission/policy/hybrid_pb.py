"""Hybrid PackBox controller (Oct 28): classical RGB-D perception of the box pose + geometry-aware scripted pushing of
both flaps closed, each arm holding its own flap on the way (task stages 1-5). Only the standard observations are used
(head/front depth for the box, proprioception for the arms); nothing is read from the simulator.

Frames. Everything is in the robot frame of fk and CAMS (arm bases at (0, +-0.3, 0.6)); in PackBox the world frame is
this one shifted by (0.2, 0.3, 0.2), which never matters because actions are deltas. The box is fixed for the whole
episode (it has no free joint), so its pose is estimated once, at the first step.

Perception (locate_box). Head and front depth are back-projected with CAMS. At reset only the box occupies the height
band 0.70-0.95 m (below the hinges); those points are the box's wall shell, a square. Yaw = the minimum-area rectangle of
the shell, searched over 52-128 deg (one minimum for every task yaw, 90 +- 30 deg), coarse to fine; centre = the middle
of the shell's trimmed extents minus the shell's offset in the box frame. Error on 30 training seeds: 0.5 mm median
(max 1.7 mm), yaw 0.01 deg median (max 0.10 deg).

Geometry (from the asset, box frame). flap_1 (the right flap; the RIGHT arm must hold it) hinges on the line x = -0.21,
z = 0.1443 (axis -y), flap_2 (left flap, LEFT arm) on x = 0.2178, z = 0.144 (axis +y); lengths 0.2188 / 0.2089 m,
thickness ~9 mm, width 0.42 m. In a flap's radial plane (X outward from its hinge, Z up), theta = the flap's elevation:
20 deg when open (the joint's upper limit), 180 deg flat on the box (lower limit); the task calls it closed beyond
170.3 (flap_1) / 172.2 deg (flap_2). The flap damping (10 N m s/rad) dominates its dynamics: pushing it at 2.5 deg per
step needs ~9 N m; gravity (~1 N m) is small.

Pusher. Both grippers stay closed (as at reset): the two fingertip pads side by side form a compact pusher, the hand
points down and the finger line runs along the hinge, so both pads touch the flap together (the scorer's "holding" =
both finger pads in contact) and the stage of each arm is reached as soon as it pushes.
  Lift (theta <= 90 deg): the pads' inner side face pushes the flap's tip edge, which rides H_E = 15 mm above the
  fingertip, mid-height on the 24 mm pads (at 25 mm it rode on the pads' top edge and the contact sometimes dropped late
  in the lift: a slip); friction makes the push lift efficiently and gravity keeps the flap on the pads.
  Fold (theta > 90 deg): the pads' inner bottom corner presses the flap's outer face at the same distance from the hinge
  (L - H_E): sliding along the flap (pad friction 3) makes the contact chatter, and then the arm slips.
The fingertip follows this path in a feed-forward virtual command (sum of the deltas sent) plus a servo-lag feed-forward
(the hand trails its command by ~0.09 s x velocity; ctrl - q ~ kv/kp qdot of the Panda servos), so the pads stay on the
flap. The aim point advances 2.5 deg per step and pauses while the hand trails the command by more than 8 mm (estimated
from q + 0.09 qdot) or tilts more than 4 deg from its target (the wrist joints saturate at 12 N m under too much load);
past 95 deg it keeps at least 1 deg per step unless the lag exceeds 16 mm (a pause there lets gravity carry the flap off
the pads: lost holding, a slip).
Orientation: feedback on the measured hand with the command kept within 3 deg of the target (integral action against
deflection: 1 deg of roll lifts one pad off the flap). At the end the fingers press the closed flap and hold still.

Planning (once, from the perceived pose). Push position s along the hinge (-0.18..0.18 m) per arm from its reach: the
largest distance from the arm's shoulder to the wrist point (hand origin + 0.107 m up the hand axis) over the planned
run-in, lift and fold; comfortable below 0.70 m (roboeval's IK and the servos stall ~0.05 m beyond). If no position is
comfortable (far boxes turned ~30 deg), the hand is rolled about the flap's outward axis during the early lift (up to 28
deg, gone by theta 75 deg): the inner face keeps its plane and the wrist leans toward the shoulder.
  Parallel mode: if both arms can push comfortably 0.30 m apart along the hinge, both push all the way at once.
  Sequenced mode otherwise: the arm with the easier reach finishes first, opens its fingers (the opening shows in the
  slip check before contact ends, so no slip) and moves to its own side, its height following the other hand. The other
  arm waits off its flap at its run-in point, its fingertip height following the first arm's (coordination metric),
  until the first arm's flap passes 130 deg; it then pushes at the full rate and normally reaches 85 deg just after the
  first arm has left (it would wait there, below 90 deg where gravity keeps the flap on the pads, if not). Slow or
  stationary pushing loses the pads' contact (a slip) and is avoided.
The episode ends at success (both flaps closed), so no arm has to let go at the end.

Interface: HybridPB(**cfg); reset(episode); act(obs) -> (14,) float32 end-effector deltas [dp_L dr_L | dp_R dr_R | g_L g_R]
(world frame, grippers 0 open / 1 closed)."""
import numpy as np

try:
    from .hybrid_s2b import fk, CAMS
except ImportError:
    from hybrid_s2b import fk, CAMS

# ---- box perception -------------------------------------------------------------------------------------------------
Z_BOX = 0.85  # box body height (robot frame), fixed by the task
BAND_WALL = (0.70, 0.95)  # wall points (below the hinges at 0.994); at reset nothing else is in 0.70-1.09
BAND_FLAP = (1.00, 1.09)  # open flaps (0.994 .. 1.07); the home fingertips are at 1.109 (used as a sanity count)
VIS_OFF = np.array([0.0045, 0.0035])  # centre of the visual box shell in the box frame (x: -0.214..0.223, y: -0.215..0.222)


def _cloud(obs, n):
    c = CAMS[n]
    R, p = np.array(c["mat"]), np.array(c["pos"])
    dep = np.asarray(obs[f"depth_{n}"])
    h, w = dep.shape
    ys, xs = np.mgrid[0:h, 0:w]
    f = (h / 2) / np.tan(np.radians(c["fovy"]) / 2)
    P = np.stack([(xs + 0.5 - w / 2) / f * dep, -(ys + 0.5 - h / 2) / f * dep, -dep], -1) @ R.T + p
    return P.reshape(-1, 3)


def _extent(xy, yaw, q=0.002):
    c, s = np.cos(yaw), np.sin(yaw)
    u = xy @ np.array([c, s])
    v = xy @ np.array([-s, c])
    ul, uh = np.quantile(u, [q, 1 - q])
    vl, vh = np.quantile(v, [q, 1 - q])
    return ul, uh, vl, vh


def locate_box(obs):
    """box pose in the robot frame from the head and front depth images at reset: {xy, yaw (rad), ok}"""
    P = np.concatenate([_cloud(obs, n) for n in ("head", "front")])
    P = P[(P[:, 0] > 0.12) & (P[:, 0] < 1.15) & (np.abs(P[:, 1] + 0.05) < 0.8)]
    W = P[(P[:, 2] > BAND_WALL[0]) & (P[:, 2] < BAND_WALL[1]), :2]
    nf = int(np.sum((P[:, 2] > BAND_FLAP[0]) & (P[:, 2] < BAND_FLAP[1])))
    if len(W) < 200 or nf < 100:
        return {"xy": np.array([0.5, -0.05]), "yaw": np.pi / 2, "ok": False, "nw": len(W), "nf": nf}
    best = None
    for yy in np.radians(np.arange(52.0, 128.01, 1.0)):  # a square shell: period 90 deg; one minimum in 52..128 deg
        ul, uh, vl, vh = _extent(W, yy)
        a = (uh - ul) * (vh - vl)
        if best is None or a < best[0]:
            best = (a, yy)
    for step, span in ((0.1, 1.0), (0.01, 0.1)):
        y1 = best[1]
        for dy in np.radians(np.arange(-span, span + step / 2, step)):
            ul, uh, vl, vh = _extent(W, y1 + dy)
            a = (uh - ul) * (vh - vl)
            if a < best[0]:
                best = (a, y1 + dy)
    yaw = best[1]
    ul, uh, vl, vh = _extent(W, yaw)
    ex, ey = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    ctr = ex * ((ul + uh) / 2 - VIS_OFF[0]) + ey * ((vl + vh) / 2 - VIS_OFF[1])
    return {"xy": ctr, "yaw": float(yaw), "ok": True, "nw": len(W), "nf": nf, "size": (uh - ul, vh - vl)}


# ---- flap geometry (box frame, from the asset) ----------------------------------------------------------------------
FLAP = (  # index = arm: the LEFT arm (0) must hold flap_2 (box +x side), the RIGHT arm (1) flap_1 (box -x side)
    {"ax": 0.2178, "az": 0.144, "o": 1.0, "L": 0.2089, "ht": 0.0044, "th0": np.radians(20.2)},
    {"ax": -0.21, "az": 0.1443, "o": -1.0, "L": 0.2188, "ht": 0.0048, "th0": np.radians(20.3)},
)
FT = 0.1154  # hand origin -> bottom of the closed fingers' pads (hand z axis)
WF = 0.0105  # half width of a finger along the hand x axis
SHOULDER = (np.array([0.0, 0.3, 0.933]), np.array([0.0, -0.3, 0.933]))  # Panda joint 2 (left, right)
ROLL_TH1, ROLL_TH2 = np.radians(40), np.radians(75)  # rolled hand: full roll below TH1, none above TH2


class FlapPath:
    """fingertip path of arm k for its flap, at push position s along the hinge (see the module docstring)"""

    def __init__(self, k, cxy, yaw, s, depth=0.003, h_e=0.015, roll0=0.0):
        self.k, self.f, self.s, self.depth, self.h_e, self.roll0 = k, FLAP[k], s, depth, h_e, roll0
        self.c = np.array([cxy[0], cxy[1], Z_BOX])
        cy, sy = np.cos(yaw), np.sin(yaw)
        self.R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
        self.axis = self.R[:, 1].copy()  # hinge axis direction (box y), robot frame

    def to_robot(self, X, Z):
        f = self.f
        return self.c + self.R @ np.array([f["ax"] + f["o"] * X, self.s, f["az"] + Z])

    def roll(self, th):
        if self.roll0 <= 0:
            return 0.0
        return self.roll0 * float(np.clip((ROLL_TH2 - th) / (ROLL_TH2 - ROLL_TH1), 0.0, 1.0))

    def XZ(self, th):
        """fingertip (pad centre) in the radial plane at flap angle th"""
        f = self.f
        L, ht, d = f["L"], f["ht"], self.depth
        if th <= np.pi / 2:  # lift: inner face on the tip's trailing corner (a roll about X keeps this face in place)
            return L * np.cos(th) + ht * np.sin(th) + WF - d, L * np.sin(th) - self.h_e
        rho = L - self.h_e  # fold: inner bottom corner on the outer face, where the lift left it (no sliding)
        cx = rho * np.cos(th) + (ht - d) * np.sin(th)
        cz = rho * np.sin(th) - (ht - d) * np.cos(th)
        return cx + WF, cz

    def point(self, th):
        return self.to_robot(*self.XZ(th))


# ---- control constants ----------------------------------------------------------------------------------------------
VMAX, VMAX_FAST, KP = 0.012, 0.016, 0.6  # free motion: m per step (20 Hz), P gain on the measured hand position
RMAX, KR = 0.06, 0.8  # rad per step, orientation gain
RATE = np.radians(2.5)  # aim advance per step along the flap path (flap damping 10 N m s/rad; wrist joints 12 N m)
RATE_SMOOTH = 0.35  # first-order smoothing of the aim rate (pauses and resumes without command jumps)
TILT_OK = np.radians(4)  # the hand deflects from its target orientation when the wrist saturates: pause the aim beyond
P_LAG = 0.008  # ... and while the command leads the (lag-compensated) hand by more than this
KV_KP = 0.09  # Panda joint servos: ctrl - q ~ (kv/kp = 0.1) qdot; 0.09 fits the accelerating case too
PAUSE_MAX = 25  # after this many paused steps the command re-syncs to the robot (guards an IK failure)
VMAX_PUSH = 0.016  # command speed limit in contact
FF_MAX = 0.025  # servo-lag feed-forward limit
R_DEV = np.radians(3)  # orientation command kept within this of the target
EXT = np.radians(12)  # path extension before theta0: a horizontal run-in of X_RUN to the flap's tip edge
X_RUN = 0.03
Z_CLEAR = 0.03  # transit height of the fingertip above the open flap's tip
TH_END = np.radians(180.8)  # pressing target (the joint stops at 180.3 / 180.2 deg)
HOLD_TIMEOUT = 12  # steps at the end of the path before holding anyway
S_GRID = np.round(np.arange(-0.18, 0.181, 0.03), 2)  # candidate push positions along the hinge (flap: -0.207..0.214)
D_SEP = 0.30  # parallel mode: push positions this far apart (hands 0.2 m wide along the hinge, wrists and forearms
# beside them: 0.22-0.24 m left 0-40 mm between the arms on some boxes)
D_COMF = 0.70  # comfortable reach (wrist point to shoulder)
ROLL_LIST = np.radians([0.0, 12.0, 20.0, 28.0])  # early-lift roll options for an arm that cannot reach comfortably
RATE_MIN_FOLD = np.radians(1.0)  # in the fold the aim never stops (gravity would carry the flap off the pads) unless
LAG_HARD = 0.016  # ... the lag exceeds this or the hand deflects beyond TILT_OK (real overload or a stalled IK)
Z_WAIT = (0.0, 0.15)  # sequenced mode: the second arm waits off its flap at the run-in point, its fingertip height
# following the other arm's (offset range above the run-in height), until the first arm's flap passes TH_GO2; it then
# pushes at full rate, and only waits at CAP2 (below 90 deg, where gravity keeps the flap on the pads) if the first arm
# has not left yet
TH_GO2 = np.radians(130)
CAP2 = np.radians(85)
REL_UP = 0.04  # the finished arm: up this much with the fingers opening, then its fingertips to AWAY (its own side,
AWAY = (np.array([0.38, 0.42, 1.15]), np.array([0.38, -0.42, 1.15]))  # clear of the box and of the other arm)


def _rotvec(Rm):
    """rotation vector of a rotation matrix"""
    c = (np.trace(Rm) - 1) / 2
    ang = np.arccos(np.clip(c, -1, 1))
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    if np.pi - ang < 1e-3:  # near pi: use the symmetric part
        w, V = np.linalg.eigh((Rm + Rm.T) / 2)
        ax = V[:, -1]
        return ax * ang * (1 if ax @ v >= 0 else -1)
    return v / (2 * np.sin(ang)) * ang


def _expm(v):
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class _Arm:
    def __init__(self, k):
        self.k, self.phase, self.th, self.path = k, "start", 0.0, None
        self.c_pos, self.c_R = None, None  # virtual command (sum of the deltas sent in contact)
        self.rate_f, self.prev_hand, self.pause_run, self.at_end, self.rel_pt = 0.0, None, 0, 0, None
        self.tilt_max, self.paused, self.waited = 0.0, 0, 0  # diagnostics


class HybridPB:
    carry_queue = False  # the bundle's policy.py resets every episode

    def __init__(self, log=False, depth=0.003, h_e=0.015, rate_lift=None, rate_fold=None, s_fixed=None, force_mode=None,
                 first_arm=None, roll_fixed=None):
        """cfg (tests only; the defaults are the controller): push depth and tip-edge height (m), aim rates (deg per
        step), fixed push positions [s_L, s_R], mode ("parallel" / "seq"), first arm, fixed rolls (deg)"""
        self.log, self.depth, self.h_e = log, depth, h_e
        self.rate_lift = np.radians(rate_lift) if rate_lift else RATE
        self.rate_fold = np.radians(rate_fold) if rate_fold else RATE
        self.s_fixed, self.force_mode, self.first_arm = s_fixed, force_mode, first_arm
        self.roll_fixed = None if roll_fixed is None else [np.radians(x) for x in roll_fixed]
        self.reset()

    def reset(self, episode=None):
        self.t, self.box, self.first, self.mode, self.reach, self.roll, self.tips = 0, None, None, None, None, None, None
        self.arms = [_Arm(0), _Arm(1)]
        self.trace = []

    # -- planning -------------------------------------------------------------------------------------------------
    def _perceive_box(self, obs):
        return locate_box(obs)

    def _hand_R(self, path, th=None):
        """target hand orientation: z down, y along the hinge (wrist offset toward the robot), rolled early if planned"""
        ax = path.axis
        y = ax if ax @ np.array([-1.0, 0, 0]) >= 0 else -ax
        z = np.array([0.0, 0.0, -1.0])
        R0 = np.stack([np.cross(y, z), y, z], 1)
        ph = path.roll(path.f["th0"] if th is None else th)
        if ph <= 0:
            return R0
        o = path.R @ np.array([path.f["o"], 0.0, 0.0])  # the flap's outward axis: the inner face keeps its plane
        # direction fixed by where the shoulder lies along the hinge (rotating -z about o by +a tilts the fingertips
        # along o x z): fingertips lean away from the shoulder, the wrist toward it
        sh = SHOULDER[path.k] - path.to_robot(0.0, 0.0)
        sgn = -1.0 if np.cross(o, z) @ sh > 0 else 1.0
        return _expm(o * (sgn * ph)) @ R0

    def _plan_points(self, k, cxy, yaw, s, roll0=0.0):
        P = FlapPath(k, cxy, yaw, s, self.depth, self.h_e, roll0)
        f = FLAP[k]
        X0, Z0 = P.XZ(f["th0"])
        ths = [None, None] + [np.radians(t) for t in (30, 45, 60, 75, 90, 135, 180)]
        tips = [P.to_robot(X0 + X_RUN, f["L"] * np.sin(f["th0"]) + Z_CLEAR), P.to_robot(X0 + X_RUN, Z0)]
        tips += [P.point(t) for t in ths[2:]]
        out = []
        for t, th in zip(tips, ths):
            Rt = self._hand_R(P, th)
            out.append((t - FT * Rt[:, 2], Rt))
        return out

    def _reach_cost(self, k, cxy, yaw, s, roll0=0.0):
        """largest distance from the arm's shoulder to a planned wrist point (run-in, lift, fold)"""
        return max(np.linalg.norm(p - 0.107 * R[:, 2] - SHOULDER[k]) for p, R in self._plan_points(k, cxy, yaw, s, roll0))

    def _plan(self, obs):
        est = self._perceive_box(obs)
        self.box = est
        cxy, yaw = est["xy"], est["yaw"]
        ROLL = [0.0, 0.0] if self.roll_fixed is None else list(self.roll_fixed)
        if self.s_fixed is not None:  # fixed push positions (tests)
            S, self.mode = [float(self.s_fixed[0]), float(self.s_fixed[1])], self.force_mode or "parallel"
        else:
            D, ROLL = [], []
            for k in (0, 1):  # reach of every push position; roll the early lift only if no position is comfortable
                opts = []
                for i, r in enumerate(ROLL_LIST if self.roll_fixed is None else [self.roll_fixed[k]]):
                    Dk = {s: self._reach_cost(k, cxy, yaw, s, r) for s in S_GRID}
                    opts.append((max(min(Dk.values()), D_COMF - 0.03) + 0.004 * i, float(r), Dk))
                    if i == 0 and min(Dk.values()) <= D_COMF:
                        break
                _, r, Dk = min(opts, key=lambda o: o[0])
                D.append(Dk)
                ROLL.append(r)
            best = None
            for sl in S_GRID:
                for sr in S_GRID:
                    if abs(sl - sr) >= D_SEP - 1e-9 and max(D[0][sl], D[1][sr]) <= D_COMF:
                        c = max(D[0][sl], D[1][sr])
                        if best is None or c < best[0]:
                            best = (c, sl, sr)
            if best is not None and self.force_mode != "seq":
                S, self.mode = [best[1], best[2]], "parallel"
            else:  # most comfortable positions, spread apart where the comfort limit allows; one arm finishes first
                S = [min(S_GRID, key=lambda s: D[k][s]) for k in (0, 1)]
                easy = int(np.argmin([D[0][S[0]], D[1][S[1]]]))
                ok = [s for s in S_GRID if D[easy][s] <= max(D_COMF, D[easy][S[easy]])]
                S[easy] = max(ok, key=lambda s: (abs(s - S[1 - easy]), -D[easy][s]))
                self.mode = "seq"
            self.reach = [round(D[k][S[k]], 3) for k in (0, 1)]
        if self.mode == "seq":  # the arm with the easier reach goes first (a far arm that struggles never blocks it)
            if self.first_arm is not None:
                self.first = self.first_arm
            elif self.reach is not None:
                self.first = int(np.argmin(self.reach))
            else:
                self.first = 1
        for A in self.arms:
            A.path = FlapPath(A.k, cxy, yaw, S[A.k], self.depth, self.h_e, ROLL[A.k])
            A.phase, A.th = "transit", FLAP[A.k]["th0"] - EXT
        self.roll = [round(float(np.degrees(x)), 1) for x in ROLL]

    def _pt(self, A, th):
        """fingertip target at path parameter th (below theta0: the horizontal run-in toward the tip edge)"""
        f = A.path.f
        if th >= f["th0"]:
            return A.path.point(th)
        X0, Z0 = A.path.XZ(f["th0"])
        return A.path.to_robot(X0 + X_RUN * (f["th0"] - th) / EXT, Z0)

    # -- one arm --------------------------------------------------------------------------------------------------
    def _arm(self, A, gp, gR, a, cp, cR):
        k = A.k
        tip = gp + FT * gR[:, 2]
        f = A.path.f
        th_start = f["th0"] - EXT
        Rt = self._hand_R(A.path, A.th if A.phase in ("push", "hold") else None)
        tilt = float(np.arccos(np.clip(gR[:, 2] @ Rt[:, 2], -1, 1)))  # deflection from the target (wrist overload)
        A.tilt_max = max(A.tilt_max, tilt)
        if A.phase in ("release", "away"):  # fingers open (the opening shows in the slip check before contact ends)
            if A.phase == "release":
                target = A.rel_pt + np.array([0, 0, REL_UP])
                if tip[2] > A.rel_pt[2] + REL_UP - 0.01:
                    A.phase = "away"
            else:  # its own side; height follows the other arm's fingertips (the scorer compares the hand heights)
                target = AWAY[k].copy()
                if self.tips is not None:
                    target[2] = float(np.clip(self.tips[1 - k][2], 1.02, 1.22))
            a[6 * k:6 * k + 3] = np.clip(KP * (target - tip), -VMAX, VMAX)
            a[6 * k + 3:6 * k + 6] = np.clip(KR * _rotvec(Rt @ gR.T), -RMAX, RMAX)
            a[12 + k] = 0.0
            return tip, target
        if A.phase in ("transit", "descend"):  # free motion: P control on the measured hand pose
            if A.phase == "transit":  # above the run-in start, Z_CLEAR over the open flap's tip
                X0, Z0 = A.path.XZ(f["th0"])
                target = A.path.to_robot(X0 + X_RUN, f["L"] * np.sin(f["th0"]) + Z_CLEAR)
                if np.linalg.norm(tip - target) < 0.02:
                    A.phase = "descend"
            else:
                target = self._pt(A, th_start)
                F1 = self.arms[self.first] if self.mode == "seq" else None
                wait = F1 is not None and self.first != k and F1.phase in ("transit", "descend", "push") and F1.th < TH_GO2
                if wait:  # off the flap, outside its tip: mirror the other arm's fingertip height
                    target = target.copy()
                    target[2] += float(np.clip(self.tips[1 - k][2] - target[2], *Z_WAIT))
                    A.waited += 1
                elif np.linalg.norm(tip - target) < 0.008:
                    A.phase, A.th = "push", th_start
                    A.c_pos, A.c_R = cp.copy(), cR.copy()
            e = target - tip
            vmax = VMAX_FAST if np.linalg.norm(e) > 0.05 else VMAX
            a[6 * k:6 * k + 3] = np.clip(KP * e, -vmax, vmax)
            a[6 * k + 3:6 * k + 6] = np.clip(KR * _rotvec(Rt @ gR.T), -RMAX, RMAX)
            a[12 + k] = 1.0
            return tip, target
        # contact phases: feed-forward virtual command along the path; the aim pauses when the hand lags or tilts
        lead_err = np.linalg.norm(A.c_pos - cp)
        A.dbg = (round(float(lead_err) * 1000, 1), round(float(np.degrees(tilt)), 1))
        if A.phase == "push":
            rate = self.rate_lift if A.th < np.pi / 2 else self.rate_fold
            lim = TH_END
            if self.mode == "seq" and self.first != k and self.arms[self.first].phase != "away":
                lim = CAP2
            if A.th >= lim - 1e-6:
                rate = 0.0
                A.waited += 1
            elif tilt > TILT_OK or lead_err > P_LAG:
                if A.th > np.radians(95) and tilt <= TILT_OK and lead_err <= LAG_HARD:
                    rate = min(rate, RATE_MIN_FOLD)
                else:
                    rate = 0.0
                A.paused += 1
                A.pause_run += 1
            else:
                A.pause_run = 0
            A.rate_f += RATE_SMOOTH * (rate - A.rate_f)
            if A.pause_run > PAUSE_MAX:  # guard against a command that no longer matches the robot (IK failure)
                A.c_pos, A.c_R, A.pause_run = cp.copy(), cR.copy(), 0
            A.th = min(lim, A.th + A.rate_f) if A.th < lim else max(A.th, min(A.th, lim))
            target_tip = self._pt(A, A.th)
            if A.th >= TH_END:
                A.at_end += 1
                if (lead_err < 0.006 and np.linalg.norm(target_tip - tip) < 0.008) or A.at_end > HOLD_TIMEOUT:
                    A.phase = "hold"
        if A.phase == "hold":  # press still (pressing on would wind the command up); first arm in seq mode: let go
            d = np.zeros(3)
            target_tip = tip
            A.prev_hand = None
            if self.mode == "seq" and self.first == k and self.arms[1 - k].phase not in ("hold", "release", "away"):
                A.phase, A.rel_pt = "release", tip.copy()
        else:
            target_hand = target_tip - FT * Rt[:, 2]
            ff = np.zeros(3) if A.prev_hand is None else (target_hand - A.prev_hand) * (KV_KP * 20)  # servo-lag feed-forward
            A.prev_hand = target_hand
            n_ff = np.linalg.norm(ff)
            if n_ff > FF_MAX:
                ff *= FF_MAX / n_ff
            d = np.clip(target_hand + ff - A.c_pos, -VMAX_PUSH, VMAX_PUSH)
        A.c_pos = A.c_pos + d
        a[6 * k:6 * k + 3] = d
        w = np.clip(KR * _rotvec(Rt @ gR.T), -RMAX, RMAX)  # measured-orientation feedback, command within R_DEV of target
        cR_new = _expm(w) @ A.c_R
        dev = _rotvec(cR_new @ Rt.T)
        nd = np.linalg.norm(dev)
        if nd > R_DEV:
            w = np.clip(_rotvec(_expm(dev * (R_DEV / nd)) @ Rt @ A.c_R.T), -RMAX, RMAX)
            cR_new = _expm(w) @ A.c_R
        A.c_R = cR_new
        a[6 * k + 3:6 * k + 6] = w
        a[12 + k] = 1.0  # fingers closed (a compact two-pad pusher; at reset they are closed)
        return tip, target_tip

    def act(self, obs):
        self.t += 1
        a = np.zeros(14, np.float32)
        q = np.asarray(obs["proprioception"], np.float64)  # [q_L(7) fingers_L(2) q_R(7) fingers_R(2) qdot_L(9) qdot_R(9)]
        G = [fk(q[9 * k:9 * k + 7], k) for k in (0, 1)]
        C = [fk(q[9 * k:9 * k + 7] + KV_KP * q[18 + 9 * k:25 + 9 * k], k) for k in (0, 1)]  # unloaded-command estimate
        self.tips = [G[k][0] + FT * G[k][2][:, 2] for k in (0, 1)]
        if self.box is None:
            self._plan(obs)
        rec = []
        for A in self.arms:
            gp, _, gR = G[A.k]
            tip, target = self._arm(A, gp, gR, a, C[A.k][0], C[A.k][2])
            rec.append((A.phase, round(float(np.degrees(A.th)), 1), np.round(tip, 3).tolist(), np.round(target, 3).tolist(),
                        getattr(A, "dbg", None)))
        if self.log:
            self.trace.append((self.t, rec))
        return a
