"""Hybrid StackTwoBlocks policy v3 (Oct 24 21:00): both arms act, so the episode reaches the task's stage 2 (both blocks held
at the same time; the scorer's task progression counts it, 30% of the outcome term) and the arms move together (the
coordination metrics compare the two arms' velocities). Perception and kinematics as v2 (hybrid_s2b.py).

Plan (from the perceived blocks at the start):
  Mode A (the stacking arm s reaches both blocks): s picks P and stacks it on B; meanwhile the helper arm b grips B where it
         stands, holds it until s holds P (both held: stage 2), opens, rises and goes home; s waits above until b is clear
         and B has been seen again, then stacks.
  Mode B (no single arm reaches both: blocks far apart on opposite sides): both arms pick their own block; b sets B down at
         a meeting point M mid-table and goes home; s, holding P at its side meanwhile, then stacks P on B.
Actions: 14-D end-effector deltas [dp_L dr_L | dp_R dr_R | g_L g_R] (world frame), grippers 0 open / 1 closed."""
import numpy as np

try:
    from .hybrid_s2b import fk, locate, wrap, BASE_Y, H_GRASP, H_APPROACH, H_CARRY, VMAX, KP, RMAX, KR, NTOP_MIN, Z_TABLE
except ImportError:
    from hybrid_s2b import fk, locate, wrap, BASE_Y, H_GRASP, H_APPROACH, H_CARRY, VMAX, KP, RMAX, KR, NTOP_MIN, Z_TABLE

REACH_A = 0.64  # Mode A if the stacking arm's pick and place both lie within this horizontal distance of its base
H_GRASP3 = 0.110  # v2's 0.100 put the fingertip pads on the table (an environment collision for the scorer)
VMAX_FAST = 0.016  # transit speed while > 5 cm from the target (v2: 0.012 everywhere)
ASIDE = 0.12  # after letting go, arm b moves this far (xy) from B toward its own side
HOME = (np.array([0.40, 0.32, 1.15]), np.array([0.40, -0.32, 1.15]))
CLEAR_XY = 0.14  # the stacking arm waits this far (xy) from B while the other arm works there


def _reach(arm, xy):
    return float(np.hypot(xy[0], xy[1] - BASE_Y[arm]))


class _Arm:
    def __init__(self, k):
        self.k, self.phase, self.count, self.off, self.zhist, self.recov, self.hold_t = k, "idle", 0, None, [], 0, 0
        self.aside = None


class HybridS2B3:
    carry_queue = False  # the bundle's policy.py resets every episode

    def __init__(self, log=False):
        self.log = log
        self.reset()

    def reset(self, episode=None):
        self.t, self.mode, self.s, self.b, self.P, self.B, self.M = 0, None, None, None, None, None, None
        self.blk = [None, None]
        self.arms = [_Arm(0), _Arm(1)]
        self.dirtyB = False  # B was gripped or moved by arm b: lower onto it only after a fresh estimate
        self.b_done_t = None  # step at which arm b reached "done"
        self.z_rest = Z_TABLE
        self.trace = []

    # -- helpers --------------------------------------------------------------------------------------------------
    @property
    def arm(self):
        return self.s

    @property
    def phase(self):
        return self.arms[self.s].phase if self.s is not None else "start"

    @property
    def recoveries(self):
        return sum(a.recov for a in self.arms)

    @property
    def plan(self):
        return [self.mode] if self.mode == "A" else [self.mode, self.mode]

    def _holding(self, obs, k):
        f = float(np.asarray(obs["proprioception_grippers"])[k])
        return 0.15 < f < 0.8

    def _perceive(self, obs, G, held):
        est = locate(obs)
        if not est:
            return
        if self.P in held and self.B not in held and self.arms[self.b].phase in ("home", "done"):
            low = [e for e in est if e["z"] < Z_TABLE + 0.03 and e["ntop"] >= NTOP_MIN
                   and not any(np.linalg.norm(g[0][:2] - e["xy"]) < 0.10 and g[0][2] < e["z"] + 0.16 for g in G)]
            if len(low) == 1:
                self.blk[self.B] = low[0]
                self.dirtyB = False
            return
        for i in (0, 1):
            if i in held:
                continue
            b = self.blk[i]
            e = min(est, key=lambda e: np.linalg.norm(e["xy"] - b["xy"]))
            if np.linalg.norm(e["xy"] - b["xy"]) > 0.04 or e["ntop"] < NTOP_MIN:
                continue
            if any(np.linalg.norm(g[0][:2] - b["xy"]) < 0.10 and g[0][2] < b["z"] + 0.16 for g in G):
                continue
            self.blk[i] = e
            if i == self.B and self.dirtyB and self.arms[self.b].phase in ("home", "done"):
                self.dirtyB = False

    def _plan(self, G):
        best = None
        for s in (0, 1):
            for P in (0, 1):
                B = 1 - P
                if _reach(s, self.blk[P]["xy"]) > REACH_A or _reach(s, self.blk[B]["xy"]) > REACH_A:
                    continue
                cost = np.linalg.norm(G[s][0][:2] - self.blk[P]["xy"]) + np.linalg.norm(self.blk[P]["xy"] - self.blk[B]["xy"]) \
                    + 0.5 * np.linalg.norm(G[1 - s][0][:2] - self.blk[B]["xy"])
                if best is None or cost < best[0]:
                    best = (cost, s, P, B)
        if best is not None:
            _, self.s, self.P, self.B = best
            self.mode = "A"
        else:
            own = [int(np.argmin([_reach(k, self.blk[i]["xy"]) for k in (0, 1)])) for i in (0, 1)]
            if own[0] == own[1]:
                own[1] = 1 - own[0]
            # the stacking arm: the one nearer its own block (it waits holding it)
            P = int(np.argmin([np.linalg.norm(G[own[i]][0][:2] - self.blk[i]["xy"]) for i in (0, 1)]))
            self.P, self.B, self.s = P, 1 - P, own[P]
            self.mode = "B"
            bx = float(np.clip(self.blk[self.B]["xy"][0], 0.45, 0.55))
            My = BASE_Y[self.s] + np.sign(self.blk[self.B]["xy"][1] - BASE_Y[self.s]) * 0.32
            self.M = np.array([bx, My])
        self.b = 1 - self.s
        if self.mode == "A" and np.linalg.norm(self.blk[self.P]["xy"] - self.blk[self.B]["xy"]) < 0.12:
            self.arms[self.b].phase = "late"  # blocks close together: grip B only once P is lifted out of the way
        else:
            self.arms[self.b].phase = "approach"
        self.arms[self.s].phase = "approach"

    # -- one arm's step -------------------------------------------------------------------------------------------
    def _pick(self, A, gp, gy, gR, i, holding, nxt):
        """approach/descend/grasp block i; returns (target, yaw_ref, grip); on a held grasp sets A.phase = nxt"""
        X = self.blk[i]
        pp, py = np.r_[X["xy"], X["z"]], X["yaw"]
        grip = 0.0
        if A.phase == "approach":
            target, yaw_ref = np.r_[pp[:2], pp[2] + H_APPROACH], py
            if np.linalg.norm(target - gp) < 0.012 and abs(wrap(py - gy, np.pi / 2)) < np.radians(6):
                A.phase = "descend"
        elif A.phase == "descend":
            target, yaw_ref = np.r_[pp[:2], pp[2] + H_GRASP3], py
            if abs(wrap(py - gy, np.pi / 2)) > np.radians(12) and gp[2] < pp[2] + H_GRASP3 + 0.02:
                A.phase = "approach"
            elif abs(gp[2] - target[2]) < 0.006 and np.linalg.norm(gp[:2] - pp[:2]) < 0.01:
                A.phase, A.count = "grasp", 0
        elif A.phase == "grasp":
            target, yaw_ref, grip = gp, None, 1.0
            A.count += 1
            if A.count >= 10:
                if holding:
                    rel = gR.T @ (pp - gp)
                    rel[1] = 0.0  # the fingers centre the block along the gripper's y axis
                    A.off = gR @ rel
                    A.phase, A.count = nxt, 0
                else:
                    A.recov += 1
                    A.phase, A.count = "reopen", 0
        else:  # reopen: open, rise, re-approach
            target, yaw_ref = np.r_[gp[:2], pp[2] + H_APPROACH], None
            A.count += 1
            if A.count >= 6 and gp[2] > pp[2] + H_APPROACH - 0.03:
                A.phase = "approach"
        return target, yaw_ref, grip

    def _place(self, A, gp, pp, dest_xy, z_place, nxt):
        """carry the held block over dest_xy at carry height, lower to z_place, open; then A.phase = nxt"""
        grip = 1.0
        if A.phase == "carry":
            target = np.r_[gp[:2] + (dest_xy - pp[:2]), Z_TABLE + H_CARRY]
            if np.linalg.norm(pp[:2] - dest_xy) < 0.008:
                A.phase, A.zhist = "lower", []
        elif A.phase == "lower":
            target = np.r_[gp[:2] + (dest_xy - pp[:2]), gp[2] - (pp[2] - z_place)]
            A.zhist.append(gp[2])
            blocked = len(A.zhist) > 6 and A.zhist[-7] - gp[2] < 0.002 and pp[2] < z_place + 0.03
            if pp[2] < z_place + 0.004 or blocked:
                A.phase, A.count = "release", 0
        else:  # release
            target, grip = gp, 0.0
            A.count += 1
            if A.count >= 10:
                A.phase, A.count = nxt, 0
        return target, None, grip

    def _go_home(self, A, gp, base_z):
        """after letting go of B: straight up until the fingers clear it, then step aside toward this arm's side"""
        Bxy = self.blk[self.B]["xy"]
        if A.phase == "rise":
            target = np.r_[gp[:2], base_z + H_APPROACH]
            if gp[2] > base_z + H_APPROACH - 0.01:
                A.phase = "home"
                A.aside = np.r_[Bxy[0] - 0.03, Bxy[1] + np.sign(BASE_Y[A.k]) * ASIDE, base_z + H_APPROACH]
        elif A.phase == "home":
            target = A.aside
            if np.linalg.norm(gp[:2] - Bxy) > ASIDE - 0.01:
                A.phase = "done"
        else:  # done
            target = gp
        return target, None, 0.0

    # -- policy -------------------------------------------------------------------------------------------------------
    def act(self, obs):
        self.t += 1
        a = np.zeros(14, np.float32)
        G = [fk(np.asarray(obs["proprioception"], np.float64)[9 * k:9 * k + 7], k) for k in (0, 1)]
        if self.s is None:  # start: no-op; perceive and plan
            est = locate(obs)
            if len(est) < 2:
                est = (est * 2)[:2] if est else [{"xy": np.array([0.5, 0.1]), "z": Z_TABLE, "yaw": 0.0, "ntop": 0, "n": 0},
                                                 {"xy": np.array([0.5, -0.1]), "z": Z_TABLE, "yaw": 0.0, "ntop": 0, "n": 0}]
            self.blk = list(est)
            self._plan(G)
            return a
        S, Bm = self.arms[self.s], self.arms[self.b]
        hold_s, hold_b = self._holding(obs, self.s), self._holding(obs, self.b)
        held = set()
        if S.off is not None:
            held.add(self.P)
        if Bm.off is not None or Bm.phase in ("grasp", "hold"):
            held.add(self.B)
        self._perceive(obs, G, held)

        # ---- arm b: helper (A) or mover (B) ----
        gp, gy, gR = G[self.b]
        if Bm.phase == "late":  # wait (still) until the stacking arm has lifted P clear of the table
            if S.phase in ("wait", "carry") or (S.phase == "lift" and G[self.s][0][2] > Z_TABLE + H_CARRY - 0.06):
                Bm.phase = "approach"
            elif self.t > 150:
                Bm.phase = "done"
        if Bm.phase == "late":
            tb = (gp, None, 0.0)
        elif Bm.phase in ("approach", "descend", "grasp", "reopen"):
            if Bm.phase == "grasp":
                self.dirtyB = True
            nxt = "hold" if self.mode == "A" else "lift"
            tb = self._pick(Bm, gp, gy, gR, self.B, hold_b, nxt)
            if self.mode == "A" and Bm.phase == "reopen":  # a helper miss: give up the helper role
                Bm.phase, Bm.count = "release", 0
        elif Bm.phase == "hold":  # helper: keep B gripped until the stacking arm holds P (both held)
            tb = (gp, None, 1.0)
            Bm.count += 1
            if (S.off is not None and hold_s and S.phase in ("lift", "wait", "carry")) or Bm.count > 60:
                Bm.hold_t += 1
                if Bm.hold_t >= 3 or Bm.count > 60:
                    Bm.phase, Bm.count = "release", 0
        elif Bm.phase == "release" and self.mode == "A":
            tb = (gp, None, 0.0)
            Bm.count += 1
            f = float(np.asarray(obs["proprioception_grippers"])[self.b])
            if (Bm.count >= 5 and f > 0.85) or Bm.count >= 15:
                Bm.phase, Bm.off = "rise", None
        elif Bm.phase == "lift":  # mover
            tb = (np.r_[gp[:2], Z_TABLE + H_CARRY], None, 1.0)
            if gp[2] > Z_TABLE + H_CARRY - 0.012:
                Bm.phase = "carry"
            if not hold_b:
                Bm.recov += 1; Bm.off = None; Bm.phase, Bm.count = "reopen", 0
        elif Bm.phase in ("carry", "lower", "release"):  # mover: set B down at M
            ppb = gp + Bm.off
            if Bm.phase != "release" and not hold_b:
                Bm.recov += 1; Bm.off = None; Bm.phase, Bm.count = "reopen", 0
                tb = (gp, None, 0.0)
            else:
                was = Bm.phase
                tb = self._place(Bm, gp, ppb, self.M, self.z_rest + 0.002, "rise")
                if was == "release" and Bm.phase == "rise":
                    self.blk[self.B] = {"xy": ppb[:2].copy(), "z": Z_TABLE, "yaw": 0.0, "ntop": 0, "n": 0}
                    Bm.off = None
                    self.dirtyB = True
        else:  # rise / home / done
            tb = self._go_home(Bm, gp, Z_TABLE)
        if Bm.phase == "done" and self.b_done_t is None:
            self.b_done_t = self.t
        self._write(a, self.b, gp, gy, *tb)

        # ---- arm s: stack P on B ----
        gp, gy, gR = G[self.s]
        Bx = self.blk[self.B]
        if S.phase in ("approach", "descend", "grasp", "reopen"):
            ts = self._pick(S, gp, gy, gR, self.P, hold_s, "lift")
            if S.phase == "lift":
                self.z_rest = float(self.blk[self.P]["z"])
        elif S.phase in ("lift", "wait", "carry", "lower") and not hold_s:
            S.recov += 1; S.off = None; S.phase, S.count = "reopen", 0
            ts = (gp, None, 0.0)
        elif S.phase == "lift":
            ts = (np.r_[gp[:2], Z_TABLE + H_CARRY], None, 1.0)
            if gp[2] > Z_TABLE + H_CARRY - 0.012:
                S.phase = "wait"
        elif S.phase == "wait":  # until arm b is out of the way and B has been seen again
            pp = gp + S.off
            if self.mode == "A":  # B stays put: wait CLEAR_XY from it on this side (back off if closer)
                d = pp[:2] - Bx["xy"]
                n = np.linalg.norm(d)
                stop = Bx["xy"] + d / max(n, 1e-6) * CLEAR_XY
            else:  # B is being moved to M by arm b: hold P on this arm's own side
                stop = np.array([pp[0], BASE_Y[self.s] * 0.5]) if abs(pp[1] - BASE_Y[self.s]) > 0.15 else pp[:2]
            ts = (np.r_[gp[:2] + (stop - pp[:2]), Z_TABLE + H_CARRY], None, 1.0)
            far = np.linalg.norm(G[self.b][0][:2] - Bx["xy"]) > ASIDE - 0.02
            ready = Bm.phase in ("home", "done") and far and not self.dirtyB
            if ready or (self.b_done_t is not None and self.t - self.b_done_t > 20):
                S.phase = "carry"
        elif S.phase in ("carry", "lower", "release"):
            pp = gp + S.off
            ts = self._place(S, gp, pp, Bx["xy"], Bx["z"] + 0.042, "retreat")
        else:  # retreat
            ts = (np.r_[gp[:2], Z_TABLE + H_CARRY], None, 0.0)
        self._write(a, self.s, gp, gy, *ts)
        if self.log:
            self.trace.append((self.t, self.mode, S.phase, Bm.phase, G[self.s][0].round(3).tolist(), G[self.b][0].round(3).tolist(),
                               int(hold_s), int(hold_b)))
        return a

    @staticmethod
    def _write(a, k, gp, gy, target, yaw_ref, grip):
        off = 6 * k
        e = np.asarray(target) - gp
        vmax = VMAX_FAST if np.linalg.norm(e) > 0.05 else VMAX
        a[off:off + 3] = np.clip(KP * e, -vmax, vmax)
        if yaw_ref is not None and np.isfinite(yaw_ref):
            a[off + 5] = np.clip(KR * wrap(yaw_ref - gy, np.pi / 2), -RMAX, RMAX)
        a[12 + k] = grip
