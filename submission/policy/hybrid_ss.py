"""Hybrid StackSingleBookShelf controller (Oct 28, v3): classical RGB-D perception of the book + scripted control. Only the
standard observations are used (head depth image, proprioception, gripper openings); nothing is read from the simulator.

Task: lift the 4 kg book (9.4 x 13.8 x 2.65 cm) from the counter onto a shelf board and let go (success = the book touches
the lower or upper board and no gripper holds it; the episode ends there). The gripper (8 cm) cannot span the book's width,
so it is pinched across its thickness at the spine, which must overhang the counter's front edge (as in the human demos).

Plan (one arm, the one on the book's side; the other arm lowers to mid height and waits):
 1. perceive the book (head depth, ss_perc);
 2. if the spine overhangs the counter edge by less than D_MIN under the centre of mass, push the far face toward the robot
    with the front finger of the open gripper (closing axis along x: one finger touches, so the gripper never "holds" the
    book; the gripper is either open or closed, and closed fingers squeezed by a push touch each other, which the scorer
    counts as a self-collision). The lower shelf board hangs 32 cm above the counter with its front edge 5-9 cm behind the
    book's far face, so the hand crosses over the book pitched 25 deg forward (wrist back) and pushes pitched 15 deg (palm
    clear of the book, wrist clear of the board). An integral correction removes the ~1 cm deflection of the arm; the book
    is then perceived again and the push extended if needed;
 3. pinch the spine at the point under the centre of mass with the hand tilted (closing axis PSI from vertical, lower pad
    under the bottom edge, upper pad on the top face). Preferred: psi 55 deg with the start orientation simply pitched
    forward (a small turn from the push pose); a planning IK checks every candidate's path for joint-range margin and
    sets segment durations that keep the joints under their rate limit;
 4. close, check the opening (holding = 0.15-0.8), lift while turning the book to ~PSI_CARRY nose-down, carry it in until
    its far edge is over the lower board, lower until the far edge touches the board, open.
Recovery: a missed grasp or a drop -> open, back off, perceive, push/pinch again.

Frames: robot frame of hybrid_s2b's fk and CAMS (world + (0.15, 0, 0.1)); actions are world-frame deltas, so the offset
never matters. Motion: minimum-jerk Bezier curves / segments of the flange pose, sent as feed-forward deltas of the
reference (the environment integrates the deltas on the commanded pose), plus a bounded correction toward the reference
while it is still. Paths are checked at plan time against the book, the counter and the shelf boards with a box model of
the hand (palm and fingers, from the asset).
"""
import numpy as np

try:
    from .hybrid_s2b import fk
    from . import ss_lib as L
    from . import ss_ik as IK
    from . import ss_perc as PC
except ImportError:
    from hybrid_s2b import fk
    import ss_lib as L
    import ss_ik as IK
    import ss_perc as PC

COM_B = np.array([-0.0043, -0.0113])  # book centre of mass in the body frame (xy), from the asset
FIN_OUT = 0.0663  # open finger (the gripper is open or closed only: the action is rounded): outer face from the hand
# axis along the closing axis; finger body from 0.0585 to 0.1123 along the approach axis (asset)
FIN_TIP = 0.1123
LOWER_BOT = 1.1087  # lower shelf board bottom (robot frame)
# hand model in the gripper frame (x, y = closing axis, z = approach), corners of the palm box and the open fingers
_PALM = np.array([[sx * 0.0316, sy * 0.1004, z] for sx in (-1, 1) for sy in (-1, 1) for z in (-0.026, 0.066)])
_FING = np.array([[sx * 0.0105, sg * yy, z] for sx in (-1, 1) for sg in (-1, 1) for yy in (0.0425, 0.0663) for z in (0.0585, FIN_TIP)])
HAND_PTS = np.r_[_PALM, _FING]
DEFAULTS = dict(
    D_MIN=0.015,  # spine overhang (under the COM) needed for the pinch
    D_PUSH=0.021,  # target overhang of a push
    H_PUSH=0.006,  # fingertip height above the counter while pushing
    PITCH=25.0,  # deg, forward pitch while crossing over the book
    PITCH_P=15.0,  # deg, forward pitch while pushing (wrist clear of the lower board, palm clear of the book)
    BACK=0.07,  # insertion start: distance back along the approach axis from the grasp pose
    X_FAR=0.75,  # book far edge x when placed (lower board front edge 0.6656)
    PSI_CARRY=20.0,  # deg: the hand turns the book from ~PSI to ~PSI_CARRY nose-down while lifting
    DROOP=4.0,  # deg: the book hangs this much more nose-down than the pads (measured)
    CLEAR=0.035,  # far edge height above the board top while carrying in
    Z_PRESS=0.004,  # far edge commanded this much below the board top
    Z_IDLE=1.08,  # the idle arm's hand height (coordination: wrist heights)
    T_CLOSE=8,  # steps with the gripper closing before the check
    VMAX=0.05,  # max position delta per step (feed-forward + correction)
    KP=0.3, FB_MAX=0.015,  # correction toward the reference while it is still
    KI_PUSH=0.25, FB_PUSH=0.025,  # integral correction while pushing
    BASE=0.3, SPEED=1.7, ROT_SPEED=0.45,  # duration = BASE s + SPEED * path length (m) + ROT_SPEED * rotation (rad)
    SPEED_LOAD=2.4,  # s/m while carrying the 4 kg book
    LAG=0.025,  # m: the reference pauses while the hand trails it by more than this along the motion (carrying)
    KI_LOAD=0.1, FB_LOAD=0.035,  # integral correction of the sag while carrying
    MAX_TRIES=3,
    MARGIN=0.35,  # rad: smallest joint-range margin accepted along a planned pinch, lift and carry
    PSI_MAX=60,  # deg: pinch tilts tried from PSI_MAX down to 45 in 5 deg steps (60: the pads still span the pinch)
)


class HybridSS:
    carry_queue = False  # the bundle's policy.py resets every episode

    def __init__(self, log=False, **cfg):
        self.log = log
        self.c = dict(DEFAULTS)
        for k, v in cfg.items():
            if k not in self.c:
                raise KeyError(f"unknown cfg key {k}")
            self.c[k] = v
        self.book_source = None  # analysis only: a test harness may supply ground-truth book poses (callable)
        self.force_pinch = None  # analysis only: restrict the pinch candidates to one (psi_deg, daz_deg, flip)
        self.reset()

    def reset(self, episode=None):  # the episode argument is not used
        self.t = 0
        self.phase, self.i_phase = "start", 0
        self.k = None
        self.segs = [None, None]
        self.grip = [0.0, 0.0]
        self.fb = np.zeros((2, 3))
        self.unwind = np.zeros((2, 3))
        self.book = None
        self.push = None
        self.plan = None
        self.tries = 0
        self.pushes = 0
        self.ext = 0
        self.looks = 0
        self.events = []
        self.trace = []
        self.zh = []

    # -- perception -----------------------------------------------------------------------------------------------
    def _perceive(self, obs, G, prior=None):
        if self.book_source is not None:
            return self.book_source()
        gq = np.asarray(obs["proprioception_grippers"], np.float64)
        boxes = []
        for k in (0, 1):
            p, _, R = G[k]
            boxes += PC.hand_boxes(p, R, gq[k])
        r = PC.locate_book(obs, boxes=boxes, prior=prior)
        if not r.get("ok"):
            return None
        return {"xy": np.asarray(r["xy"], float), "yaw": float(r["yaw"])}

    @staticmethod
    def _frame(book):
        c, s = np.cos(book["yaw"]), np.sin(book["yaw"])
        return np.array([c, s]), np.array([-s, c]), np.array([[c, -s], [s, c]])

    def _spine_point(self, book):
        """spine bottom-edge point under the centre of mass (robot xy)"""
        u, t, R2 = self._frame(book)
        return book["xy"] + R2 @ np.array([PC.TOP_X[0], COM_B[1]])

    def _overhang(self, book):
        return L.X_EDGE - self._spine_point(book)[0]

    # -- geometry checks ------------------------------------------------------------------------------------------
    def _hand_clear(self, p, R):
        """smallest clearance (m) of the hand model at flange pose (p, R) to the book (as perceived), the counter top and
        the two shelf boards; negative = intrusion"""
        W = HAND_PTS @ R.T + p
        clr = np.inf
        b = self.book
        if b is not None:
            c, s = np.cos(b["yaw"]), np.sin(b["yaw"])
            rel = W[:, :2] - b["xy"]
            bx, by = rel @ np.array([c, s]), rel @ np.array([-s, c])
            over = (bx > PC.TOP_X[0]) & (bx < PC.TOP_X[1]) & (by > PC.TOP_Y[0]) & (by < PC.RIB_Y[1])
            if over.any():
                clr = min(clr, float(np.min(W[over, 2] - (L.Z_COUNTER + L.BOOK_T))))
        on_counter = W[:, 0] > L.X_EDGE - 0.004
        if on_counter.any():
            clr = min(clr, float(np.min(W[on_counter, 2] - L.Z_COUNTER)))
        shelf = W[:, 0] > L.SHELF_X[0] - 0.006
        if shelf.any():
            z = W[shelf, 2]
            below_lower = LOWER_BOT - z  # clear if positive (under the lower board)
            between = np.minimum(z - L.LOWER_TOP, L.UPPER_BOT - z)  # clear if positive (between the boards)
            clr = min(clr, float(np.min(np.maximum(below_lower, between))))
        return clr

    def _curve_clear(self, curve, m=20, s0=0.15):
        """smallest hand clearance along a curve, sampled by its path parameter from s0 (the start may touch the book)"""
        return min(self._hand_clear(curve.point(sv), L.expm(sv * curve.rv) @ curve.R0) for sv in np.linspace(s0, 1.0, m))

    # -- planning -------------------------------------------------------------------------------------------------
    def _dur(self, length, R0=None, R1=None, base=None):
        r = float(np.linalg.norm(L.rotvec(R1 @ R0.T))) if R0 is not None else 0.0
        base = self.c["BASE"] if base is None else base
        return max(4, int(round((base + self.c["SPEED"] * length + self.c["ROT_SPEED"] * r) / L.DT)))

    def _push_plan(self, G):
        c = self.c
        b = self.book
        u, t, R2 = self._frame(b)
        over = self._overhang(b)
        if over >= c["D_MIN"]:
            return None
        shift = c["D_PUSH"] - over
        com = b["xy"] + R2 @ COM_B
        yc = float(com[1])
        ff = b["xy"] + PC.TOP_X[1] * u  # a point of the far face (fore-edge) line, direction t
        x_hit = ff[0] + (yc - ff[1]) * (t[0] / t[1])  # far face x on the push line y = yc
        x_clear = x_hit + 0.011 * abs(t[0] / t[1])  # far face x within the finger's width (+-1.1 cm in y)
        fx = FIN_OUT
        k = self.k
        gp, gy, gR = G[k]
        ya = gR[:, 1]
        Rv = L.rz(np.pi - np.arctan2(ya[1], ya[0])) @ gR  # vertical hand, closing axis along world x (y axis = -x)

        def pitched(deg):  # fingertips forward, wrist back: approach axis (sin th, 0, -cos th)
            th = np.radians(deg)
            return np.array([[np.cos(th), 0, -np.sin(th)], [0, 1, 0], [np.sin(th), 0, np.cos(th)]]) @ Rv

        Rc, Rp = pitched(c["PITCH"]), pitched(c["PITCH_P"])
        z_top = L.Z_COUNTER + L.BOOK_T
        # front finger (toward the robot): outer face at y = fx in the gripper frame, bottom at z = FIN_TIP
        corner_p = Rp @ np.array([0.0, fx, FIN_TIP])
        z_push = L.Z_COUNTER + c["H_PUSH"] - corner_p[2]
        # the finger's slanted outer face meets the book's far top edge: flange x for the face through (x_e, z_top)
        thp = np.radians(c["PITCH_P"])

        def fx_at(x_e):
            return x_e + (fx + (z_top - z_push) * np.sin(thp)) / np.cos(thp)

        S = np.array([fx_at(x_clear) + 0.008, yc, z_push])  # push start
        E = np.array([fx_at(x_hit - shift) - 0.004, yc, z_push])  # push end (4 mm for the finger's give)
        corner_c = min((Rc @ np.array([0.0, sg * fx, FIN_TIP]))[2] for sg in (-1, 1))
        z_cross = z_top + 0.015 - corner_c  # lowest point of the pitched open hand 1.5 cm above the book
        B = np.array([S[0], yc, z_cross])
        return dict(Rv=Rv, Rc=Rc, Rp=Rp, B=B, S=S, E=E, shift=shift, over=over, x_hit=x_hit, fx=fx, z_cross=z_cross)

    def _pinch_plan(self, q_arm):
        plan = self._pinch_plan1(q_arm, self.c["MARGIN"])
        if plan is None:  # fallback: accept a smaller joint margin rather than giving up
            plan = self._pinch_plan1(q_arm, 0.2)
            if plan is not None:
                self.events.append((self.t, "pinch margin relaxed"))
        return plan

    def _pinch_plan1(self, q_arm, margin):
        """best reachable tilted pinch at the spine point under the COM (planning IK from the current joints). Preferred:
        flip 1 (the start orientation pitched forward by 90 - psi: a small turn from the push pose), psi = PSI_MAX, the
        approach perpendicular to the spine; other candidates only if that one lacks joint margin"""
        b = self.book
        P = np.r_[self._spine_point(b), L.Z_COUNTER]
        k = self.k
        cands = []
        for flip in ((self.force_pinch[2],) if self.force_pinch else (1, 0)):
            psis = tuple(range(int(self.c["PSI_MAX"]), 44, -5))  # steepest first: the smallest turn from the push pose
            for psi_d in ((self.force_pinch[0],) if self.force_pinch else psis):
                for daz_d in ((self.force_pinch[1],) if self.force_pinch else (0, -10, 10, -20, 20)):
                    cost0 = 0.3 * (1 - flip) + 0.02 * abs(daz_d) / 10 + 0.02 * (self.c["PSI_MAX"] - psi_d) / 5
                    cands.append((cost0, psi_d, daz_d, flip))
        cands.sort()
        for cost0, psi_d, daz_d, flip in cands:
            psi = np.radians(psi_d)
            yaw_a = b["yaw"] + np.radians(daz_d)
            ua = np.array([np.cos(yaw_a), np.sin(yaw_a)])
            R = L.pinch_frame(psi, ua, flip=bool(flip))
            a_ax = R[:, 2]
            c_ax = R[:, 1] if R[2, 1] > 0 else -R[:, 1]
            # lower pad tip region must stay in front of the counter's front face
            tipx = P[0] + L.PAD_L * a_ax[0] + 0.010 * abs(R[0, 0])
            if tipx > L.X_EDGE - 0.003:
                continue
            g = (L.BOOK_T + L.PAD_L * np.sin(psi)) / np.cos(psi)
            F = P + 0.5 * g * c_ax - L.PAD_C * a_ax
            via = F - self.c["BACK"] * a_ax
            Rk = L.pinch_frame(np.radians(self.c["PSI_CARRY"]), ua, flip=bool(flip))
            up, carry = self._carry_poses(F, ua, psi)
            _, ep, er, mg, nmin = IK.path_plan(q_arm, k, [(via, R), (F, R), (up, Rk), (carry, Rk)], n_sub=5)
            if ep > 0.003 or er > 0.03 or mg < margin:
                continue
            return dict(R=R, a=a_ax, c=c_ax, F=F, via=via, P=P, psi=psi, daz=daz_d, flip=flip, margin=mg,
                        Rk=Rk, up=up, carry=carry, ua=ua, nmin=nmin)
        return None

    def _carry_poses(self, F, ua, psi):
        """flange poses for the lift (far edge CLEAR above the lower board top) and the carry (far edge at X_FAR) with
        the hand turned to PSI_CARRY. F: grasp flange position; ua: horizontal approach direction"""
        far = self._far_edge_offset(np.radians(self.c["PSI_CARRY"] + self.c["DROOP"]))
        pad0 = F[:2] + L.PAD_C * np.cos(psi) * ua  # the pads (and the book) swing about the flange: keep the pad xy
        pc = np.radians(self.c["PSI_CARRY"])
        Fxy = pad0 - L.PAD_C * np.cos(pc) * ua
        z = L.LOWER_TOP + self.c["CLEAR"] - far[1]
        up = np.r_[Fxy, z]
        fe = Fxy + far[0] * ua
        carry = np.r_[Fxy + np.array([self.c["X_FAR"] - fe[0], 0.0]), z]
        return up, carry

    @staticmethod
    def _far_edge_offset(psi):
        """book far bottom edge relative to the flange (along the approach azimuth, vertical) for the book hanging
        aligned with the pads at tilt psi; checked against the simulator in development"""
        a2 = np.array([np.cos(psi), -np.sin(psi)])  # approach axis in the (horizontal along u, vertical) plane
        c2 = np.array([np.sin(psi), np.cos(psi)])
        lower = L.PAD_C * a2 - 0.5 * L.BOOK_T * c2  # the spine's bottom edge sits at the lower pad centre
        return lower + (PC.TOP_X[1] - PC.TOP_X[0]) * a2  # far bottom edge: one book width along the pad face

    def _carry_curve(self, i, p0, R0):
        """lift + carry: Bezier from the grasp to the carry pose, rising first; the control point is raised until the
        book's far edge stays above the lower board wherever it is over the board"""
        P = self.plan
        c = self.c
        ua = P["ua"]
        hi = P["carry"][2]
        for _ in range(4):
            cv = L.Curve(i, [p0, np.r_[p0[:2], hi], np.r_[p0[:2], hi], P["carry"]], R0, P["Rk"], 10)
            ok = True
            for j in range(21):
                s = j / 20
                q = cv.point(s)
                phi = P["psi"] + (np.radians(c["PSI_CARRY"]) - P["psi"]) * s + np.radians(c["DROOP"])
                far = self._far_edge_offset(phi)
                fe = q[:2] + far[0] * ua
                if fe[0] > L.SHELF_X[0] - 0.02 and q[2] + far[1] < L.LOWER_TOP + 0.012 + c["LAG"]:
                    ok = False
            if ok:
                break
            hi += 0.03
        n = max(int(round((c["BASE"] + c["SPEED_LOAD"] * cv.length()) / L.DT)), P["nmin"][2] + P["nmin"][3])
        return L.Curve(i, cv.pts, R0, P["Rk"], n)

    # -- phase helpers --------------------------------------------------------------------------------------------
    def _seg(self, i, p1, R1, n=None, phase=None, base=None):
        k = self.k
        p0, R0 = self.segs[k].at(i)
        self.segs[k] = L.Seg(i, p0, p1, R0, R1, n if n is not None else self._dur(np.linalg.norm(p1 - p0), R0, R1, base))
        if phase:
            self.phase, self.i_phase = phase, i

    def _curve(self, i, pts, R1, n_min=0, phase=None):
        k = self.k
        p0, R0 = self.segs[k].at(i)
        cv = L.Curve(i, [p0] + list(pts), R0, R1, 10)
        cv.n = max(self._dur(cv.length(), R0, R1), n_min)
        self.segs[k] = cv
        if phase:
            self.phase, self.i_phase = phase, i
        return cv

    def _start_push(self, i):
        Pp = self.push
        k = self.k
        p0, R0 = self.segs[k].at(i)
        self.pushes += 1
        self.ext = 0
        self.grip[k] = 0.0  # open: only the front finger touches the book
        self.events.append((i, "push", round(Pp["shift"], 4), round(Pp["over"], 4)))
        zc = Pp["z_cross"]
        if p0[0] < 0.56:  # in front of the board: come down in front of the book, then forward under the board
            ctrl = np.array([min(p0[0], 0.46), Pp["B"][1], zc - 0.01])
        else:  # behind the book: rise first
            ctrl = np.array([p0[0], Pp["B"][1], max(zc, p0[2]) + 0.01])
        cv = self._curve(i, [ctrl, Pp["B"]], Pp["Rc"], phase="p_go")
        self.events.append((i, "p_go clear", round(self._curve_clear(cv) * 1000, 1)))

    def _start_pinch(self, i, q_arms):
        self.plan = self._pinch_plan(q_arms[self.k])
        if self.plan is None:
            self.events.append((i, "no pinch plan"))
            return False
        P = self.plan
        k = self.k
        p0, R0 = self.segs[k].at(i)
        self.grip[k] = 0.0
        via = P["via"]
        sp_x = P["P"][0]
        if p0[0] > sp_x + 0.03:  # behind the book (after a push): up, then over the book to the insertion start
            zc = max(self.push["z_cross"] if self.push else via[2], via[2]) + 0.02
            for _ in range(5):  # raise the control point until the hand clears the book, counter and boards by 8 mm
                cv = L.Curve(i, [p0, np.array([p0[0] + 0.01, p0[1], zc]), via], R0, P["R"], 10)
                if self._curve_clear(cv) > 0.008:
                    break
                zc += 0.02
            pts = cv.pts[1:]
        else:
            pts = [np.array([via[0], via[1], max(p0[2], via[2] + 0.06)]), via]
        cv = self._curve(i, pts, P["R"], n_min=P["nmin"][0], phase="g_go")
        self.events.append((i, "pinch", P["daz"], int(round(np.degrees(P["psi"]))), P["flip"], round(P["margin"], 2),
                            round(self._curve_clear(cv) * 1000, 1)))
        return True

    def _replan(self, i, obs, G, q_arms):
        """perceive, then push or pinch; if the book is not found, move the hand out of the view and look again"""
        b = self._perceive(obs, G)
        if b is None:
            self.events.append((i, "no book"))
            if self.looks >= 3:
                return False
            self.looks += 1
            k = self.k
            p0, R0 = self.segs[k].at(i)
            view = np.array([0.42, 0.36 if k == 0 else -0.36, 1.15])
            self._seg(i, view, R0, phase="r_look")
            return True
        self.book = b
        self.push = self._push_plan(G) if self.pushes < 3 else None
        if self.push is not None:
            self._start_push(i)
            return True
        return self._start_pinch(i, q_arms)

    def _retreat(self, i, why):
        k = self.k
        p0, R0 = self.segs[k].at(i)
        a_ax = self.plan["a"] if self.plan else np.array([0, 0, -1.0])
        self.grip[k] = 0.0
        self.tries += 1
        self.events.append((i, why))
        self._seg(i, p0 - 0.05 * a_ax + np.array([0, 0, 0.08]), R0, n=16, phase="r_back")

    # -- policy ---------------------------------------------------------------------------------------------------
    def act(self, obs):
        c = self.c
        q = np.asarray(obs["proprioception"], np.float64)
        gq = np.asarray(obs["proprioception_grippers"], np.float64)
        q_arms = [q[0:7], q[9:16]]
        G = [fk(q_arms[0], 0), fk(q_arms[1], 1)]
        i = self.t
        if self.phase == "start":
            for kk in (0, 1):
                self.segs[kk] = L.hold(i, G[kk][0], G[kk][2])
            b = self._perceive(obs, G)
            self.k = 0 if (b is None or b["xy"][1] >= 0) else 1
            o = 1 - self.k  # the idle arm waits at mid height (wrist heights stay close: coordination)
            po = G[o][0]
            tgt = np.array([min(po[0], 0.50), po[1], c["Z_IDLE"]])
            self.segs[o] = L.Seg(i, po, tgt, G[o][2], G[o][2], self._dur(np.linalg.norm(tgt - po)) + 6)
            if not self._replan(i, obs, G, q_arms):
                self.phase = "idle"
        k = self.k
        ph = self.phase
        seg = self.segs[k]
        gp = G[k][0]
        done = i >= seg.end()
        Pp = self.push
        if ph == "p_go" and i >= seg.end() - 1 and (np.linalg.norm(gp - Pp["B"]) < 0.015 or i > seg.end() + 12):
            self._seg(i, Pp["S"], Pp["Rp"], n=self._dur(np.linalg.norm(Pp["S"] - Pp["B"]), Pp["Rc"], Pp["Rp"], base=0.15),
                      phase="p_down")
        elif ph == "p_down" and done and abs(gp[2] - Pp["S"][2]) < 0.004 and abs(gp[0] - Pp["S"][0]) < 0.006:
            self._seg(i, Pp["E"], Pp["Rp"], n=int(round((0.25 + 6.0 * np.linalg.norm(Pp["E"] - Pp["S"])) / L.DT)), phase="p_push")
            self.fb[k] = 0.0
        elif ph in ("p_go", "p_down") and i > seg.end() + 20:  # blocked: back up over the book and re-plan
            self.events.append((i, "push blocked " + ph))
            self._seg(i, Pp["B"] + np.array([0, 0, 0.03]), Pp["Rc"], n=12, phase="p_up")
        elif ph == "p_push" and done and (gp[0] < Pp["E"][0] + 0.003 or i > seg.end() + 8):
            moved = max(0.0, Pp["S"][0] - 0.008 - gp[0]) if self.ext == 0 else 0.0  # finger travel after contact
            prior = {"xy": self.book["xy"] - np.array([moved, 0.0]), "yaw": self.book["yaw"]}
            b = self._perceive(obs, G, prior=prior)
            if b is not None:
                self.book = b
            over = self._overhang(self.book)
            if over < c["D_MIN"] and self.ext < 2 and b is not None:  # extend the push from here
                self.ext += 1
                more = c["D_PUSH"] - over
                p0, R0 = seg.at(i)
                self.events.append((i, "push+", round(more, 4)))
                self._seg(i, p0 - np.array([more + 0.002, 0, 0]), R0, n=int(round((0.25 + 6.0 * more) / L.DT)), phase="p_push")
            else:
                self.unwind[k] = -self.fb[k]  # take back the push's integral offset of the commanded pose
                self.fb[k] = 0.0
                if not self._start_pinch(i, q_arms):  # e.g. the spine still overhangs too little: back up, push again
                    self._seg(i, Pp["B"] + np.array([0.01, 0, 0.02]), Pp["Rc"], n=12, phase="p_up")
        elif ph == "p_up" and done and gp[2] > seg.p1[2] - 0.01:
            if not self._replan(i, obs, G, q_arms):
                self.phase = "idle"
        elif ph == "g_go" and done and gq[k] > 0.9 and (np.linalg.norm(gp - self.plan["via"]) < 0.008 or i > seg.end() + 12):
            self._seg(i, self.plan["F"], self.plan["R"], n=max(8, self.plan["nmin"][1]), phase="g_insert")
        elif ph == "g_insert" and done and (np.linalg.norm(gp - self.plan["F"]) < 0.004 or i > seg.end() + 8):
            self.grip[k] = 1.0
            self.phase, self.i_phase = "g_close", i
        elif ph == "g_close" and i - self.i_phase >= c["T_CLOSE"]:
            if gq[k] < 0.12 and self.tries < c["MAX_TRIES"]:
                self._retreat(i, "missed")
            else:
                p0, R0 = seg.at(i)
                self.segs[k] = self._carry_curve(i, p0, R0)
                self.phase, self.i_phase = "c_go", i
        elif ph in ("c_go", "c_lower") and gq[k] < 0.08 and self.tries < c["MAX_TRIES"]:
            self._retreat(i, "dropped")
        elif ph == "c_go" and done and (np.linalg.norm(gp - seg.p1) < 0.012 or i > seg.end() + 15):
            p0, R0 = seg.at(i)
            far = self._far_edge_offset(np.radians(c["PSI_CARRY"] + c["DROOP"]))
            self._seg(i, np.r_[p0[:2], L.LOWER_TOP - c["Z_PRESS"] - far[1]], R0, n=10, phase="c_lower")
            self.zh = []
        elif ph == "c_lower":
            self.zh.append(gp[2])
            stalled = len(self.zh) > 5 and self.zh[-5] - gp[2] < 0.0015 and i - self.i_phase > 5
            if i >= seg.end() + 2 or stalled:
                self.grip[k] = 0.0
                self.phase, self.i_phase = "c_release", i
        elif ph == "c_release" and i - self.i_phase > 6:
            p0, R0 = seg.at(i)
            self._seg(i, p0 + np.array([-0.06, 0, 0.03]), R0, n=14, phase="done")
        elif ph in ("r_back", "r_look") and i >= seg.end() + 2:
            if not self._replan(i, obs, G, q_arms):
                self.phase = "idle"
        # lag guard: while carrying, the reference waits for the hand (no feed-forward this step; the segment is delayed)
        paused = False
        if self.phase in ("c_go",) and i < self.segs[k].end():
            p_ref, _ = self.segs[k].at(i)
            v = self.segs[k].at(i + 1)[0] - p_ref
            if np.linalg.norm(v) > 1e-5:
                lag = float((p_ref - self.fb[k] - gp) @ v) / np.linalg.norm(v)  # along the motion, sag correction removed
                if lag > c["LAG"]:
                    self.segs[k].i0 += 1
                    paused = True
        # commands
        a = np.zeros(14, np.float32)
        for kk in (0, 1):
            S = self.segs[kk]
            p0, R0 = S.at(i)
            p1, R1 = S.at(i + 1)
            if paused and kk == k:
                p1, R1 = p0, R0
            gpk = G[kk][0]
            if kk == k and self.phase == "p_push":  # integral correction: the book's friction deflects the arm
                d = np.clip(self.fb[kk] + c["KI_PUSH"] * (p0 - gpk), -c["FB_PUSH"], c["FB_PUSH"]) - self.fb[kk]
                self.fb[kk] += d
            elif kk == k and self.phase in ("c_go",):  # integral correction of the sag under the book's weight
                d = np.clip(self.fb[kk] + c["KI_LOAD"] * (p0 - gpk), -c["FB_LOAD"], c["FB_LOAD"]) - self.fb[kk]
                self.fb[kk] += d
            elif np.linalg.norm(p1 - p0) < 2e-4 and not (kk == k and self.phase in ("g_close", "c_lower", "c_release")):
                d = np.clip(self.fb[kk] + c["KP"] * (p0 - gpk), -c["FB_MAX"], c["FB_MAX"]) - self.fb[kk]
                self.fb[kk] += d
            else:
                d = np.zeros(3)
            uw = np.clip(self.unwind[kk], -0.01, 0.01)  # at most 1 cm per step
            self.unwind[kk] -= uw
            a[6 * kk:6 * kk + 3] = np.clip(p1 - p0 + d + uw, -c["VMAX"], c["VMAX"])
            a[6 * kk + 3:6 * kk + 6] = np.clip(L.rotvec(R1 @ R0.T), -0.15, 0.15)
            a[12 + kk] = self.grip[kk]
        if self.log:
            self.trace.append((i, self.phase, G[k][0].round(4).tolist(), gq.round(2).tolist()))
        self.t += 1
        return a
