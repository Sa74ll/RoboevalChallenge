"""Hybrid PickSingleBookFromTable controller (Oct 28, v3): classical RGB-D perception of the book + scripted control. The
book can only be pinched across its thickness (the 8.2 cm gripper cannot span its 9.4 cm width, and the simulator's
effective pad friction, 3/sqrt(impratio 10) = 0.95, is too low for a corner pinch across the side faces), so one finger
has to get under it: where it overhangs the counter's front edge. The controller pushes the book toward the robot until
an edge or a corner overhangs enough (if needed), then pinches the overhang with a tilted hand (lower pad under the
near-bottom edge, upper pad on the top face) and lifts. Only the standard observations are used (nothing is read from
the simulator at test time).

Frames: everything is in the robot frame of fk and CAMS (arm bases at (0, +-0.3, 0.6)); the world frame is that frame
shifted by (-0.15, 0, -0.1) (robot at world (-0.15, 0, -0.1)), which never matters because actions are deltas.
"""
import numpy as np

try:
    from .hybrid_s2b import fk
    from . import pk_perception as PP
    from . import pk_ik as IK
except ImportError:
    from hybrid_s2b import fk
    import pk_perception as PP
    import pk_ik as IK

# ---- geometry (robot frame) -----------------------------------------------------------------------------------------
Z_COUNTER = 0.7901  # counter top (world 0.6901)
X_EDGE = 0.4959  # counter front edge (world 0.3459); below the top slab (1.7 cm) the frame is recessed
BOOK_X = (-0.0467, 0.0472)  # main block in the book body frame: spine .. fore-edge
BOOK_Y = (-0.0777, 0.0605)  # bottom end .. top end (a ribbon sticks out beyond the top end to 0.0866)
BOOK_T = 0.0267  # thickness (hull)
COM_B = np.array([-0.0043, -0.0113])  # centre of mass in the body frame (xy)
PAD_C = 0.1034  # flange (fk position) -> finger pad centre along the hand axis
PAD_L = 0.012  # pad half length along the hand axis
TIP = 0.1154  # flange -> fingertip (pad bottom) along the hand axis: the lowest point of the gripper
FW = 0.0105  # finger half width across the closing axis
DT = 0.05
KV_KP = 0.09  # Panda joint servos: ctrl - q ~ (kv/kp) qdot (PackBox agent's measurement)
ROT_RATE = 0.11  # rad per step: cap on the reference's peak rotation rate (joint targets move <= 0.109 rad/step)
PARK = (np.array([0.35, 0.45, 1.00]), np.array([0.35, -0.45, 1.00]))  # where a finished pusher waits: out of the
# head camera's view of the book and of the other arm's way


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


_rotvec, _expm = IK._rotvec, IK._expm


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rot2(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def book_corners(book):
    """main-block corners (robot frame xy), counterclockwise in the body frame: (-x,-y), (+x,-y), (+x,+y), (-x,+y)"""
    c, s = np.cos(book["yaw"]), np.sin(book["yaw"])
    R2 = np.array([[c, -s], [s, c]])
    P = [(BOOK_X[0], BOOK_Y[0]), (BOOK_X[1], BOOK_Y[0]), (BOOK_X[1], BOOK_Y[1]), (BOOK_X[0], BOOK_Y[1])]
    return [np.asarray(book["xy"]) + R2 @ np.array(p) for p in P]


def book_com(book):
    c, s = np.cos(book["yaw"]), np.sin(book["yaw"])
    return np.asarray(book["xy"]) + np.array([[c, -s], [s, c]]) @ COM_B


def shifted(book, dx):
    return {"xy": np.asarray(book["xy"]) + np.array([dx, 0.0]), "yaw": book["yaw"]}


def pinch_frame(psi, u, flip):
    """hand rotation for a tilted pinch: closing axis c (lower -> upper pad) at psi from vertical in the vertical plane of
    the horizontal direction u; approach axis a along u and down (psi below horizontal); flip: which finger is up"""
    u3 = np.array([u[0], u[1], 0.0]) / np.linalg.norm(u)
    a = np.cos(psi) * u3 + np.array([0, 0, -np.sin(psi)])
    c = np.sin(psi) * u3 + np.array([0, 0, np.cos(psi)])
    y = -c if flip else c
    x = np.cross(y, a)
    return np.stack([x, y, a], 1), a, c


def pinch_geometry(P, u, psi, delta=0.0):
    """closing centre, approach/closing axes and pad gap for a pinch whose lower pad catches the near-bottom edge point P
    (xy) delta from its centre toward the fingertip, while the upper pad's tip edge rests on the top face"""
    u3 = np.array([u[0], u[1], 0.0])
    a = np.cos(psi) * u3 + np.array([0, 0, -np.sin(psi)])
    c = np.sin(psi) * u3 + np.array([0, 0, np.cos(psi)])
    g = (BOOK_T + (PAD_L - delta) * np.sin(psi)) / np.cos(psi)
    O = np.array([P[0], P[1], Z_COUNTER]) + 0.5 * g * c - delta * a
    return O, a, c, g


def tip_clear(P, u, delta=0.0, psi=np.radians(40.0)):
    """how far the lower finger's tip region stays in front of the counter's front face (m, positive = clear): the tip is
    (PAD_L - delta) along the approach axis beyond P ((PAD_L - delta) cos psi along u, (PAD_L - delta) sin psi below the
    counter top); the pad's corners reach 1.05 cm across u"""
    return X_EDGE - (P[0] + (PAD_L - delta) * np.cos(psi) * u[0] + FW * abs(u[1]))


def pinch_candidates(book, margin=0.005, delta=0.0, psi=np.radians(40.0)):
    """pinch candidates: on the main block's edges that face the robot (u = inward normal), and at its corners (u = the
    corner's bisector into the book). Each has the lower-pad point P, direction u, kind, and clearance/usable length."""
    C = book_corners(book)
    com = book_com(book)
    out = []
    for i in range(4):
        A, B = C[i], C[(i + 1) % 4]
        e = B - A
        L = float(np.linalg.norm(e))
        t = e / L
        u = -np.array([t[1], -t[0]])
        if u[0] < 0.5:  # must face the robot (azimuth of u within 60 deg of +x)
            continue
        s_all = np.linspace(0.016, L - 0.016, 121)  # pad (+-1.05 cm) on the edge, 5 mm off the corners
        Pts = A + s_all[:, None] * t
        clr = np.array([tip_clear(p, u, delta, psi) for p in Pts])
        ok = clr >= margin
        if not ok.any():
            continue
        s_ok = s_all[ok]
        s_com = float(np.clip((com - A) @ t, 0.0, L))
        j = int(np.argmin(np.abs(s_ok - s_com)))
        s_pick = float(s_ok[j])
        out.append(dict(kind="edge", P=A + s_pick * t, u=u, clear=float(clr[ok][j]), usable=float(s_ok.ptp() + 0.032),
                        t=t))
    for i in range(4):
        Cc = C[i]
        e1 = C[(i + 1) % 4] - Cc
        e2 = C[(i - 1) % 4] - Cc
        b = e1 / np.linalg.norm(e1) + e2 / np.linalg.norm(e2)
        u = b / np.linalg.norm(b)
        if u[0] < 0.5:
            continue
        P = Cc + 0.003 * u  # just inside the corner: the lower pad catches both bottom edges near the apex
        clr = tip_clear(P, u, delta, psi)
        if clr >= margin:
            out.append(dict(kind="apex", P=P, u=u, clear=float(clr), usable=0.0, t=None))
    return out


class _Seg:
    """minimum-jerk segment: flange position p0 -> p1 and orientation R0 -> R1 (geodesic) over n steps from step i0"""

    def __init__(self, i0, p0, p1, R0, R1, n):
        self.i0, self.p0, self.p1, self.R0, self.n = i0, np.array(p0, float), np.array(p1, float), R0, max(1, n)
        self.rv = _rotvec(R1 @ R0.T)

    def at(self, i):
        s = minjerk((i - self.i0) / self.n)
        return self.p0 + s * (self.p1 - self.p0), _expm(s * self.rv) @ self.R0

    def end(self):
        return self.i0 + self.n


def _hold(i, p, R):
    return _Seg(i, p, p, R, R, 1)


class HybridPK:
    carry_queue = False  # the bundle's policy.py resets every episode

    def __init__(self, log=False, psi_deg=40.0, back=0.04, lift=0.20, pull=0.04, t_reach=(0.6, 2.2), t_in=0.5, t_close=8,
                 t_lift=1.2, vmax=0.03, kp=0.3, fb_max=0.015, clear_min=0.008, usable_min=0.024, ik_margin=0.55,
                 com_keep=0.025, push_h=0.008, push_margin=0.004, push_extra=0.007, max_pushes=2, max_retries=2, push_tilt_deg=30.0,
                 push_k=16.0, push_fb=0.0, push_lag=0.008, reach_tol=0.025, reach_rot=15.0, beta=0.3, idle_drop=0.0, delta=0.0, psi2_deg=46.0, arm_clear=0.12, ik_relaxed=0.35, face_nx=0.6):
        self.log = log
        self.psi, self.back, self.lift, self.pull = np.radians(psi_deg), back, lift, pull
        self.t_reach, self.t_in, self.t_close, self.t_lift = t_reach, t_in, t_close, t_lift
        self.vmax, self.kp, self.fb_max = vmax, kp, fb_max
        self.clear_min, self.usable_min, self.ik_margin = clear_min, usable_min, ik_margin
        self.com_keep, self.push_h, self.push_margin, self.push_extra = com_keep, push_h, push_margin, push_extra
        self.max_pushes, self.max_retries = max_pushes, max_retries
        self.push_tilt = np.radians(push_tilt_deg)
        self.push_k, self.push_fb, self.push_lag = push_k, push_fb, push_lag
        self.reach_tol, self.reach_rot = reach_tol, reach_rot
        self.beta, self.idle_drop, self.delta = beta, idle_drop, delta
        self.psi2 = np.radians(psi2_deg)
        self.arm_clear, self.ik_relaxed, self.face_nx = arm_clear, ik_relaxed, face_nx
        self.book_source = None  # analysis only: a test harness may supply ground-truth book poses (callable)
        self.reset()

    def reset(self, episode=None):
        self.t = 0
        self.book, self.plan, self.push = None, None, None
        self.phase, self.i_phase = "start", 0
        self.segs = [None, None]
        self.grip = [0.0, 0.0]
        self.k = None  # pinching arm
        self.fb = np.zeros((2, 3))
        self.pushes, self.retries = 0, 0
        self.trace = []
        self.est = None
        self.events = []
        self._seg_ids = [None, None]
        self.bad = []  # (arm, kind, u_x) options that did not reach their pre-pose
        self.went_home = False

    def _from(self, k, i, G):
        """start pose for a new segment of arm k: the current command, or the measured pose before the first segment"""
        if self.segs[k] is None:
            return G[k][0].copy(), G[k][2].copy()
        return self._cmd(k, i)

    def _idle_seg(self, i, k, G):
        """the arm that is not working lowers its hand by idle_drop on its own side (closer to the working hand's height:
        the bimanual vertical-difference metric) and waits there"""
        p0, R0 = G[k][0].copy(), G[k][2].copy()
        return _Seg(i, p0, p0 - np.array([0.0, 0.0, self.idle_drop]), R0, R0, 40)

    def _cmd(self, k, i):
        """current reference pose of arm k: new segments start here (never at the measured pose: the command leads the
        measured hand by the servo lag, and starting from the measured pose kept pushing the book ~3 cm after a push)"""
        return self.segs[k].at(i)

    # -- perception -----------------------------------------------------------------------------------------------
    def _perceive(self, obs, G):
        if self.book_source is not None:
            return self.book_source()
        excl = []
        for k in (0, 1):
            p, _, R = G[k]
            excl += [(p + 0.04 * R[:, 2], 0.11), (p + 0.10 * R[:, 2], 0.06)]
        r = PP.locate_book(obs, exclude=excl)
        self.est = r
        if not r.get("ok"):
            return None
        return {"xy": np.asarray(r["xy"], float), "yaw": float(r["yaw"])}

    # -- planning -------------------------------------------------------------------------------------------------
    def _options(self, book, extra=0.0):
        """pinch options (no IK yet), cheapest first"""
        opts = []
        for C in pinch_candidates(book, self.clear_min + extra, self.delta, self.psi):
            if C["kind"] == "edge" and C["usable"] < self.usable_min:
                continue
            dazs = (0.0, -0.25, 0.25, -0.45, 0.45, -0.6, 0.6) if C["kind"] == "edge" else (0.0, -0.15, 0.15, -0.3, 0.3)
            for daz in dazs:
                u = _rot2(C["u"], daz)
                if u[0] < 0.4:
                    continue
                clr = tip_clear(C["P"], u, self.delta, self.psi)
                if clr < self.clear_min + extra:
                    continue
                cost = 0.5 * abs(daz) - 3.0 * min(clr, 0.015) - 0.5 * min(C["usable"], 0.06) + (0.03 if C["kind"] == "apex" else 0.0)
                opts.append((cost, C, u, daz, clr))
        opts.sort(key=lambda o: o[0])
        return opts

    def _pinch_plan(self, G, q_arms, book=None, max_checks=24, arms=(0, 1)):
        """best reachable pinch for the book estimate, or None (tilt psi first, then the alternative tilt psi2)"""
        psi0 = self.psi
        try:
            for j, psi in enumerate((psi0, self.psi2)):
                self.psi = psi
                r = self._pinch_plan1(G, q_arms, book, max_checks, arms)
                if r is not None:
                    r["psi"] = psi
                    return r
            return None
        finally:
            self.psi = psi0

    def _pinch_plan1(self, G, q_arms, book=None, max_checks=24, arms=(0, 1)):
        book = book if book is not None else self.book
        best = None
        n_checks = 0
        for cost0, C, u, daz, clr in self._options(book):
            if best is not None and cost0 > best[0]:
                break
            O, a, c, g = pinch_geometry(C["P"], u, self.psi, self.delta)
            flange = O - PAD_C * a
            pre = flange - self.back * a
            for k in arms:
                if (k, C["kind"], round(float(u[0]), 2)) in self.bad:
                    continue
                for flip in (0, 1):
                    R, _, _ = pinch_frame(self.psi, u, flip)
                    rot = float(np.linalg.norm(_rotvec(R @ G[k][2].T)))
                    if rot > 2.5:
                        continue
                    cost = cost0 + 0.1 * rot + (0.3 if (C["P"][1] > 0) != (k == 0) and abs(C["P"][1]) > 0.05 else 0.0)
                    if best is not None and cost >= best[0]:
                        continue
                    n_checks += 1
                    qf, ep, er, mg = IK.path_check(q_arms[k], k, [(pre, R), (flange, R)], n_sub=5)
                    if ep > 0.002 or er > 0.02 or mg < self.ik_margin:
                        continue
                    # arm-arm clearance at the grasp pose (joint origins and link mid points of both arms)
                    dmin = np.min(np.linalg.norm(IK.link_points(qf, k)[:, None, :] - IK.link_points(q_arms[1 - k], 1 - k)[None, :, :], axis=2))
                    if dmin < self.arm_clear:
                        continue
                    best = (cost, dict(k=k, C=C, kind=C["kind"], u=u, O=O, a=a, c=c, g=g, R=R, flange=flange, pre=pre,
                                       flip=flip, margin=mg, daz=daz, clear=clr))
            if n_checks >= max_checks:
                break
        return best[1] if best else None

    def _push_plan(self, G, q_arms):
        """push the book so that it overhangs the counter's front edge enough for a pinch, keeping its centre of mass
        com_keep behind the edge. The pusher stands behind one of the two faces at the book's far corner, closing axis
        parallel to that face (fingers closed: both on the face, so the book translates without turning), tilted
        push_tilt from vertical with its fingertips pointing away from the book (the wrist stays clear of the bookshelf's
        lower shelf board above the counter), and pushes along the face's inward normal. The contact is the book's far top
        edge against the fingers' back face (the tilted fingers lean over the book)."""
        book = self.book
        com = book_com(book)
        C = np.array(book_corners(book))
        j = int(np.argmax(C[:, 0]))
        th = self.push_tilt
        best = None
        for e0, e1 in ((C[j], C[(j + 1) % 4]), (C[(j - 1) % 4], C[j])):
            t_f = (e1 - e0) / np.linalg.norm(e1 - e0)
            n_f = np.array([t_f[1], -t_f[0]])  # outward normal (counterclockwise corners)
            if n_f[0] < self.face_nx:  # within ~53 deg of +x (60 deg in the relaxed fallback)
                continue
            d_dir = -n_f
            L = float(np.linalg.norm(e1 - e0))
            s_c = float(np.clip((com - e0) @ t_f, 0.03, L - 0.03))  # closed fingers span +-2.6 cm
            contact = e0 + s_c * t_f
            d_max = (com[0] - (X_EDGE + self.com_keep)) / -d_dir[0]
            if d_max < 0.004:
                continue
            shift = None
            for d in np.arange(0.004, d_max + 1e-9, 0.003):
                if self._options({"xy": np.asarray(book["xy"]) + d * d_dir, "yaw": book["yaw"]}, extra=self.push_extra):
                    shift = d
                    break
            if shift is None:
                shift = d_max
            shift = float(min(shift + self.push_margin, d_max))
            if self.pushes > 0:  # a repeat push: the first one fell short; push at least 1.2 cm
                shift = float(min(max(shift, 0.012), d_max))
            # the predicted pose must also have a reachable pinch (IK check); otherwise push further
            while shift < d_max and self._pinch_plan(G, q_arms, book={"xy": np.asarray(book["xy"]) + shift * d_dir,
                                                                      "yaw": book["yaw"]}, max_checks=12) is None:
                shift = float(min(shift + 0.005, d_max))
            n3 = np.array([n_f[0], n_f[1], 0.0])
            a_p = np.sin(th) * n3 + np.array([0.0, 0.0, -np.cos(th)])
            for k in (0, 1):
                gR = G[k][2]
                if abs(com[1]) > 0.04 and (com[1] > 0) != (k == 0):  # the arm on the book's side pushes (a cross-arm
                    continue  # push tipped books and later collided with the other arm)
                for ysgn in (1.0, -1.0):
                    yv = ysgn * np.array([t_f[0], t_f[1], 0.0])
                    hx = np.cross(yv, a_p)
                    if hx @ n3 > 0:  # the finger face that pushes must face the book (-n_f)
                        continue
                    Rp = np.stack([hx, yv, a_p], 1)
                    # fingertips push_h above the counter; the book's far top edge meets the fingers' back face z_c up
                    f_z = Z_COUNTER + self.push_h - (TIP * a_p[2] + FW * hx[2])
                    z_c = (f_z + FW * hx[2] - (Z_COUNTER + BOOK_T)) / -a_p[2]
                    off = z_c * a_p + FW * hx  # flange -> contact point
                    top = np.array([contact[0], contact[1], Z_COUNTER + BOOK_T])
                    start = top + 0.016 * n3 - off
                    above = start + np.array([0, 0, 0.07]) - 0.01 * n3
                    stop = top + shift * np.r_[d_dir, 0.0] - off
                    ang = float(np.linalg.norm(_rotvec(Rp @ gR.T)))
                    cost = shift * 10 + 0.3 * ang + (0.3 if (com[1] > 0) != (k == 0) else 0.0)
                    if best is not None and cost >= best[0]:
                        continue
                    _, ep, er, mg = IK.path_check(q_arms[k], k, [(above, Rp), (start, Rp), (stop, Rp)], n_sub=4)
                    if ep > 0.002 or er > 0.02 or mg < self.ik_margin:
                        continue
                    best = (cost - 0.3 * mg, dict(k=k, shift=shift, R=Rp, above=above, start=start, end=stop, margin=mg,
                                                  mode="face", grip=1.0, d_dir=d_dir,
                                                  pred={"xy": np.asarray(book["xy"]) + shift * d_dir, "yaw": book["yaw"]}))
        return best[1] if best else None

    def _start_pinch(self, i, G):
        P = self.plan
        k = P["k"]
        p0, R0 = self._from(k, i, G)
        # via a point 7 cm above and 3 cm in front of the pre-grasp pose: a straight reach from above the counter swept the
        # fingers onto the book's top face (and tipped an overhanging book off the edge)
        via = P["pre"] + np.array([0.0, 0.0, 0.07]) - 0.03 * np.r_[P["u"], 0.0]
        D = float(np.linalg.norm(via - p0))
        rot = float(np.linalg.norm(_rotvec(P["R"] @ R0.T)))
        n = max(int(round((self.t_reach[0] + self.t_reach[1] * D) / DT)), int(np.ceil(1.9 * rot / ROT_RATE)))
        self.k = k
        self.segs[k] = _Seg(i, p0, via, R0, P["R"], n)
        P["via_end"] = i + n
        o = 1 - k
        if self.segs[o] is None:
            self.segs[o] = self._idle_seg(i, o, G)
        self.grip[k] = 0.0
        self.phase, self.i_phase = "g_reach", i

    def _replan(self, i, obs, G, q_arms):
        self.book = self._perceive(obs, G)
        if self.book is None:
            self.events.append((i, "no book"))
            return False
        self.plan = self._pinch_plan(G, q_arms)
        overhang = X_EDGE - float(np.min(np.array(book_corners(self.book))[:, 0]))
        relaxed_tried = self.plan is None and (self.pushes > 0 or overhang >= 0.02)
        if relaxed_tried:
            # after a push, or when the book already overhangs 2 cm (pushing it further risks tipping it over the edge):
            # accept a tighter pinch rather than push
            saved = (self.clear_min, self.ik_margin, self.usable_min)
            self.clear_min, self.ik_margin, self.usable_min = 0.005, self.ik_relaxed, 0.020
            try:
                self.plan = self._pinch_plan(G, q_arms)
            finally:
                self.clear_min, self.ik_margin, self.usable_min = saved
        if self.plan is None and self.pushes > 0 and not self.went_home:
            # nothing reachable from the current arm poses (a pusher parked in its tilted push orientation sits on a poor
            # IK branch): if a pinch is reachable from an arm's start configuration, send that arm home first
            ph_plan = self._pinch_plan(G, self.q_home, arms=(self.push["k"],)) if self.push is not None else None
            if ph_plan is not None:
                k = ph_plan["k"]
                p0, R0 = self._from(k, i, G)
                hp, hR = self.home[k]
                n = max(12, int(np.ceil(1.9 * float(np.linalg.norm(_rotvec(hR @ R0.T))) / ROT_RATE)),
                        int(round((0.4 + 2.0 * float(np.linalg.norm(hp - p0))) / DT)))
                self.segs[k] = _Seg(i, p0, hp, R0, hR, n)
                self.went_home = True
                self.home_k, self.home_end = k, i + n
                self.events.append((i, "home", k))
                self.phase, self.i_phase = "r_home", i
                return True
        if self.plan is not None:
            self.events.append((i, "pinch", self.plan["kind"], round(self.plan["clear"], 4), self.plan["k"]))
            self._start_pinch(i, G)
            return True
        self.push = self._push_plan(G, q_arms) if self.pushes < self.max_pushes else None
        if self.push is None and self.pushes < self.max_pushes:  # no push passes the normal margin: relaxed margin
            saved = (self.ik_margin, self.face_nx)
            self.ik_margin, self.face_nx = self.ik_relaxed, 0.5
            try:
                self.push = self._push_plan(G, q_arms)
            finally:
                self.ik_margin, self.face_nx = saved
        if self.push is None and self.plan is None and not relaxed_tried:  # last resort before giving up
            saved = (self.clear_min, self.ik_margin, self.usable_min)
            self.clear_min, self.ik_margin, self.usable_min = 0.005, self.ik_relaxed, 0.020
            try:
                self.plan = self._pinch_plan(G, q_arms)
            finally:
                self.clear_min, self.ik_margin, self.usable_min = saved
            if self.plan is not None:
                self.events.append((i, "pinch", self.plan["kind"], round(self.plan["clear"], 4), self.plan["k"]))
                self._start_pinch(i, G)
                return True
        if self.push is not None:
            self.events.append((i, "push", self.push["mode"], round(self.push["shift"], 4), self.push["k"]))
            k = self.push["k"]
            p0, R0 = self._from(k, i, G)
            # rotate first at a via point in front of the bookshelf (the palm is 20 cm wide along the closing axis: turning
            # it near x 0.6 swept it into the lower shelf board), then translate above the push start, then descend
            ab = self.push["above"]
            via = np.array([min(p0[0], 0.50), 0.5 * (p0[1] + ab[1]), max(min(p0[2], 1.15), ab[2] + 0.06)])
            rot = float(np.linalg.norm(_rotvec(self.push["R"] @ R0.T)))
            n = max(int(round((0.4 + 2.4 * np.linalg.norm(via - p0)) / DT)), int(np.ceil(1.9 * rot / ROT_RATE)))
            self.segs[k] = _Seg(i, p0, via, R0, self.push["R"], n)
            self.push["via_end"] = i + n
            if self.segs[1 - k] is None:
                self.segs[1 - k] = self._idle_seg(i, 1 - k, G)
            self.grip[k] = self.push["grip"]
            self.pushes += 1
            self.phase, self.i_phase = "p_reach", i
            return True
        self.events.append((i, "no plan"))
        return False

    def _retreat(self, i, G, why):
        """open, back off along the approach axis and up, then re-perceive and re-plan"""
        k = self.k
        p, R = self._cmd(k, i)
        # back off along the approach axis, then up and toward the arm's own side, out of the head camera's view of the
        # book's near edge (a hand in front of the book hides it and biases the fit toward the camera)
        side = np.array([0.0, 0.10 if k == 0 else -0.10, 0.0])
        up = p - 0.05 * self.plan["a"] + np.array([0, 0, 0.16]) + side
        self.segs[k] = _Seg(i, p, up, R, R, 16)
        self.grip[k] = 0.0
        self.retries += 1
        self.events.append((i, why))
        self.phase, self.i_phase = "r_back", i

    # -- control --------------------------------------------------------------------------------------------------
    def act(self, obs):
        q = np.asarray(obs["proprioception"], np.float64)
        gq = np.asarray(obs["proprioception_grippers"], np.float64)
        q_arms = [q[0:7], q[9:16]]
        G = [fk(q_arms[0], 0), fk(q_arms[1], 1)]
        a = np.zeros(14, np.float32)
        i = self.t
        if self.phase == "start":
            self.home = [(G[0][0].copy(), G[0][2].copy()), (G[1][0].copy(), G[1][2].copy())]
            self.q_home = [q_arms[0].copy(), q_arms[1].copy()]
            if not self._replan(i, obs, G, q_arms):
                self.phase = "idle"
                for k in (0, 1):
                    self.segs[k] = _hold(i, G[k][0], G[k][2])
        ph = self.phase
        pushing = False
        if ph.startswith("p_"):
            k = self.push["k"]
            seg = self.segs[k]
            gp = G[k][0]
            if ph == "p_reach" and "via_end" in self.push and i >= self.push["via_end"]:
                p, R = self._cmd(k, i)
                del self.push["via_end"]
                n = int(round((0.3 + 2.4 * np.linalg.norm(self.push["above"] - p)) / DT))
                self.segs[k] = _Seg(i, p, self.push["above"], R, self.push["R"], n)
            elif ph == "p_reach" and "via_end" not in self.push and i >= seg.end() and (
                    np.linalg.norm(gp - self.push["above"]) < 0.008 or i - seg.end() > 15):
                self.segs[k] = _Seg(i, seg.at(i)[0], self.push["start"], seg.at(i)[1], self.push["R"], 10)
                self.phase, self.i_phase = "p_down", i
            elif ph == "p_down" and i >= seg.end() and (abs(gp[2] - self.push["start"][2]) < 0.003 or i - seg.end() > 10):
                p, R = self._cmd(k, i)
                end = self.push["end"] + self.push_lag * np.r_[self.push["d_dir"], 0.0]
                n = int(round((0.3 + self.push_k * np.linalg.norm(end - p)) / DT))
                self.segs[k] = _Seg(i, p, end, R, R, n)
                self.phase, self.i_phase = "p_push", i
            elif ph == "p_push" and i >= seg.end() and ((gp - self.push["end"])[:2] @ self.push["d_dir"] > -0.002 or i - seg.end() > 12):
                p, R = self._cmd(k, i)
                back = gp - 0.012 * np.r_[self.push["d_dir"], 0.0]  # off the book first, then up
                self.segs[k] = _Seg(i, p, back, R, R, 4)
                self.grip[k] = 0.0
                self.phase, self.i_phase = "p_off", i
            elif ph == "p_off" and i >= seg.end():
                p, R = self._cmd(k, i)
                self.segs[k] = _Seg(i, p, p + np.array([0.0, 0, 0.08]), R, R, 9)
                self.phase, self.i_phase = "p_up", i
            elif ph == "p_up" and i >= seg.end():
                p, R = self._cmd(k, i)
                self.segs[k] = _Seg(i, p, PARK[k], R, R, 11)
                self.phase, self.i_phase = "p_clear", i
            elif ph == "p_clear" and i >= seg.end() - 5:
                if not self._replan(i, obs, G, q_arms):
                    self.phase = "idle"
            pushing = self.phase == "p_push"
        elif ph.startswith("g_"):
            k = self.k
            seg = self.segs[k]
            gp = G[k][0]
            P = self.plan
            if ph == "g_reach" and "via_end" in P and i >= P["via_end"]:
                p, R = self._cmd(k, i)
                del P["via_end"]
                self.segs[k] = _Seg(i, p, P["pre"], R, P["R"], 9)
                seg = self.segs[k]
            rot_err = float(np.linalg.norm(_rotvec(G[k][2] @ P["R"].T))) if ph == "g_reach" else 0.0
            if ph == "g_reach" and "via_end" not in P and i >= seg.end() and gq[k] > 0.9 and (
                    (np.linalg.norm(gp - P["pre"]) < 0.006 and rot_err < np.radians(4)) or i - seg.end() > 12):
                self.events.append((i, "reach_err", round(float(np.linalg.norm(gp - P["pre"])), 4), round(float(np.degrees(rot_err)), 1)))
                if (np.linalg.norm(gp - P["pre"]) > self.reach_tol or rot_err > np.radians(self.reach_rot)) and self.retries < self.max_retries:
                    self.bad.append((k, P["kind"], round(float(P["u"][0]), 2)))  # unreachable in practice: re-plan
                    self._retreat(i, G, "unreached")
                else:
                    p, R = self._cmd(k, i)
                    self.segs[k] = _Seg(i, p, P["flange"], R, R, int(round(self.t_in / DT)))
                    self.phase, self.i_phase = "g_insert", i
            elif ph == "g_insert" and i >= seg.end() and (np.linalg.norm(gp - P["flange"]) < 0.003 or i - seg.end() > 12):
                self.grip[k] = 1.0
                self.phase, self.i_phase = "g_close", i
            elif ph == "g_close" and i - self.i_phase >= self.t_close:
                if gq[k] < 0.12 and self.retries < self.max_retries:  # closed on nothing
                    self._retreat(i, G, "missed")
                else:
                    p, R = self._cmd(k, i)
                    up = p + np.array([0, 0, self.lift]) - self.pull * np.r_[P["u"], 0.0]
                    self.segs[k] = _Seg(i, p, up, R, R, int(round(self.t_lift / DT)))
                    self.phase, self.i_phase = "g_lift", i
            elif ph in ("g_lift", "g_hold") and gq[k] < 0.08 and self.retries < self.max_retries:  # dropped
                self._retreat(i, G, "dropped")
            elif ph == "g_lift" and i >= seg.end():
                self.phase, self.i_phase = "g_hold", i
        elif ph == "r_home":
            if i >= self.home_end + 3:
                if not self._replan(i, obs, G, q_arms):
                    self.phase = "idle"
        elif ph == "r_back":
            k = self.k
            if i >= self.segs[k].end() + 2:
                self.k = None
                if not self._replan(i, obs, G, q_arms):
                    self.phase = "idle"
        # commands: feedforward increments of the reference, plus (in free motion) a correction toward the reference from
        # the estimated environment command, fk(q + KV_KP qdot). The environment adds each delta to the forward kinematics of
        # its joint targets, which move at most 0.109 rad per step: whatever that cap cuts off is lost, so fast wrist
        # rotations drift ~8 deg / several cm without the correction (it also removes the gravity sag). It is off in
        # contact phases (push, close, lift, hold), where it would wind up a force.
        qd = np.asarray(obs["proprioception"], np.float64)
        qd_arms = [qd[18:25], qd[27:34]]
        contact_ph = self.phase in ("p_push", "g_close", "g_lift", "g_hold")
        for kk in (0, 1):
            S = self.segs[kk]
            p0, R0 = S.at(i)
            p1, R1 = S.at(i + 1)
            dp = p1 - p0
            dr = _rotvec(R1 @ R0.T)
            active = (self.push is not None and self.phase.startswith("p_") and kk == self.push["k"]) or (
                self.k == kk and self.phase.startswith(("g_", "r_")))
            if not (contact_ph and active):
                cp, _, cR = fk(q_arms[kk] + KV_KP * qd_arms[kk], kk)
                dp = dp + np.clip(self.beta * (p0 - cp), -0.01, 0.01)
                dr = dr + np.clip(self.beta * _rotvec(R0 @ cR.T), -0.05, 0.05)
            a[6 * kk:6 * kk + 3] = np.clip(dp, -self.vmax, self.vmax)
            a[6 * kk + 3:6 * kk + 6] = np.clip(dr, -0.15, 0.15)
            a[12 + kk] = self.grip[kk]
        if self.log:
            kk = self.k if self.k is not None else (self.push["k"] if self.push else 0)
            self.trace.append((i, self.phase, G[kk][0].round(4).tolist(), gq.round(3).tolist()))
        self.t += 1
        return a
