"""StackSingleBookShelf hybrid controller: shared helpers (Oct 28). Robot frame of hybrid_s2b (fk, CAMS): world + (0.15, 0, 0.1).
Kinematics (fk with the full flange rotation), rotation helpers, minimum-jerk pose segments and the scene constants measured
once from the scene description during development (ana/scene_probe.py). Nothing here reads the simulator."""
import numpy as np

try:
    from .hybrid_s2b import fk, DH, BASE_Y, TOOL_YAW, _mdh
except ImportError:
    from hybrid_s2b import fk, DH, BASE_Y, TOOL_YAW, _mdh

# ---- scene (robot frame) ---------------------------------------------------------------------------------------------
Z_COUNTER = 0.7901  # counter top
X_EDGE = 0.4959  # counter front edge
BOOK_X = (-0.0467, 0.0472)  # book body frame: spine (near the robot for |yaw| <= 30 deg) .. fore-edge
BOOK_Y = (-0.0775, 0.0605)  # main block ends (the ribbon sticks out of the +y end, near the spine, to 0.0866)
BOOK_T = 0.0265  # thickness; the body origin is 0.8 mm below the top face
BOOK_TOP_OFF = 0.0008
SHELF_X = (0.6656, 0.8717)  # lower/upper shelf boards (x extent)
SHELF_Y = (-0.4412, 0.5723)
LOWER_TOP = 1.1259  # lower shelf board top (bottom 1.1087)
UPPER_BOT = 1.3549  # upper shelf board bottom
# ---- gripper ---------------------------------------------------------------------------------------------------------
PAD_C = 0.1034  # flange (fk position) -> finger pad centre along the approach axis
PAD_L = 0.012  # pad half length along the approach axis
TIP = PAD_C + PAD_L  # flange -> fingertip
DT = 0.05


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    a = np.arccos(c)
    if a < 1e-9:
        return np.zeros(3)
    if np.pi - a < 1e-4:
        w, V = np.linalg.eigh((R + R.T) / 2)
        return V[:, -1] * a
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (a / (2 * np.sin(a)))


def expm(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Seg:
    """minimum-jerk segment: flange position p0 -> p1 and orientation R0 -> R1 (geodesic) over n steps from step i0"""

    def __init__(self, i0, p0, p1, R0, R1, n):
        self.i0, self.p0, self.p1, self.R0, self.n = i0, np.array(p0, float), np.array(p1, float), np.array(R0, float), max(1, int(n))
        self.R1 = np.array(R1, float)
        self.rv = rotvec(self.R1 @ self.R0.T)

    def at(self, i):
        s = minjerk((i - self.i0) / self.n)
        return self.p0 + s * (self.p1 - self.p0), expm(s * self.rv) @ self.R0

    def end(self):
        return self.i0 + self.n


def hold(i, p, R):
    return Seg(i, p, p, R, R, 1)


def book_corners(xy, yaw):
    """main-block corners (robot frame xy): (-x,-y), (+x,-y), (+x,+y), (-x,+y) of the book body frame"""
    c, s = np.cos(yaw), np.sin(yaw)
    R2 = np.array([[c, -s], [s, c]])
    P = [(BOOK_X[0], BOOK_Y[0]), (BOOK_X[1], BOOK_Y[0]), (BOOK_X[1], BOOK_Y[1]), (BOOK_X[0], BOOK_Y[1])]
    return np.array([np.asarray(xy) + R2 @ np.array(p) for p in P])


def pinch_frame(psi, u, flip=False):
    """gripper rotation (columns x, y = closing axis, z = approach) for a tilted pinch: closing axis c at psi from vertical in
    the vertical plane of the horizontal direction u (into the book); approach axis a along u and down"""
    u3 = np.array([u[0], u[1], 0.0]) / np.linalg.norm(u[:2])
    a = np.cos(psi) * u3 + np.array([0, 0, -np.sin(psi)])
    c = np.sin(psi) * u3 + np.array([0, 0, np.cos(psi)])
    y = -c if flip else c
    x = np.cross(y, a)
    return np.stack([x, y, a], 1)


class Curve:
    """minimum-jerk motion along a Bezier curve through control points pts (flange position) with the orientation
    turning R0 -> R1 (geodesic) on the same time scale, n steps from step i0"""

    def __init__(self, i0, pts, R0, R1, n):
        self.i0, self.n = i0, max(1, int(n))
        self.pts = [np.array(p, float) for p in pts]
        self.p0, self.p1 = self.pts[0], self.pts[-1]
        self.R0, self.R1 = np.array(R0, float), np.array(R1, float)
        self.rv = rotvec(self.R1 @ self.R0.T)

    def point(self, s):
        P = list(self.pts)
        while len(P) > 1:  # de Casteljau
            P = [(1 - s) * P[j] + s * P[j + 1] for j in range(len(P) - 1)]
        return P[0]

    def at(self, i):
        s = minjerk((i - self.i0) / self.n)
        return self.point(s), expm(s * self.rv) @ self.R0

    def end(self):
        return self.i0 + self.n

    def length(self, m=20):
        q = [self.point(s) for s in np.linspace(0, 1, m + 1)]
        return float(sum(np.linalg.norm(q[j + 1] - q[j]) for j in range(m)))
