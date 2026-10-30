"""Compose the explainer video of one controller (step 3 of 3: trace -> explain -> compose).

Every frame is drawn with matplotlib from trace.py's recording and explain.py's perception stages and piped to ffmpeg.
Chapters: a title card; 1 Perceive (the controller's own perception, stage by stage, on the observation it planned
from); 2 Plan; 3 Act (the episode at half speed with the controller's estimates, planned paths and phases drawn in);
the result. With --gif, the Act chapter is also written as a GIF at full speed (10 frames per second). With --stills,
the perception and plan slides are also saved as PNG files.

  python scripts/explainer/compose.py /dev/shm/ctrlvid/RotateValve_900000 --out assets/videos/RotateValve.mp4 \
      --gif assets/controllers/RotateValve.gif --ffmpeg /path/to/ffmpeg
Needs numpy, matplotlib and Pillow (no simulator).
"""
import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402
from PIL import Image  # noqa: E402

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "submission" / "policy"))
import hybrid_s2b as hs  # noqa: E402  (the controllers' camera constants and forward kinematics)

W, H, FPS = 1280, 720, 30
STEP_FRAMES = 3  # frames per 50 ms control step at 30 fps: half speed
BG, PANEL, FG, MUTED, LINE = "#0D1117", "#161B22", "#E6EDF3", "#8B949E", "#30363D"
C_ARM = ("#58A6FF", "#FFA657")  # left arm, right arm
C_FIT, C_PLAN, C_OK, C_ALT, C_GREY = "#3FE0A0", "#F2CC60", "#3FB950", "#D2A8FF", "#59616C"
ARM = ("Left arm", "Right arm")
PAD_C = 0.1034  # flange -> finger pad centre (Panda hand)
plt.rcParams.update({"font.family": "Noto Sans", "font.size": 12, "text.color": FG, "axes.edgecolor": LINE,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": PANEL,
                     "figure.facecolor": BG, "axes.titlecolor": FG, "font.weight": "normal"})
HALO = [pe.withStroke(linewidth=3, foreground=BG)]

# official run (results/official_results.json): successes out of 50 and task score
OFFICIAL = {"StackTwoBlocks": (50, 0.948), "RotateValve": (50, 1.000), "PackBox": (49, 0.954), "CubeHandover": (50, 1.000),
            "LiftTray": (50, 1.000), "PickSingleBookFromTable": (50, 0.987), "StackSingleBookShelf": (50, 1.000)}
GOAL = {"StackTwoBlocks": "Stack one block on the other.",
        "RotateValve": "Turn both valves; each arm turns one.",
        "PackBox": "Close both flaps of the box.",
        "CubeHandover": "Hand the red rod from one gripper to the other, clear of the table.",
        "LiftTray": "Lift the 34 kg tray with both arms.",
        "PickSingleBookFromTable": "Pick the book up from the counter.",
        "StackSingleBookShelf": "Put the book on the shelf and let go."}
SHORT = {"PickSingleBookFromTable": "PickSingleBookFromTable (PickBook)", "StackSingleBookShelf": "StackSingleBookShelf (ShelfBook)"}

# ---- phase categories (timeline colours) -----------------------------------------------------------------------------
CAT = {"free": ("free motion", "#3B6EA8"), "push": ("push", "#A371F7"), "grasp": ("close", "#E6EDF3"),
       "hold": ("hold, carry, turn", "#2EA043"), "release": ("open", "#F85149"), "idle": ("wait", "#30363D")}
FREE = {"approach", "descend", "reach", "transit", "insert", "p_go", "p_down", "g_reach", "g_insert", "g_go", "p_reach",
        "p_up", "p_clear", "p_off", "rise", "home", "away", "retreat", "reopen", "retry"}
GRASP = {"grasp", "close", "g_close"}
PUSH = {"push", "p_push"}
HOLD = {"lift", "carry", "lower", "turn", "hold", "g_lift", "g_hold", "c_go", "c_lower"}
RELEASE = {"release", "c_release"}


def category(ph, closed):
    if ph in GRASP:
        return "grasp"
    if ph in PUSH:
        return "push"
    if ph in HOLD or (ph == "wait" and closed):
        return "hold"
    if ph in RELEASE:
        return "release"
    if ph in FREE:
        return "free"
    return "idle"


# ---- what each phase does (captions), per controller ------------------------------------------------------------------
DESC = {
    "StackTwoBlocks": {
        "approach": "moves above its block and turns the hand to the block's yaw (mod 90 deg)",
        "descend": "lowers straight down until the pads are around the block",
        "grasp": "closes for 10 steps; a finger opening of 0.15-0.8 means it holds the block",
        "reopen": "missed: opens, rises and approaches again",
        "lift": "lifts its block to carry height",
        "wait": "holds its block 14 cm from the base block until the other arm has let go and moved away",
        "carry": "carries the block over the base block's centre, seen again by the cameras",
        "lower": "lowers onto the base block and stops when it rests on it",
        "release": "opens: the block rests on the other one",
        "retreat": "rises clear",
        "late": "waits until the other arm has lifted its block out of the way (the blocks are close together)",
        "hold": "grips the base block where it stands until the other arm holds its block: both blocks held at once",
        "rise": "rises straight up until the fingers clear the block",
        "home": "steps aside toward its own side",
        "done": "waits out of the way"},
    "RotateValve": {
        "settle": "waits, grippers closed, until the valves stop moving in the depth images (they drop onto the table)",
        "reach": "opens and moves above its wheel on a minimum-jerk path; both arms take the same time",
        "descend": "lowers the pads around the wheel rim, then waits until the height is within 1.5 mm",
        "close": "closes; a finger opening of 0.45-0.92 means the rim is between the pads",
        "retry": "missed the rim: opens, rises and descends again",
        "turn": "turns the hand 36 deg about the wheel axis, counterclockwise seen from above",
        "hold": "keeps the grasp; if the episode has not ended 1.2 s later, it turns 20 deg further"},
    "PackBox": {
        "start": "waits",
        "transit": "moves, fingers closed (a compact pusher), above the run-in point beside its flap's tip",
        "descend": "lowers to the run-in point; the second arm waits here until the first flap passes 130 deg",
        "push": "lifts the flap by its tip edge, then folds it past 90 deg; the aim advances 2.5 deg per step",
        "hold": "presses the closed flap and holds still",
        "release": "its flap is closed: opens the fingers and rises 4 cm",
        "away": "moves to its own side, clear of the box and of the other arm"},
    "CubeHandover": {
        ("G", "approach"): "moves above its grasp point, 6.5 cm from the rod centre on its own end, fingers across",
        ("G", "descend"): "lowers straight down onto the rod",
        ("G", "grasp"): "closes; a finger opening of 0.15-0.8 means it holds the rod",
        ("G", "lift"): "lifts the rod 6 cm and brings it to the handover pose, turning it along the y axis",
        ("G", "hold"): "holds the rod still at the handover pose",
        ("G", "release"): "opens: now only the other gripper holds the rod",
        ("G", "reopen"): "missed: opens and looks again from above and aside",
        ("R", "approach"): "waits 6 cm beyond its grasp point, out along the rod's axis, on its own side",
        ("R", "insert"): "slides in along the rod with open fingers while the giver brings the rod the last 5 cm",
        ("R", "grasp"): "closes on the rod's free end",
        ("R", "hold"): "holds the rod",
        ("R", "reopen"): "missed: backs out and tries again"},
    "LiftTray": {
        "approach": "moves above the end wall on its side, fingers open, the hand turned across the wall",
        "descend": "lowers until the pads are 2-4.5 cm below the wall top; both arms move together",
        "grasp": "closes; both arms holding (finger opening 0.15-0.8) for 2 steps starts the lift",
        "lift": "pulls the tray 20 cm toward the robot while lifting 12 cm (pulling is what makes the lift possible)",
        "ready": "holds its wall and waits until the other hand holds too",
        "retry": "missed: both arms open, rise, perceive the tray again and grasp again",
        "setdown": "a wall slipped: both hands go back down to the grasp height before a joint retry",
        "relower": "no success after 55 lift steps: puts the tray back down, grasp kept, to pull again from there"},
    "PickSingleBookFromTable": {
        "p_reach": "moves the pushing hand above the book's far side, the hand tilted",
        "p_down": "lowers the fingertips to just above the counter behind the book",
        "p_push": "pushes the book toward the robot so it overhangs the counter's front edge",
        "p_off": "backs off the book",
        "p_up": "rises clear",
        "p_clear": "moves out of the head camera's view: the book is perceived again",
        "g_reach": "moves to the pinch pose: hand tilted 40 deg, lower finger aimed under the overhanging edge",
        "g_insert": "slides in: lower pad under the book's bottom edge, upper pad on its top face",
        "g_close": "closes for 8 steps; an opening below 0.12 would mean it closed on nothing (then it tries again)",
        "g_lift": "lifts the book 20 cm",
        "g_hold": "holds the book up",
        "idle": "waits out of the way"},
    "StackSingleBookShelf": {
        "p_go": "crosses over the book with the hand pitched 25 deg forward, clear of the lower shelf board",
        "p_down": "lowers the front finger behind the book's far face, 6 mm above the counter",
        "p_push": "pushes the book toward the robot with one finger of the open gripper",
        "g_go": "moves to the pinch pose behind the spine, the hand tilted",
        "g_insert": "slides in: lower pad under the spine's bottom edge, upper pad on the top face",
        "g_close": "closes; an opening below 0.12 would mean it closed on nothing (then it tries again)",
        "c_go": "lifts while turning the book nose-down and carries it in over the lower shelf board",
        "c_lower": "lowers until the book's far edge touches the board",
        "c_release": "opens: the book stays on the shelf",
        "idle": "lowers to mid height and waits"},
}


def describe(task, k, ph, snap):
    D = DESC[task]
    if task == "CubeHandover":
        role = "G" if snap.get("G") == k else "R"
        return D.get((role, ph), ph)
    return D.get(ph, ph)


# ---- geometry --------------------------------------------------------------------------------------------------------
def project(X, res):
    """robot-frame points (..., 3) -> pixel coordinates in a res x res render of the head camera"""
    c = hs.CAMS["head"]
    X = np.asarray(X, float)
    Xc = (X.reshape(-1, 3) - np.array(c["pos"])) @ np.array(c["mat"])
    z = -Xc[:, 2]
    f = (res / 2) / np.tan(np.radians(c["fovy"]) / 2)
    uv = np.stack([Xc[:, 0] * f / z + res / 2 - 0.5, -Xc[:, 1] * f / z + res / 2 - 0.5], 1)
    return uv.reshape(X.shape[:-1] + (2,))


def rect_xy(c, yaw, x0, x1, y0, y1, z, n=8):
    xs = np.r_[np.linspace(x0, x1, n), np.full(n, x1), np.linspace(x1, x0, n), np.full(n, x0), x0]
    ys = np.r_[np.full(n, y0), np.linspace(y0, y1, n), np.full(n, y1), np.linspace(y1, y0, n), y0]
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.stack([c[0] + cy * xs - sy * ys, c[1] + sy * xs + cy * ys, np.full(len(xs), z)], 1)


def kinematics(q):
    """per step and arm: finger pad centre, gripper body position (fk) and gripper yaw"""
    n = len(q)
    pad, body, yaw = np.zeros((n, 2, 3)), np.zeros((n, 2, 3)), np.zeros((n, 2))
    for t in range(n):
        for k in (0, 1):
            p, y, R = hs.fk(q[t, 9 * k:9 * k + 7], k)
            pad[t, k], body[t, k], yaw[t, k] = p + PAD_C * R[:, 2], p, y
    return pad, body, yaw


def arm_phases(task, snaps):
    out = []
    for s in snaps:
        if task == "RotateValve":
            ph = [A["phase"] for A in s["arms"]] if s.get("arms") else ["settle", "settle"]
        elif task == "LiftTray":
            ph = [str(p) for p in s["phase"]]
        elif task in ("PickSingleBookFromTable", "StackSingleBookShelf"):
            p, k = s["phase"], s.get("k")
            if p.startswith("p_") and s.get("push") is not None:
                k = s["push"].get("k", k)
            ph = ["idle", "idle"]
            ph[0 if k is None else int(k)] = p
        else:
            ph = [A["phase"] for A in s["arms"]]
        out.append(ph)
    return out


# ---- drawing helpers ---------------------------------------------------------------------------------------------------
def new_fig():
    return plt.figure(figsize=(W / 100, H / 100), dpi=100)


def ax_px(fig, x, y, w, h, frame=False):
    """axes from the top-left corner, in pixels; pixel data coordinates when frame is False"""
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    if not frame:
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
    return ax


def text_px(fig, x, y, s, **kw):
    return fig.text(x / W, 1 - y / H, s, **kw)


def box_px(fig, x, y, w, h, fc=PANEL, ec=LINE, r=10):
    ax = ax_px(fig, x, y, w, h)
    ax.add_patch(FancyBboxPatch((0.5, 0.5), w - 1, h - 1, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=1))
    return ax


_TTP = TextToPath()


def text_w(s, size, bold=False):
    """width of a string in pixels (Noto Sans at `size` points, 100 dpi)"""
    prop = FontProperties(family="Noto Sans", size=size, weight="bold" if bold else "normal")
    return _TTP.get_text_width_height_descent(s, prop, ismath=False)[0] * 100 / 72


def wrap(s, w_px, size, bold=False):
    """greedy line breaking on measured widths; explicit newlines are kept"""
    out = []
    for para in s.split("\n"):
        line = ""
        for word in para.split(" "):
            cand = f"{line} {word}" if line else word
            if line and text_w(cand, size, bold) > w_px:
                out.append(line)
                line = word
            else:
                line = cand
        out.append(line)
    return "\n".join(out)


def header(fig, task, chap):
    text_px(fig, 24, 34, SHORT.get(task, task), fontsize=21, fontweight="bold", va="center")
    x = W - 24
    for i, (lab, bw) in reversed(list(enumerate((("1  Perceive", 118), ("2  Plan", 88), ("3  Act", 80))))):
        on = chap == i
        x -= bw
        box_px(fig, x, 17, bw, 34, fc=FG if on else PANEL, ec=LINE, r=17)
        text_px(fig, x + bw / 2, 34, lab, ha="center", va="center", fontsize=12.5, color=BG if on else MUTED,
                fontweight="bold" if on else "normal")
        x -= 10


def caption(fig, s, x=16, y=632, w=W - 32, h=74):
    ax = box_px(fig, x, y, w, h)
    for size in (14.5, 13.5, 12.5):
        txt = wrap(s, w - 48, size)
        if txt.count("\n") < 2 or size == 12.5:
            break
    ax.text(20, h / 2, txt, va="center", ha="left", fontsize=size, linespacing=1.45 if txt.count("\n") < 2 else 1.22)


def slide_title(fig, s, sub=None):
    text_px(fig, 24, 84, s, fontsize=17, fontweight="bold", va="center", color=FG)
    if sub:
        text_px(fig, 24, 110, sub, fontsize=12, va="center", color=MUTED)


def panel_label(fig, x, y, s, color=MUTED):
    text_px(fig, x, y - 10, s, fontsize=11.5, color=color, va="bottom")


def show_img(fig, x, y, w, h, img, label=None, cmap=None, vmin=None, vmax=None):
    ax = ax_px(fig, x, y, w, h, frame=True)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest" if img.shape[0] <= 300 else "antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color(LINE)
    if label:
        panel_label(fig, x, y, label)
    return ax


def topdown(fig, x, y, w, h, cx, cy, span, label=None, bar=0.1):
    """top view axes: robot x (away from the robot) points up the screen, robot y to the left, as the head camera sees it.
    cx, cy: centre (robot frame); span: metres across the panel's width"""
    ax = ax_px(fig, x, y, w, h, frame=True)
    sh = span * h / w
    ax.set_xlim(-cy - span / 2, -cy + span / 2)
    ax.set_ylim(cx - sh / 2, cx + sh / 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color(LINE)
    x0, y0 = -cy - span / 2 + span * 0.05, cx - sh / 2 + sh * 0.06
    ax.plot([x0, x0 + bar], [y0, y0], color=FG, lw=2, solid_capstyle="butt")
    ax.text(x0 + bar / 2, y0 + sh * 0.025, f"{bar * 100:.0f} cm", color=FG, fontsize=9.5, ha="center", va="bottom")
    if label:
        panel_label(fig, x, y, label)
    return ax


def td(P):
    P = np.asarray(P)
    return -P[..., 1], P[..., 0]


def plot_axes(fig, x, y, w, h, label=None):
    ax = ax_px(fig, x, y, w, h, frame=True)
    ax.tick_params(labelsize=9.5, length=3)
    ax.grid(color=LINE, lw=0.6)
    ax.set_axisbelow(True)
    for s_ in ax.spines.values():
        s_.set_color(LINE)
    if label:
        panel_label(fig, x, y, label)
    return ax


def bullets(fig, x, y, w, items, size=13, gap=14):
    """a column of (colour or None, bold head, text) items, laid out on measured text heights; returns the y below"""
    r = fig.canvas.get_renderer()
    for col, head, body in items:
        y0 = y
        if head:
            t = text_px(fig, x + 14, y, head, fontsize=size, fontweight="bold", va="top", color=col or FG)
            y += t.get_window_extent(r).height + 4
        if body:
            t = text_px(fig, x + 14, y, wrap(body, w - 18, size - 0.5), fontsize=size - 0.5, va="top", color=FG,
                        linespacing=1.35)
            y += t.get_window_extent(r).height
        if col:
            fig.patches.append(Rectangle((x / W, 1 - (y - 1) / H), 4 / W, (y - y0 - 2) / H, transform=fig.transFigure,
                                         fc=col))
        y += gap
    return y


# ---- frame sinks -------------------------------------------------------------------------------------------------------
def grab(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return a


class Sink:
    def __init__(self, path, ffmpeg, fps=FPS, gif_width=None):
        if gif_width:
            vf = (f"scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=full[p];"
                  f"[b][p]paletteuse=dither=bayer:bayer_scale=4")
            enc = ["-vf", vf, "-loop", "0"]
        else:
            enc = ["-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.p = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", *enc, str(path)], stdin=subprocess.PIPE)
        self.frames = 0

    def put(self, img, k=1):
        b = np.ascontiguousarray(img).tobytes()
        for _ in range(k):
            self.p.stdin.write(b)
        self.frames += k

    def close(self):
        self.p.stdin.close()
        assert self.p.wait() == 0


# ---- the episode context -------------------------------------------------------------------------------------------------
class Episode:
    def __init__(self, d):
        self.d = Path(d)
        self.meta = json.loads((self.d / "meta.json").read_text())
        self.task = self.meta["task"]
        tr = np.load(self.d / "trace.npz")
        self.q, self.grip, self.act = tr["q"], tr["grip"], tr["act"]
        self.O = dict(np.load(self.d / "obs.npz"))
        with open(self.d / "snaps.pkl", "rb") as f:
            S = pickle.load(f)
        self.snaps, self.plans, self.flaps = S["snaps"], S["plans"], S["flaps"]
        with open(self.d / "explain.pkl", "rb") as f:
            self.ex = pickle.load(f)
        self.n = len(self.act)  # control steps; frames 0..n
        self.res = self.meta["res"]
        self.pad, self.body, self.yaw = kinematics(self.q)
        self.ph = arm_phases(self.task, self.snaps)
        closed = self.act[:, 12:14] > 0.5
        self.cat = [[category(self.ph[t][k], bool(closed[t, k])) for k in (0, 1)] for t in range(self.n)]
        self.crop = self._crop()

    def snap(self, t):
        return self.snaps[min(t, self.n - 1)]

    def hi(self, t):
        return Image.open(self.d / "hi" / f"{t:04d}.jpg")

    def _crop(self, aspect=760 / 560):
        """main-view crop of the head render: where the hands push, grasp, hold or let go, and the fitted object"""
        act = [self.pad[t, k] for t in range(self.n) for k in (0, 1) if self.cat[t][k] in ("push", "grasp", "hold", "release")]
        pts = [project(np.array(act), self.res)]
        pts.append(project(model_points(self, self.ex["t"]), self.res))
        A = np.concatenate(pts)
        lo, hi = np.percentile(A, 1, 0), np.percentile(A, 99, 0)
        cx, cy = (lo + hi) / 2
        w = max(hi[0] - lo[0], (hi[1] - lo[1]) * aspect, 0.45 * self.res) * 1.35
        w = min(w, self.res, self.res * aspect)
        h = w / aspect
        x0 = float(np.clip(cx - w / 2, 0, self.res - w))
        y0 = float(np.clip(cy - h / 2 + 0.04 * h, 0, self.res - h))
        return x0, y0, w, h


# ---- each controller's estimate, for the overlays ---------------------------------------------------------------------
def book_outline(ex, pose, z):
    (bx, by), (rx, ry) = ex["outline"]["block"], ex["outline"]["ribbon"]
    return [rect_xy(pose["xy"], pose["yaw"], bx[0], bx[1], by[0], by[1], z),
            rect_xy(pose["xy"], pose["yaw"], rx[0], rx[1], ry[0], ry[1], z, n=3)]


def tray_points(est, n=900):
    z = np.load(REPO / "submission" / "policy" / "tray_mesh.npz")
    V, F = z["V"].astype(float), z["F"].astype(np.int64)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    rng = np.random.default_rng(0)
    i = rng.choice(len(area), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n))[:, None], rng.random(n)[:, None]
    S = (1 - r1) * A[i] + r1 * (1 - r2) * B[i] + r1 * r2 * C[i]
    c, s = np.cos(est["th"]), np.sin(est["th"])
    return np.stack([est["c"][0] + c * S[:, 0] - s * S[:, 1], est["c"][1] + s * S[:, 0] + c * S[:, 1], est["z0"] + S[:, 2]], 1)


def model_points(E, t):
    """points of the controller's object estimate at step t (for the crop)"""
    s, ex, task = E.snap(t), E.ex, E.task
    if task == "StackTwoBlocks":
        return np.concatenate([rect_xy(b["xy"], b["yaw"], -0.0215, 0.0215, -0.0215, 0.0215, b["z"] + 0.0215) for b in s["blk"]])
    if task == "RotateValve":
        return np.concatenate([p[::10] for p in ex["posed"]])
    if task == "PackBox":
        return np.c_[ex["rect"], np.full(len(ex["rect"]), 0.994)]
    if task == "CubeHandover":
        r = s["rod"]
        return rect_xy(r["c"], np.arctan2(r["u"][1], r["u"][0]), -0.1, 0.1, -0.02, 0.02, r["c"][2] + 0.02)
    if task == "LiftTray":
        return tray_points(ex["est"], 300)
    return np.concatenate(book_outline(ex, s["book"], ex["est"]["h_top"]))


class Overlay:
    """draws robot-frame geometry onto the main view (crop coordinates)"""

    def __init__(self, E, ax, scale, t):
        self.E, self.ax, self.k = E, ax, scale
        self.x0, self.y0 = E.crop[0], E.crop[1]
        self.depth = E.O["depth_head"][t]

    def visible(self, X):
        """points not hidden behind something nearer in the head camera's depth image at this step"""
        c = hs.CAMS["head"]
        Xc = (np.asarray(X, float) - np.array(c["pos"])) @ np.array(c["mat"])
        z = -Xc[:, 2]
        n = self.depth.shape[0]
        f = (n / 2) / np.tan(np.radians(c["fovy"]) / 2)
        u = np.clip(np.floor(Xc[:, 0] * f / z + n / 2).astype(int), 0, n - 1)
        v = np.clip(np.floor(-Xc[:, 1] * f / z + n / 2).astype(int), 0, n - 1)
        return z <= self.depth[v, u] + 0.012

    def uv(self, X):
        uv = project(X, self.E.res)
        return (uv[..., 0] - self.x0) * self.k, (uv[..., 1] - self.y0) * self.k

    def line(self, X, color, lw=2.0, ls="-", alpha=1.0, z=5):
        u, v = self.uv(X)
        self.ax.plot(u, v, color=color, lw=lw, ls=ls, alpha=alpha, zorder=z, solid_capstyle="round",
                     path_effects=[pe.withStroke(linewidth=lw + 2, foreground="#000000", alpha=0.35 * alpha)])

    def poly(self, X, color, alpha=0.18, lw=2.0):
        u, v = self.uv(X)
        self.ax.add_patch(Polygon(np.c_[u, v], closed=True, fc=color, ec=color, alpha=alpha, lw=0, zorder=4))
        self.line(np.r_[X, X[:1]], color, lw=lw)

    def dots(self, X, color, s=3.0, alpha=0.6, z=4):
        X = np.asarray(X)[self.visible(X)]
        u, v = self.uv(X)
        self.ax.scatter(u, v, s=s, c=color, alpha=alpha, lw=0, zorder=z)

    def arrow(self, X0, X1, color, lw=2.4):
        (u0, u1), (v0, v1) = self.uv(np.array([X0, X1]))
        self.ax.annotate("", (u1, v1), (u0, v0), zorder=8, arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.4",
                                                                           color=color, lw=lw, ls="--"))

    def mark(self, X, color, m="o", s=60, z=7):
        u, v = self.uv(np.atleast_2d(X))
        self.ax.scatter(u, v, s=s, c=color, marker=m, edgecolors="white", linewidths=1.2, zorder=z)

    def tag(self, X, s, color, dx=12, dy=-12, size=11.5, ha="left"):
        u, v = self.uv(np.atleast_2d(X))
        self.ax.text(u[0] + dx, v[0] + dy, s, color=color, fontsize=size, fontweight="bold", ha=ha, va="center",
                     zorder=9, path_effects=HALO)


def draw_estimate(E, o, t, plan_view=False):
    """the controller's object estimate (green) and task geometry at step t"""
    s, ex, task = E.snap(t), E.ex, E.task
    tt = min(t, E.n - 1)
    if task == "StackTwoBlocks":
        Sa, A = s["s"], s["arms"]
        for i, b in enumerate(s["blk"]):
            if i == s["P"] and A[Sa]["off"] is not None and A[Sa]["phase"] in ("lift", "wait", "carry", "lower", "release"):
                continue  # held: the controller tracks it through the hand, not the cameras
            xy, z, yaw = b["xy"], b["z"], b["yaw"]
            o.poly(rect_xy(xy, yaw, -0.0215, 0.0215, -0.0215, 0.0215, z + 0.0215, n=2)[:-1], C_FIT)
            o.tag(np.r_[xy, z + 0.03], "pick" if i == s["P"] else "base", C_FIT, dx=-18, dy=-22, ha="right")
    elif task == "RotateValve":
        if t < ex["t"]:
            return
        for k, P in enumerate(ex["posed"]):
            o.dots(P[::4], C_FIT, s=2.2, alpha=0.45)
            o.mark(ex["est"][k]["wheel"], C_FIT, m="P", s=70)
        for k in (0, 1):
            if E.ph[tt][k] in ("turn", "hold"):
                t1 = next(j for j in range(E.n) if E.ph[j][k] == "turn")
                d = np.unwrap(E.yaw[t1:t + 1, k])
                c = np.asarray(ex["est"][k]["wheel"])
                a = E.yaw[t1, k] + np.linspace(0, d[-1] - d[0], 24)
                arc = np.stack([c[0] + 0.065 * np.cos(a), c[1] + 0.065 * np.sin(a), np.full(24, c[2] + 0.02)], 1)
                o.line(arc, C_ARM[k], lw=3.5)
                o.mark(arc[-1], C_ARM[k], m="o", s=30)
                o.tag(c + [0, 0, 0.13], f"turned {np.degrees(d[-1] - d[0]):.0f} deg", C_ARM[k], dx=0, dy=0, ha="center")
    elif task == "PackBox":
        o.line(np.c_[ex["rect"], np.full(len(ex["rect"]), 0.994)], C_FIT, lw=2)
        F = E.flaps[tt]
        for k in (0, 1):
            if F and F[k] is not None:
                o.poly(F[k], C_FIT, alpha=0.22, lw=1.6)
    elif task == "CubeHandover":
        r, G = s["rod"], s["G"]
        if E.ph[tt][G] in ("approach", "descend", "grasp") or plan_view:
            o.poly(rect_xy(r["c"], np.arctan2(r["u"][1], r["u"][0]), -0.1, 0.1, -0.02, 0.02, r["c"][2] + 0.02, n=2)[:-1], C_FIT)
        if s.get("C_h") is not None and E.ph[tt][G] in ("lift", "hold"):
            o.mark(s["C_h"], C_PLAN, m="X", s=70)
            o.tag(s["C_h"], "handover", C_PLAN, dx=12, dy=14)
    elif task == "LiftTray":
        if plan_view or E.ph[tt][0] in ("approach", "descend", "grasp"):
            o.dots(tray_points(ex["est"]), C_FIT, s=3, alpha=0.5)
        tr = s["tray"]
        for k in (0, 1):
            if plan_view or E.ph[tt][k] in ("approach", "descend"):
                h = np.r_[tr["h"][k], 0.97]
                o.mark(h, C_ARM[k], m="o", s=55)
        if E.ph[tt][0] == "lift":
            tl = next(j for j in range(E.n) if E.ph[j][0] == "lift")
            d = np.array([0.345 - ex["est"]["c"][0], 0.0, 0.12])  # tray centre to x = X_PULL, hands up LIFT
            for k in (0, 1):
                o.arrow(E.pad[tl, k], E.pad[tl, k] + d, C_PLAN)
            o.tag(E.pad[tl, 0] + d, "lift target: 20 cm closer, 12 cm up", C_PLAN, dx=0, dy=-18, ha="left")
    else:  # the two book tasks
        edge = np.array([[0.4959, y, 0.7901] for y in np.linspace(-0.32, 0.32, 20)])
        o.line(edge, C_PLAN, lw=1.6, ls="--", alpha=0.85)
        o.tag(edge[3], "counter edge", C_PLAN, dx=0, dy=16, size=10.5, ha="center")
        ph = E.ph[tt][0] if E.ph[tt][0] != "idle" else E.ph[tt][1]
        if plan_view or not ph.startswith(("g_lift", "g_hold", "c_")):
            for X in book_outline(ex, s["book"], ex["est"]["h_top"]):
                o.line(X, C_FIT, lw=2)
        pred = (s.get("push") or {}).get("pred")
        if pred is not None and (plan_view or ph.startswith("p_")):
            for X in book_outline(ex, pred, ex["est"]["h_top"]):
                o.line(X, C_PLAN, lw=1.8, ls=(0, (3, 2)))
            o.tag(np.r_[pred["xy"], ex["est"]["h_top"]], "after the push (predicted)", C_PLAN, dx=70, dy=10, ha="left")


# ---- 3 Act: one frame per control step -----------------------------------------------------------------------------------
MX, MY, MW, MH = 16, 64, 760, 560
RX, RW = 792, 472


def gantt(fig, E, t, y=64, h=168):
    box_px(fig, RX, y, RW, h)
    text_px(fig, RX + 16, y + 20, "PHASES", fontsize=10.5, color=MUTED, fontweight="bold", va="center")
    text_px(fig, RX + RW - 16, y + 20, f"{t / 20:.2f} s of {E.n / 20:.2f} s", fontsize=10.5, color=MUTED, ha="right",
            va="center")
    ax = ax_px(fig, RX + 100, y + 36, RW - 116, 92)
    ax.set_xlim(0, E.n)
    ax.set_ylim(2, 0)
    for k in (0, 1):
        text_px(fig, RX + 16, y + 36 + 23 + 46 * k, ARM[k], fontsize=11.5, color=C_ARM[k], fontweight="bold", va="center")
        t0 = 0
        for j in range(1, E.n + 1):
            if j == E.n or E.ph[j][k] != E.ph[t0][k] or E.cat[j][k] != E.cat[t0][k]:
                col = CAT[E.cat[t0][k]][1]
                ax.add_patch(Rectangle((t0, k + 0.1), j - t0, 0.8, fc=col, ec=BG, lw=1, alpha=1.0 if t0 <= t else 0.4))
                lab = E.ph[t0][k]
                if text_w(lab, 8.5) + 6 < (j - t0) / E.n * (RW - 116):
                    ax.text((t0 + j) / 2, k + 0.5, lab, fontsize=8.5, ha="center", va="center",
                            color=BG if E.cat[t0][k] == "grasp" else FG, alpha=1.0 if t0 <= t else 0.6)
                t0 = j
    ax.axvline(t, color=FG, lw=1.8)
    lx = RX + 16
    for key in ("free", "push", "grasp", "hold", "release", "idle"):
        fig.patches.append(Rectangle((lx / W, 1 - (y + h - 18 + 5) / H), 10 / W, 10 / H, transform=fig.transFigure,
                                     fc=CAT[key][1], ec=LINE, lw=0.5))
        text_px(fig, lx + 14, y + h - 18, CAT[key][0], fontsize=9.5, color=MUTED, va="center")
        lx += 14 + text_w(CAT[key][0], 9.5) + 13


def now_panel(fig, E, t, y=240, h=84):
    box_px(fig, RX, y, RW, h)
    tt = min(t, E.n - 1)
    g = E.grip[t]
    for k in (0, 1):
        yy = y + 23 + 38 * k
        text_px(fig, RX + 16, yy, ARM[k], fontsize=11.5, color=C_ARM[k], fontweight="bold", va="center")
        text_px(fig, RX + 112, yy, E.ph[tt][k], fontsize=14.5, color=FG, fontweight="bold", va="center",
                family="DejaVu Sans Mono")
        st = "holding" if 0.15 < g[k] < 0.8 else ("open" if g[k] >= 0.8 else "closed")
        col = C_OK if st == "holding" else MUTED
        text_px(fig, RX + RW - 16, yy, st, fontsize=10.5, color=col, ha="right", va="center",
                bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.8", fc=BG, ec=col, lw=1))


def signal(E):
    """the task's own progress signal: (label, per-step values (n+1, 2), reference lines)"""
    task = E.task
    if task == "RotateValve":
        v = np.zeros((E.n + 1, 2))
        for k in (0, 1):
            t1 = next((j for j in range(E.n) if E.ph[j][k] == "turn"), E.n)
            d = np.degrees(np.unwrap(E.yaw[:, k]))
            v[:, k] = np.where(np.arange(E.n + 1) >= t1, d - d[min(t1, E.n)], 0.0)
        return "hand turn since the grasp (deg, from the joint angles)", v, [(27, "valve success: 27 deg")]
    if task == "PackBox":
        v = np.array([[np.degrees(A["th"]) for A in E.snap(t)["arms"]] for t in range(E.n + 1)])
        return "flap angle the hand aims at (deg)", np.clip(v, 0, 185), [(90, "upright"), (170, "closed")]
    base = {"StackTwoBlocks": 0.9485, "CubeHandover": 0.95, "LiftTray": 0.87}.get(task, 0.7901)
    where = "table" if task in ("StackTwoBlocks", "CubeHandover", "LiftTray") else "counter"
    return f"finger pad height above the {where} (cm)", (E.pad[:, :, 2] - base) * 100, []


def plots(fig, E, t, sig, y=362):
    ax = plot_axes(fig, RX + 40, y, RW - 52, 88, "gripper opening (green band: holding something)")
    ax.axhspan(0.15, 0.8, color=C_OK, alpha=0.13, lw=0)
    for k in (0, 1):
        ax.plot(np.arange(t + 1), E.grip[:t + 1, k], color=C_ARM[k], lw=2, alpha=0.95)
    ax.set_xlim(0, E.n)
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks(np.arange(0, E.n + 1, 20))
    ax.set_xticklabels([])
    ax.axvline(t, color=FG, lw=1, alpha=0.6)
    lab, v, refs = sig
    ax = plot_axes(fig, RX + 40, y + 124, RW - 52, 92, lab)
    for val, s in refs:
        ax.axhline(val, color=C_PLAN, lw=1, ls="--", alpha=0.8)
        ax.text(E.n * 0.01, val, s, color=C_PLAN, fontsize=8.5, va="bottom")
    for k in (0, 1):
        ax.plot(np.arange(t + 1), v[:t + 1, k], color=C_ARM[k], lw=2, alpha=0.95)
    ax.set_xlim(0, E.n)
    lo = min(np.nanmin(v), min([r[0] for r in refs], default=np.inf))
    hi = max(np.nanmax(v), max([r[0] for r in refs], default=-np.inf))
    pad = 0.1 * (hi - lo + 1e-6)
    ax.set_ylim(lo - pad, hi + pad * 2)
    ax.set_xticks(np.arange(0, E.n + 1, 20))
    ax.set_xticklabels([f"{x / 20:.0f}" for x in np.arange(0, E.n + 1, 20)])
    ax.set_xlabel("time (s)", fontsize=9.5, labelpad=1)
    ax.axvline(t, color=FG, lw=1, alpha=0.6)


def main_view(fig, E, t, trails=True, plan_view=False):
    x0, y0, w, h = E.crop
    img = E.hi(t).crop((int(x0), int(y0), int(x0 + w), int(y0 + h))).resize((MW, MH), Image.LANCZOS)
    ax = ax_px(fig, MX, MY, MW, MH)
    ax.imshow(np.asarray(img), extent=(0, MW, MH, 0), interpolation="none")
    ax.add_patch(FancyBboxPatch((0.5, 0.5), MW - 1, MH - 1, boxstyle="round,pad=0,rounding_size=10", fc="none", ec=LINE, lw=1))
    o = Overlay(E, ax, MW / w, t)
    draw_estimate(E, o, t, plan_view)
    tt = min(t, E.n - 1)
    for k in (0, 1):
        P = E.plans[tt][k] if not plan_view else E.plans[E.ex["t"]][k]
        if P is not None and len(P) > 1 and E.ph[tt][k] not in ("release", "away", "idle", "c_release"):
            o.line(P, C_ARM[k], lw=1.8, ls=(0, (4, 3)), alpha=0.9)
        if trails:
            a = max(0, t - 40)
            seg = E.pad[a:t + 1, k]
            for j in range(len(seg) - 1):
                c = E.cat[min(a + j, E.n - 1)][k]
                o.line(seg[j:j + 2], C_ARM[k], lw=3.2 if c in ("push", "hold") else 1.8, alpha=0.25 + 0.75 * (j + 1) / len(seg), z=6)
        o.mark(E.pad[t, k], C_ARM[k], s=70)
        if not plan_view:
            o.tag(E.pad[t, k], E.ph[tt][k], C_ARM[k], dx=14 if k == 1 else -14, dy=-16, ha="left" if k == 1 else "right")
    return ax, o


def act_frame(E, t, sig):
    fig = new_fig()
    header(fig, E.task, 2)
    ax, _ = main_view(fig, E, t)
    ax.text(14, 22, f"t = {t / 20:.2f} s", fontsize=12, color=FG, fontweight="bold", va="center",
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.6", fc=BG, ec=LINE, alpha=0.85))
    if E.task == "RotateValve" and t < E.ex["t"]:
        ax.text(MW / 2, MH - 30, "waiting for the valves to settle", fontsize=14, color=C_PLAN, ha="center",
                fontweight="bold", path_effects=HALO)
    gantt(fig, E, t)
    now_panel(fig, E, t)
    plots(fig, E, t, sig)
    tt = min(t, E.n - 1)
    ph = E.ph[tt]
    s = E.snap(t)
    if ph[0] == ph[1]:
        cap = f"Both arms: {describe(E.task, 0, ph[0], s)}."
    else:
        parts = [f"{ARM[k]}: {describe(E.task, k, ph[k], s)}." for k in (0, 1)]
        cap = "  ".join(parts)
    caption(fig, cap)
    return grab(fig)


# ---- title, plan and result cards --------------------------------------------------------------------------------------
def title_card(E):
    fig = new_fig()
    bg = np.asarray(E.hi(0).resize((W, W), Image.LANCZOS))[(W - H) // 2:(W - H) // 2 + H]
    ax = ax_px(fig, 0, 0, W, H)
    ax.imshow(bg, extent=(0, W, H, 0))
    ax.add_patch(Rectangle((0, 0), W, H, fc=BG, alpha=0.82))
    text_px(fig, 64, 150, SHORT.get(E.task, E.task), fontsize=34, fontweight="bold", va="center")
    text_px(fig, 64, 205, GOAL[E.task], fontsize=19, color=FG, va="center")
    text_px(fig, 64, 250, "How the hand-designed controller works, step by step", fontsize=15, color=MUTED, va="center")
    steps = [("RGB-D images", "head + front cameras,\n256 x 256"), ("3-D points", "calibrated camera\npose and lens"),
             ("Object pose", "fit of the object's\nknown geometry"), ("Plan", "which arm does what,\ngrasp and push points"),
             ("Phase machine", "per arm: smooth paths,\nP-control, checks"), ("Actions, 20 Hz", "end-effector deltas\n+ grippers")]
    bw, gap, y = 178, 22, 330
    x = 64
    for i, (a, b) in enumerate(steps):
        box_px(fig, x, y, bw, 112, fc=PANEL, ec=C_FIT if i in (1, 2) else LINE)
        text_px(fig, x + bw / 2, y + 34, a, fontsize=14.5, fontweight="bold", ha="center", va="center")
        text_px(fig, x + bw / 2, y + 76, b, fontsize=11, color=MUTED, ha="center", va="center", linespacing=1.3)
        if i < len(steps) - 1:
            fig.patches.append(FancyArrowPatch(((x + bw + 3) / W, 1 - (y + 56) / H), ((x + bw + gap - 3) / W, 1 - (y + 56) / H),
                                               transform=fig.transFigure, arrowstyle="-|>,head_length=6,head_width=4",
                                               color=MUTED, lw=1.5))
        x += bw + gap
    text_px(fig, 64, 500, "Only the standard observations: camera images, depth, joint angles and gripper openings. "
            "Nothing is read from the simulator.", fontsize=13, color=FG, va="center")
    text_px(fig, 64, 620, f"Training seed {E.meta['seed']}  ·  the submitted controller code  ·  the evaluator's "
            "episode loop  ·  the episode is shown at half speed", fontsize=12, color=MUTED, va="center")
    text_px(fig, 64, 650, "LocalModels  ·  CoRL 2026 RoboEval Diagnostic Bimanual Challenge", fontsize=12, color=MUTED,
            va="center")
    return grab(fig)


def plan_card(E, items, cap):
    fig = new_fig()
    header(fig, E.task, 1)
    t = E.ex["t"]
    main_view(fig, E, t, trails=False, plan_view=True)
    box_px(fig, RX, MY, RW, MH)
    text_px(fig, RX + 18, MY + 26, "THE PLAN", fontsize=10.5, color=MUTED, fontweight="bold", va="center")
    bullets(fig, RX + 18, MY + 52, RW - 36, items)
    caption(fig, cap)
    return grab(fig)


def result_card(E):
    fig = new_fig()
    header(fig, E.task, 2)
    ax, _ = main_view(fig, E, E.n, trails=False)
    ok = E.meta["termination"] == "success"
    ax.text(MW / 2, 40, ("Success" if ok else E.meta["termination"]) + f" after {E.n / 20:.2f} s ({E.n} control steps)",
            fontsize=18, fontweight="bold", color=C_OK if ok else "#F85149", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.8", fc=BG, ec=C_OK if ok else "#F85149", lw=1.5))
    box_px(fig, RX, MY, RW, MH)
    nsucc, score = OFFICIAL[E.task]
    text_px(fig, RX + 18, MY + 26, "THE SAME CONTROLLER ON THE OFFICIAL SEEDS", fontsize=10.5, color=MUTED, fontweight="bold", va="center")
    text_px(fig, RX + 18, MY + 80, f"{nsucc} / 50", fontsize=40, fontweight="bold", va="center")
    text_px(fig, RX + 18, MY + 124, "successful episodes (seeds 1000-1049)", fontsize=13, color=MUTED, va="center")
    text_px(fig, RX + 18, MY + 190, f"{score:.3f}", fontsize=40, fontweight="bold", va="center")
    text_px(fig, RX + 18, MY + 234, "task score (outcome, safety, efficiency, coordination)", fontsize=13, color=MUTED, va="center")
    bullets(fig, RX + 18, MY + 290, RW - 36, [(None, None, "Code: submission/policy/ in this repository. The episode in this "
                                                          "video is a training seed, not an official one.")], size=12.5)
    caption(fig, "Result. The controller's estimates (green), planned paths (dashed) and phases came from its own state "
                 "during the episode; rendering used a separate camera and did not change the simulation.")
    return grab(fig)


# ---- 1 Perceive: slides shared by all tasks ----------------------------------------------------------------------------
def depth_img(d):
    return np.clip(d, 0.4, 2.6)


def inputs_slide(E):
    fig = new_fig()
    header(fig, E.task, 0)
    t = E.ex["t"]
    slide_title(fig, "1.1  What the controller sees", f"the observation at step {t} ({t / 20:.2f} s)" +
                (", once the valves have settled" if E.task == "RotateValve" else "") + " - exactly what act() receives")
    O = E.O
    x, y, s = 24, 168, 296
    items = [("head camera  ·  RGB", O["rgb_head"][t], None), ("head camera  ·  depth (m)", depth_img(O["depth_head"][t]), "magma"),
             ("front camera  ·  RGB", O["rgb_front"][t], None), ("front camera  ·  depth (m)", depth_img(O["depth_front"][t]), "magma")]
    for i, (lab, img, cm) in enumerate(items):
        ax = show_img(fig, x + i * (s + 18), y, s, s, img, lab, cmap=cm, vmin=0.4 if cm else None, vmax=2.6 if cm else None)
        if cm:
            im = ax.get_images()[0]
            cax = ax_px(fig, x + i * (s + 18), y + s + 10, s, 10, frame=True)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=9, colors=MUTED, length=2)
            cb.outline.set_edgecolor(LINE)
    used = "head camera only (the front camera looks at the back of the shelf)" if E.task in (
        "PickSingleBookFromTable", "StackSingleBookShelf") else "both cameras"
    caption(fig, f"Inputs: 256 x 256 RGB and depth images from the robot's fixed head and front cameras, plus the arms' "
                 f"joint angles and gripper openings. This controller uses {used}. It reads nothing from the simulator.")
    return grab(fig)


def orbit(P, az, el, center=(0.45, 0.0, 0.95), dist=2.3, f=900.0, w=880, h=556):
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    c = np.asarray(center)
    cam = c + dist * np.array([ce * ca, ce * sa, se])
    fwd = (c - cam) / np.linalg.norm(c - cam)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    V = P - cam
    z = V @ fwd
    return f * (V @ right) / z + w / 2, h / 2 - f * (V @ up) / z, z


def cloud_frame(E, az, el):
    fig = new_fig()
    header(fig, E.task, 0)
    slide_title(fig, "1.2  Depth pixels become 3-D points", "every pixel of both depth images, in the robot's frame")
    P, C = E.ex["cloud"]
    ax = ax_px(fig, 16, 124, 880, 508)
    ax.set_xlim(0, 880)
    ax.set_ylim(556, 48)
    u, v, z = orbit(P.astype(float), az, el)
    o = np.argsort(-z)
    ax.scatter(u[o], v[o], s=3.2, c=C[o] / 255.0, lw=0)
    for n, col in (("head", C_FIT), ("front", C_ALT)):
        cp = np.array(hs.CAMS[n]["pos"])
        R = np.array(hs.CAMS[n]["mat"])
        th = np.tan(np.radians(22.5))
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            ray = R @ np.array([sx * th, sy * th, -1.0])
            (u0, u1), (v0, v1), _ = orbit(np.array([cp, cp + 0.35 * ray]), az, el)
            ax.plot([u0, u1], [v0, v1], color=col, lw=1.2, alpha=0.8, zorder=4)
        cu, cv, _ = orbit(cp[None], az, el)
        ax.scatter(cu, cv, s=90, marker="s", c=col, edgecolors="white", linewidths=1, zorder=5)
        ax.text(cu[0], cv[0] - 16, f"{n} camera", color=col, fontsize=11, ha="center", fontweight="bold", path_effects=HALO)
    for i, (k, col) in enumerate(((0, "#FF6B6B"), (1, "#3FB950"), (2, "#58A6FF"))):
        e = np.zeros(3)
        e[k] = 0.15
        a, b = np.zeros(3), e
        (u0, u1), (v0, v1), _ = orbit(np.array([a, b]) + [0.0, 0.75, 0.6], az, el)
        ax.annotate("", (u1, v1), (u0, v0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
        ax.text(u1, v1, " xyz"[k + 1], color=col, fontsize=11, fontweight="bold")
    box_px(fig, 912, 124, 352, 508)
    f = 128 / np.tan(np.radians(22.5))
    items = [(None, "For pixel (u, v) with depth d:", ""),
             (None, None, "x_c = (u + 0.5 - 128) d / f\ny_c = -(v + 0.5 - 128) d / f\nz_c = -d"),
             (None, None, f"f = 128 / tan(fovy / 2) = {f:.1f} px  (fovy = 45 deg)"),
             (None, "Then into the robot's frame:", ""),
             (None, None, "p = R x_c + c, with the camera's fixed pose (R, c), measured once like a real camera "
                          "calibration."),
             (None, None, f"{len(P):,} of the points are shown here, in their camera colours.")]
    bullets(fig, 930, 146, 316, items, size=12.5, gap=8)
    caption(fig, "Back-projection: each depth pixel becomes a 3-D point. All later steps (selecting the object, fitting "
                 "its model, checking contacts) work on these points in the robot's frame, where the arms' "
                 "forward kinematics also live.")
    return grab(fig)


# ---- 1 Perceive: task-specific slides ------------------------------------------------------------------------------------
def tint(img, mask, col=(63, 224, 160)):
    out = (img.astype(float) * 0.35).astype(np.uint8)
    out[mask] = col
    return out


def slides_s2b(E):
    ex = E.ex
    t = ex["t"]
    out = []
    fig = new_fig()
    header(fig, E.task, 0)
    slide_title(fig, "1.3  Find the red blocks", "colour threshold in both cameras, then a 2-means split in x-y")
    show_img(fig, 24, 168, 296, 296, tint(E.O["rgb_head"][t], ex["masks"]["head"]), "head: red pixels (R > 90, G < 70, B < 70)")
    show_img(fig, 338, 168, 296, 296, tint(E.O["rgb_front"][t], ex["masks"]["front"]), "front: red pixels")
    P, lab = ex["P"], ex["lab"]
    c = P[:, :2].mean(0)
    ax = topdown(fig, 664, 168, 592, 440, c[0], c[1], 0.36, "their 3-D points, top view, split into two blocks", bar=0.05)
    for j, col in ((0, C_FIT), (1, C_ALT)):
        Q = P[lab == bool(j)]
        ax.scatter(*td(Q), s=6, c=col, lw=0, alpha=0.8)
        m = Q.mean(0)
        ax.scatter(*td(m), s=120, marker="X", c=col, edgecolors="white", lw=1.2, zorder=5)
    caption(fig, f"Red pixels of both cameras give {len(P)} 3-D points between 0.9 and 1.4 m height. Starting from the two "
                 "points farthest apart, a 2-means split on x-y separates the two blocks (crosses: cluster centres).")
    out.append((grab(fig), 4.5))

    fig = new_fig()
    header(fig, E.task, 0)
    slide_title(fig, "1.4  Each block's top face gives its centre and yaw", "points within 6 mm of the block's highest "
                "point; yaw = angle of the minimum-area rectangle")
    for i, (bl, col) in enumerate(zip(ex["blocks"], (C_FIT, C_ALT))):
        e = bl["est"]
        ax = topdown(fig, 24 + i * 316, 168, 296, 296, e["xy"][0], e["xy"][1], 0.09, f"block {i + 1}: top face", bar=0.02)
        ax.scatter(*td(bl["Q"]), s=9, c=C_GREY, lw=0)
        ax.scatter(*td(bl["T"]), s=11, c=col, lw=0)
        R = rect_xy(e["xy"], e["yaw"], -0.0215, 0.0215, -0.0215, 0.0215, 0, n=2)
        ax.plot(*td(R), color=C_PLAN, lw=2)
        ax.scatter(*td(e["xy"]), s=80, marker="+", c=C_PLAN, lw=2)
        ax.text(0.04, 0.95, f"yaw {np.degrees(e['yaw']):.1f} deg\n{e['ntop']} top-face points", transform=ax.transAxes,
                fontsize=10.5, va="top", color=FG, path_effects=HALO)
    ax = plot_axes(fig, 700, 168, 552, 296, "bounding-rectangle area vs. angle (1 deg steps)")
    for bl, col in zip(ex["blocks"], (C_FIT, C_ALT)):
        ax.plot(bl["ang"], bl["area"] * 1e4, color=col, lw=2)
        j = int(np.argmin(bl["area"]))
        ax.scatter(bl["ang"][j], bl["area"][j] * 1e4, s=60, c=col, edgecolors="white", zorder=5)
    ax.set_xlabel("rectangle angle (deg, mod 90)", fontsize=10)
    ax.set_ylabel("area (cm$^2$)", fontsize=10)
    caption(fig, "The top face is a 4.3 cm square: its centre is the block's x-y, its height minus 2.15 cm the block's "
                 "centre, and the angle of the smallest bounding rectangle its yaw (mod 90 deg, enough for a square). "
                 "Accuracy on 40 training seeds: 1.1 mm and 0.6 deg (from the controller's notes).")
    out.append((grab(fig), 5.0))
    p = ex["plan"]
    sa, sb = ARM[p["s"]], ARM[p["b"]]
    items = [(None, f"Mode {p['mode']}", "the stacking arm reaches both blocks (each within 0.64 m of its base)"
              if p["mode"] == "A" else "no arm reaches both blocks: a relay through a meeting point mid-table"),
             (C_ARM[p["s"]], sa, "picks the 'pick' block, lifts it, waits until the base block is free again, stacks "
                                 "it on the base block and opens"),
             (C_ARM[p["b"]], sb, "waits until the pick block is lifted, grips the base block where it stands and holds "
                                 "it until the other arm holds its block (both held), then lets go and steps aside")]
    out.append((plan_card(E, items, "Plan: chosen once from the two estimates, as the cheapest (arm, block) assignment "
                                    "whose pick and place are both within comfortable reach."), 5.0))
    return out


def slides_rv(E):
    ex = E.ex
    out = []
    fig = new_fig()
    header(fig, E.task, 0)
    slide_title(fig, "1.3  Wait until the valves settle, then keep the table points",
                "the valves are free bodies: they drop onto the table in the first step")
    ax = plot_axes(fig, 70, 168, 520, 330, "depth pixels that changed (valve region, per step)")
    ch = np.maximum(np.array(ex["changes"], float), 0.5)
    ax.bar(np.arange(1, len(ch) + 1), ch, color=C_ALT, width=0.7)
    ax.set_yscale("log")
    ax.axhline(5, color=C_PLAN, ls="--", lw=1.2)
    ax.text(len(ch) + 0.4, 5, " settled: <= 5", color=C_PLAN, fontsize=10, va="center")
    ax.set_xlabel("step", fontsize=10)
    ax.set_xlim(0.3, len(ch) + 2.8)
    P = ex["P"]
    ax = topdown(fig, 640, 168, 616, 440, 0.5, 0.0, 1.05, f"{len(P):,} depth points above the table (colour: height)")
    o = np.argsort(P[:, 2])
    ax.scatter(*td(P[o]), s=2, c=P[o, 2], cmap="viridis", lw=0, vmin=0.95, vmax=1.11)
    caption(fig, f"The robot holds still (grippers closed) until the valve region of the head and front depth images "
                 f"stops changing: two steps in a row with at most 5 changed pixels, here after {ex['t']} steps. Then "
                 f"it keeps the depth points above the table top in the valves' area.")
    out.append((grab(fig), 4.5))

    fig = new_fig()
    header(fig, E.task, 0)
    slide_title(fig, "1.4  Register the valve's point model to the depth points",
                "3 unknowns per valve: x, y, yaw  ·  coarse 1 cm / 3 deg grid over the spawn range, then two finer passes")
    for k, hm in enumerate(ex["heat"]):
        e = ex["est"][k]
        ax = ax_px(fig, 24 + k * 302, 168, 290, 300, frame=True)
        S = hm["S"].astype(float)
        S[S < -1e8] = np.nan
        dx = 0.005
        ax.imshow(S[:, ::-1], origin="lower", cmap="magma", interpolation="nearest",
                  extent=(-hm["wy"][-1] - dx, -hm["wy"][0] + dx, hm["wx"][0] - dx, hm["wx"][-1] + dx))
        ax.scatter(-e["wheel"][1], e["wheel"][0], s=180, marker="+", c=C_FIT, lw=2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color(LINE)
        panel_label(fig, 24 + k * 302, 168, f"{('left', 'right')[k]} valve: score map")
        ax.text(0.03, 0.03, f"best: {e['inliers']} points\non the model", transform=ax.transAxes, fontsize=10,
                color=FG, path_effects=HALO, va="bottom")
    text_px(fig, 24, 420, "score at each candidate wheel position (best yaw);\ntop view, 1 cm grid", fontsize=10.5,
            color=MUTED, va="top")
    P, inl = ex["P"], ex["inl"]
    ax = topdown(fig, 640, 168, 616, 440, 0.47, 0.0, 1.12, "fitted models (yellow) and the depth points they explain (green)")
    ax.scatter(*td(P[~inl]), s=2, c=C_GREY, lw=0)
    ax.scatter(*td(P[inl]), s=2.5, c=C_FIT, lw=0)
    for M in ex["posed"]:
        ax.scatter(*td(M[::3]), s=0.8, c=C_PLAN, lw=0, alpha=0.6)
    for e in ex["est"]:
        ax.scatter(*td(e["wheel"][:2]), s=140, marker="+", c="white", lw=2)
    caption(fig, "Model: 21,911 surface points of the released valve asset. Score of a pose = depth points within 4 mm "
                 "of it, minus 3 x model points the cameras see through (a partly hidden valve cannot slide along its "
                 "pipe). Median wheel error on 200 training seeds: 1.0 mm.")
    out.append((grab(fig), 5.5))
    items = [(C_ARM[0], ARM[0], "takes the wheel with y > 0 (its side)"), (C_ARM[1], ARM[1], "takes the other wheel"),
             (None, "Same timing", "both arms reach above their wheel in the same time on minimum-jerk paths (dashed), "
                                   "so their heights and speeds stay matched"),
             (None, "Then", "descend around the rim, close, check the finger opening, turn 36 deg counterclockwise; "
                            "+20 deg if no success 1.2 s later")]
    out.append((plan_card(E, items, "Plan: from the two wheel centres. Motion is feed-forward along the planned paths "
                                    "with a small bounded correction while the reference is still (gravity sag)."), 5.0))
    return out


def slides_pb(E):
    ex = E.ex
    out = []
    fig = new_fig()
    header(fig, E.task, 0)
    slide_title(fig, "1.3  The box walls, selected by height", "at reset only the box is in the 0.70-0.95 m band (below the hinges)")
    P = ex["P"]
    ax = topdown(fig, 24, 168, 616, 440, ex["box"]["xy"][0], ex["box"]["xy"][1], 1.0,
                 "top view: wall band (green), open flaps (purple), the rest (grey)")
    rest = ~(ex["wall"] | ex["flap"])
    ax.scatter(*td(P[rest][::3]), s=1.5, c=C_GREY, lw=0)
    ax.scatter(*td(P[ex["flap"]][::2]), s=2, c=C_ALT, lw=0)
    ax.scatter(*td(P[ex["wall"]]), s=2, c=C_FIT, lw=0)
    ax = plot_axes(fig, 700, 168, 540, 300, "height of the depth points (m)")
    ax.hist(P[:, 2], bins=np.arange(0.6, 1.25, 0.005), color=C_GREY, orientation="horizontal")
    ax.axhspan(0.70, 0.95, color=C_FIT, alpha=0.18)
    ax.axhspan(1.00, 1.09, color=C_ALT, alpha=0.18)
    ax.text(ax.get_xlim()[1] * 0.97, 0.825, "walls", color=C_FIT, ha="right", fontsize=11)
    ax.text(ax.get_xlim()[1] * 0.97, 1.045, "open flaps", color=C_ALT, ha="right", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("points", fontsize=10)
    caption(fig, f"The box has no free joint: its pose is estimated once, at the first step. {int(ex['wall'].sum()):,} "
                 f"points fall in the wall band: seen from above they form the box's square wall shell.")
    out.append((grab(fig), 4.5))

    fig = new_fig()
    header(fig, E.task, 0)
    slide_title(fig, "1.4  Yaw from the minimum-area rectangle, centre from the extents",
                "searched over 52-128 deg (1 deg), then 0.1 and 0.01 deg steps")
    ax = plot_axes(fig, 70, 168, 520, 330, "rectangle area around the wall points vs. angle")
    ax.plot(ex["ang"], np.array(ex["area"]), color=C_FIT, lw=2)
    yb = np.degrees(ex["box"]["yaw"])
    ax.axvline(yb, color=C_PLAN, ls="--")
    ax.text(yb + 1, ax.get_ylim()[1] * 0.98, f" {yb:.2f} deg", color=C_PLAN, va="top", fontsize=11)
    ax.set_xlabel("angle (deg)", fontsize=10)
    ax.set_ylabel("area (m$^2$)", fontsize=10)
    Wp = ex["P"][ex["wall"]]
    ax = topdown(fig, 640, 168, 616, 440, ex["box"]["xy"][0], ex["box"]["xy"][1], 0.7, "wall points and the fitted box (yellow)")
    ax.scatter(*td(Wp), s=2, c=C_FIT, lw=0)
    ax.plot(*td(ex["rect"]), color=C_PLAN, lw=2.2)
    ax.scatter(*td(ex["box"]["xy"]), s=150, marker="+", c=C_PLAN, lw=2.5)
    caption(fig, "A square shell has one minimum-area angle in every 90 deg, and the task's yaw range (90 +- 30 deg) "
                 "keeps one inside the search. Centre = middle of the trimmed extents, corrected by the shell's offset "
                 "in the box frame. Error on 30 training seeds: 0.5 mm median, 0.01 deg yaw median.")
    out.append((grab(fig), 5.0))
    p = ex["plan"]
    first = p["first"]
    items = [(None, "Sequenced mode" if p["mode"] == "seq" else "Parallel mode",
              "the arms cannot both push comfortably 30 cm apart along the hinge, so one finishes first"
              if p["mode"] == "seq" else "both arms push at once, 30 cm apart along the hinge"),
             (C_ARM[first], f"{ARM[first]} first", f"easier reach ({p['reach'][first]:.3f} m to the wrist): lifts its "
                                                    "flap by the tip edge and folds it shut"),
             (C_ARM[1 - first], ARM[1 - first], "waits at its run-in point until the first flap passes 130 deg, then "
                                                "folds its own flap"),
             (None, "Pusher", "closed fingers side by side, hand pointing down; dashed: the planned fingertip paths")]
    out.append((plan_card(E, items, "Plan: each fingertip path follows its flap's arc from the asset geometry (hinge "
                                    "line, length, thickness). Each arm holds its own flap on the way."), 5.0))
    return out


def slides_ch(E):
    ex = E.ex
    t = ex["t"]
    out = []
    fig = new_fig()
    header(fig, E.task, 0)
    slide_title(fig, "1.3  Red pixels, then the rod's top face", "the 'cube' is a red rod, 4 x 20 x 4 cm, at any yaw")
    show_img(fig, 24, 168, 296, 296, tint(E.O["rgb_head"][t], ex["masks"]["head"]), "head: red pixels")
    show_img(fig, 338, 168, 296, 296, tint(E.O["rgb_front"][t], ex["masks"]["front"]), "front: red pixels")
    ax = plot_axes(fig, 700, 168, 540, 300, "height of the red points (m)")
    hz, be = ex["hist"]
    ax.barh((be[:-1] + be[1:]) / 2, hz, height=0.0018, color=C_GREY)
    ax.axhspan(ex["top"] - 0.006, ex["top"] + 0.003, color=C_FIT, alpha=0.25)
    ax.text(ax.get_xlim()[1] * 0.97, ex["top"] + 0.005, "top face", color=C_FIT, ha="right", fontsize=11)
    ax.set_xlabel("points", fontsize=10)
    caption(fig, f"Red pixels of both cameras give {len(ex['P'])} points above the table. The top face is the most "
                 "populated 2 mm height bin: the side faces spread over 4 cm, and a few 'flying' silhouette pixels "
                 "can lie above the rod, so a plain maximum would fail.")
    out.append((grab(fig), 4.5))
    fig = new_fig()
    header(fig, E.task, 0)
    slide_title(fig, "1.4  Principal axis = the rod's long axis", "centre = middle of the trimmed extents along and across it")
    r = ex["rod"]
    ax = topdown(fig, 24, 168, 616, 440, r["c"][0], r["c"][1], 0.36, "top view: red points (grey), top face (green)", bar=0.05)
    ax.scatter(*td(ex["P"]), s=5, c=C_GREY, lw=0)
    ax.scatter(*td(ex["T"]), s=7, c=C_FIT, lw=0)
    ax.plot(*td(ex["rect"]), color=C_PLAN, lw=2)
    u = np.asarray(r["u"])
    a, b = r["c"][:2] - 0.13 * u, r["c"][:2] + 0.13 * u
    ax.annotate("", td(b), td(a), arrowprops=dict(arrowstyle="-|>", color=C_PLAN, lw=2))
    ax.scatter(*td(r["c"][:2]), s=150, marker="+", c=C_PLAN, lw=2.5)
    box_px(fig, 664, 168, 592, 440)
    items = [(None, "Estimate", f"centre ({r['c'][0]:.3f}, {r['c'][1]:.3f}) m, axis {np.degrees(r['yaw']):.1f} deg, "
                                f"visible length {r['len'] * 100:.1f} cm of 20, width {r['wid'] * 100:.1f} cm of 4"),
             (None, "Axis", "the eigenvector with the larger eigenvalue of the top-face points' x-y covariance"),
             (None, "Later", "once the rod is lifted, the free end of the held rod is found the same way and used "
                             "if it agrees with the kinematic prediction")]
    bullets(fig, 684, 192, 552, items)
    caption(fig, "Rod perception works on the top face alone; it gives the centre, the long axis (direction up to "
                 "sign) and the visible size, which checks that the whole top face was seen.")
    out.append((grab(fig), 5.0))
    p = ex["plan"]
    G, R = p["G"], p["R"]
    items = [(C_ARM[G], f"{ARM[G]}: giver", "the arm on the rod's side: grasps 6.5 cm from the centre on its own end, "
                                           "fingers across, lifts 6 cm, turns the rod along the y axis"),
             (C_ARM[R], f"{ARM[R]}: receiver", "waits 6 cm out along the rod's axis, then slides in with open fingers "
                                               "and closes on the free end"),
             (None, "Then", "the giver opens: success = only the second gripper holds the rod, off the table"),
             (None, "Speed", "half of the stacking controller's speed: the arms' velocity difference is scored")]
    out.append((plan_card(E, items, "Plan: the handover pose is fixed (rod centre at x = 0.40 m, along y); the giver "
                                    "stops 5 cm short and brings the rod in while the receiver slides in."), 5.0))
    return out


def slides_lt(E):
    ex = E.ex
    out = []
    fig = new_fig()
    header(fig, E.task, 0)
    slide_title(fig, "1.3  The tray's height band and its floor", "points between 1.2 cm above the table and 1.06 m")
    P0 = ex["P0"]
    ax = topdown(fig, 24, 168, 616, 440, ex["est"]["c"][0], ex["est"]["c"][1], 0.9, "top view (colour: height)")
    o = np.argsort(P0[:, 2])
    ax.scatter(*td(P0[o][::2]), s=2, c=P0[o][::2, 2], cmap="viridis", lw=0, vmin=0.88, vmax=1.05)
    ax = plot_axes(fig, 700, 168, 540, 300, "height of those points (m)")
    hz, be = ex["hist"]
    ax.barh((be[:-1] + be[1:]) / 2, hz, height=0.0018, color=C_GREY)
    ax.axhline(ex["zf"], color=C_FIT, lw=1.5)
    ax.text(ax.get_xlim()[1] * 0.97, ex["zf"] + 0.003, "tray floor (most points)", color=C_FIT, ha="right", fontsize=11)
    ax.axhline(ex["z0"], color=C_PLAN, lw=1.2, ls="--")
    ax.text(ax.get_xlim()[1] * 0.97, ex["z0"] - 0.006, "tray bottom = floor - 2.2 cm", color=C_PLAN, ha="right", fontsize=10.5)
    ax.set_xlabel("points", fontsize=10)
    caption(fig, "At the first observation the tray still hangs 5.5 cm above the table (it drops during the first step). "
                 "The floor is the most populated height, so the fit uses the measured height and works before and "
                 "after the drop.")
    out.append((grab(fig), 4.5))
    fig = new_fig()
    header(fig, E.task, 0)
    slide_title(fig, "1.4  Register the tray's mesh to the points",
                "score = points on the model surface  ·  coarse: 3 deg over 180 deg, 1.5 cm over +-9 cm, model dilated 21 mm")
    G = ex["G"]
    ax = plot_axes(fig, 60, 168, 520, 140, "best score per long-axis angle (coarse pass)")
    ax.plot(np.degrees(ex["ths"]), G.reshape(len(G), -1).max(1), color=C_FIT, lw=2)
    ax.axvline(np.degrees(ex["est"]["th"]), color=C_PLAN, ls="--")
    ax.set_xlabel("angle (deg)", fontsize=10)
    j = int(np.argmax(G.reshape(len(G), -1).max(1)))
    ax2 = ax_px(fig, 60 + 150, 388, 220, 220, frame=True)
    cx0, cy0 = ex["c0"]
    X, Y = np.meshgrid(cx0 + ex["offs"], cy0 + ex["offs"], indexing="ij")
    ax2.scatter(-Y.ravel(), X.ravel(), c=G[j].ravel(), cmap="magma", s=190, marker="s", lw=0)
    ax2.scatter(-ex["est"]["c"][1], ex["est"]["c"][0], s=160, marker="+", c=C_FIT, lw=2.5)
    ax2.set_aspect("equal")
    ax2.set_xticks([])
    ax2.set_yticks([])
    panel_label(fig, 210, 388, "score per centre at that angle")
    P = ex["P"]
    ax = topdown(fig, 640, 168, 616, 440, ex["est"]["c"][0], ex["est"]["c"][1], 0.75, "points (grey) and the fitted mesh (green)")
    ax.scatter(*td(P[::2]), s=2, c=C_GREY, lw=0)
    ax.scatter(*td(tray_points(ex["est"], 2500)), s=1.5, c=C_FIT, lw=0, alpha=0.7)
    for k, h in enumerate(ex["tray"]["h"]):
        ax.scatter(*td(np.asarray(h)), s=120, c=C_ARM[k], edgecolors="white", lw=1.2, zorder=5)
    caption(fig, "The tray is symmetric under a half turn, so 180 deg are searched. The coarse pass uses a model widened "
                 "by 21 mm, so wall points count before the pose is close; then 9 mm and 3 mm. Handle points within "
                 "4.2 mm on 160 training seeds. Dots: each arm's grasp point on its end wall.")
    out.append((grab(fig), 5.5))
    items = [(C_ARM[0], ARM[0], "grasps the end wall on its side (+y), at the wall's middle, top-down"),
             (C_ARM[1], ARM[1], "grasps the other end wall"),
             (None, "Together", "the arm with less to go is slowed in proportion, so both arrive together and stay "
                                "at the same height"),
             (None, "Lift", "pull the tray 20 cm toward the robot while rising 12 cm: 17 kg per hand at the initial "
                            "distance saturates the shoulders, about 20 cm closer they can lift it")]
    out.append((plan_card(E, items, "Plan: grasp both end walls, fingers 8 cm open around the 2.4 cm wall; if a grasp "
                                    "misses, both arms open, the tray is perceived again and both grasp again."), 5.0))
    return out


def slides_book(E):
    ex, task = E.ex, E.task
    out = []
    est = ex["est"]
    fig = new_fig()
    header(fig, task, 0)
    slide_title(fig, "1.3  The book's top face and its near side face", "head depth only, 1.2-4.5 cm above the counter")
    P = ex["P"]
    reg = ex["reg"]
    ax = topdown(fig, 24, 168, 616, 440, est["xy"][0], est["xy"][1], 0.42, "top view: top face (green), side face (purple)", bar=0.05)
    ax.scatter(*td(P[reg & ~ex["band"]][::2]), s=3, c=C_GREY, lw=0)
    ax.scatter(*td(P[ex["side"]]), s=6, c=C_ALT, lw=0)
    ax.scatter(*td(P[ex["top"]]), s=5, c=C_FIT, lw=0)
    ax = plot_axes(fig, 700, 168, 540, 300, "height above the counter (cm)")
    hz, be = ex["hist"]
    ax.barh(((be[:-1] + be[1:]) / 2 - 0.7901) * 100, hz, height=0.18, color=C_GREY)
    ax.axhline((est["h_top"] - 0.7901) * 100, color=C_FIT, lw=1.5)
    ax.text(ax.get_xlim()[1] * 0.97, (est["h_top"] - 0.7901) * 100 + 0.1, "top face", color=C_FIT, ha="right", va="bottom", fontsize=11)
    ax.set_xlabel("points", fontsize=10)
    caption(fig, f"The top face is the dominant height 2-4 cm above the counter ({est['n_top']} points). Points 6-20 mm "
                 f"below it ({est['n_side']}) lie on the side faces that face the camera: they pin the book's near edge, "
                 "which the push and the pinch depend on.")
    out.append((grab(fig), 4.5))
    fig = new_fig()
    header(fig, task, 0)
    slide_title(fig, "1.4  Fit the book's outline template", "unknowns: x, y, yaw  ·  start from the top face's "
                "minimum-area rectangle, then refine (2, 1, 0.5 mm)")
    ax = topdown(fig, 24, 168, 616, 440, est["xy"][0], est["xy"][1], 0.26, "fitted outline (yellow) with the ribbon bookmark", bar=0.05)
    ax.scatter(*td(P[ex["top"]]), s=9, c=C_FIT, lw=0)
    ax.scatter(*td(P[ex["side"]]), s=9, c=C_ALT, lw=0)
    for X in book_outline(ex, est, 0):
        ax.plot(*td(X), color=C_PLAN, lw=2.2)
    sc, n_in, through, n_on, n_bad = ex["parts"]
    box_px(fig, 664, 168, 592, 440)
    items = [(None, "Score of a pose", "top points inside the outline + 2 x side points on its edge - 3 x template "
                                       "points the camera sees through - 2 x side points deep inside"),
             (None, "This fit", f"{n_in} inside + 2 x {n_on} on the edge - 3 x {through} seen through - 2 x {n_bad} "
                                f"= {sc}")]
    if est.get("flip_score") is not None:
        items.append((None, "Turned 180 deg", f"score {est['flip_score'][0]}: the 7 mm ribbon makes the outline "
                                              "asymmetric, so the book's orientation is not ambiguous"))
    items.append((None, "Why the side term", "with the top face alone the fit slid about 5 mm toward the camera; "
                                             "the side points fix the near edge"))
    bullets(fig, 684, 192, 552, items)
    caption(fig, "Template: the book's top outline (main block plus the ribbon bookmark sticking out of one end), from "
                 "the released asset. The see-through term rejects outlines over places where the camera sees the "
                 "counter or the floor behind.")
    out.append((grab(fig), 5.5))
    s0 = E.snap(E.ex["t"])
    over = 0.4959 - min(c[0] for c in book_outline(ex, s0["book"], 0)[0][:, :2])
    if task == "PickSingleBookFromTable":
        pk = s0["push"]["k"] if s0.get("push") else s0["plan"]["k"]
        items = [(None, "The problem", "the 8.2 cm gripper cannot span the book's 9.4 cm width, so it pinches across "
                                       "the 2.7 cm thickness: one finger must get under an edge that overhangs the "
                                       "counter"),
                 (None, "Now", f"the book overhangs the counter edge by {max(over, 0) * 100:.1f} cm: not enough"),
                 (C_ARM[pk], f"{ARM[pk]}: push", f"pushes the book {s0['push']['shift'] * 100:.1f} cm toward the "
                                                 "robot, then moves away so the head camera can see it again"),
                 (None, "Then", "refit the book, pinch the overhanging edge with the hand tilted 40 deg, lift 20 cm")]
    else:
        k = s0["k"]
        pinch = next(s_["plan"] for s_ in E.snaps if s_.get("plan"))  # planned after the push
        items = [(None, "The problem", "the gripper cannot span the book's width: it pinches the spine across the "
                                       "thickness, so the spine must overhang the counter's front edge"),
                 (C_ARM[k], f"{ARM[k]}: push", f"spine overhang {s0['push']['over'] * 1000:.0f} mm < 15 mm needed: "
                                               f"push {s0['push']['shift'] * 100:.1f} cm with one finger of the open "
                                               "gripper"),
                 (C_ARM[k], f"{ARM[k]}: pinch", f"closing axis {np.degrees(pinch['psi']):.0f} deg from vertical "
                                                "at the spine under the centre of mass; lift, turn nose-down, carry "
                                                "over the lower board, lower until it touches, open"),
                 (C_ARM[1 - k], ARM[1 - k], "lowers to mid height and waits")]
    cap = ("Plan: a planning IK follows each candidate path from the current joint angles and rejects paths that bring "
           "a joint near its limit or a link under the bookshelf's lower board." if task == "PickSingleBookFromTable" else
           "Plan: a planning IK checks each candidate path for joint-range margin, and a box model of the hand checks it "
           "against the book, the counter and the shelf boards.")
    out.append((plan_card(E, items, cap), 6.0))
    return out


SLIDES = {"StackTwoBlocks": slides_s2b, "RotateValve": slides_rv, "PackBox": slides_pb, "CubeHandover": slides_ch,
          "LiftTray": slides_lt, "PickSingleBookFromTable": slides_book, "StackSingleBookShelf": slides_book}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--gif-width", type=int, default=880)
    ap.add_argument("--stills", default=None, help="folder for PNG copies of the slides")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--only", default=None, help="comma list of parts to draw (title,perceive,plan,act,result) for tests")
    a = ap.parse_args()
    E = Episode(a.trace_dir)
    only = set(a.only.split(",")) if a.only else None
    v = Sink(a.out, a.ffmpeg)
    g = Sink(a.gif, a.ffmpeg, fps=10, gif_width=a.gif_width) if a.gif else None
    stills = Path(a.stills) if a.stills else None
    if stills:
        stills.mkdir(parents=True, exist_ok=True)

    def keep(img, name):
        if stills:
            Image.fromarray(img).save(stills / f"{E.task}_{name}.png", optimize=True)

    if not only or "title" in only:
        v.put(title_card(E), int(4.0 * FPS))
    if not only or "perceive" in only or "plan" in only:
        img = inputs_slide(E)
        keep(img, "1_inputs")
        v.put(img, int(4.5 * FPS))
        for i, az in enumerate(np.radians(225 - 90 * (0.5 - 0.5 * np.cos(np.pi * np.arange(120) / 119)))):
            img = cloud_frame(E, az, np.radians(30 + 8 * np.sin(np.pi * i / 119)))
            if i == 0:
                keep(img, "2_points")
            v.put(img)
        for j, (img, dur) in enumerate(SLIDES[E.task](E)):
            keep(img, f"{j + 3}_{'plan' if j == 2 else 'perception'}")
            v.put(img, int(dur * FPS))
    sig = signal(E)
    if not only or "act" in only:
        for t in range(E.n + 1):
            img = act_frame(E, t, sig)
            v.put(img, STEP_FRAMES)
            if g and t % 2 == 0:
                g.put(img)
        if g:
            g.put(img, 15)
    if not only or "result" in only:
        img = result_card(E)
        keep(img, "9_result")
        v.put(img, int(4.0 * FPS))
    v.close()
    if g:
        g.close()
    print(f"{E.task}: {v.frames} frames ({v.frames / FPS:.1f} s) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
