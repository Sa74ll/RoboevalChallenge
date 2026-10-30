"""Feedback-loop block diagrams: the loop every controller runs (common_loop.png) and, per controller, its blocks, the
signals that close its loops and its phase machine (<Task>_loop.png), for docs/controllers.md and the README.

  python scripts/explainer/loops.py
Writes assets/controllers/common_loop.png and assets/controllers/<Task>_loop.png. Needs numpy, matplotlib and Pillow.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compose import BG, C_ALT, C_FIT, C_PLAN, CAT, FG, GOAL, LINE, MUTED, PANEL, category  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CTL = REPO / "assets" / "controllers"
W, M = 1560, 36  # canvas width and side margin (px)
WHITE = "#C9D1D9"  # the position loop: hand poses from the joint angles
GUARD = {"g": C_ALT, "p": C_FIT, "a": C_PLAN, "h": WHITE}  # what a phase change waits for (finger opening, perception,
# the other arm, hand pose or timer); the same colours as the loop that carries the signal
ACCENT = {"perceive": C_FIT, "plan": C_PLAN, "phases": C_ALT, "motion": WHITE, "sim": MUTED}


def fp(px, bold=False):
    # px at dpi 100; DejaVu Sans supplies the arrows and signs Noto Sans lacks
    return FontProperties(family=["Noto Sans", "DejaVu Sans"], weight="bold" if bold else "normal", size=px * 0.72)


def tw(s, px, bold=False):
    """width in px of the widest line of s"""
    return max(TextPath((0, 0), ln or " ", size=px, prop=fp(px, bold)).get_extents().width for ln in s.split("\n"))


def txt(ax, x, y, s, px, color=FG, bold=False, ha="left", va="baseline", box=None, ls=1.25):
    t = ax.text(x, y, s, fontproperties=fp(px, bold), color=color, ha=ha, va=va, linespacing=ls, zorder=5)
    if box:
        t.set_bbox(dict(facecolor=box, edgecolor="none", pad=2.5))
    return t


def arrow(ax, pts, color, lw=2.0, ls="-", head=True):
    """a polyline through pts with an arrowhead at the end"""
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=2)
    if head:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        L, H = 11, 6
        ax.add_patch(Polygon([(x1, y1), (x1 - L * ux - H * uy, y1 - L * uy + H * ux),
                              (x1 - L * ux + H * uy, y1 - L * uy - H * ux)], closed=True, fc=color, ec="none", zorder=3))


def eye(ax, x, y, r=7):
    """the perception marker: a green ring with a dot"""
    ax.add_patch(Circle((x, y), r, fc="none", ec=C_FIT, lw=2, zorder=6))
    ax.add_patch(Circle((x, y), r * 0.4, fc=C_FIT, ec="none", zorder=6))


def block(ax, x, y, w, h, kind, title, lines, note=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10", fc=PANEL, ec=LINE, lw=1.5,
                                zorder=1))
    ax.plot([x + 14, x + w - 14], [y + h - 44, y + h - 44], color=ACCENT[kind], lw=2.5, alpha=0.9, zorder=2)
    txt(ax, x + 14, y + h - 32, title, 20, ACCENT[kind], bold=True)
    for i, ln in enumerate(lines):
        txt(ax, x + 14, y + h - 72 - 23 * i, ln, 16, FG)
    if note:
        txt(ax, x + 14, y + 14, note, 14, MUTED)


# ---- the loop: Perceive -> Plan -> Phases -> Motion -> Simulator, three feedback lanes back ------------------------------
GAP, BH = 80, 178  # gap between blocks, block height
BW = (W - 2 * M - 4 * GAP) / 5


def loop(ax, top, D):
    """draws the five blocks with their top at y = top and the lanes below; returns the lowest y used"""
    xs = [M + i * (BW + GAP) for i in range(5)]
    y = top - BH
    kinds = ("perceive", "plan", "phases", "motion", "sim")
    titles = ("Perceive", "Plan", "Phase machine", "Motion law", "Simulator")
    for x, k, t in zip(xs, kinds, titles):
        block(ax, x, y, BW, BH, k, t, D[k], D.get(k + "_note"))
    yc = y + BH / 2
    for i, lab in enumerate(D.get("fwd", ("object\npose", "roles,\ntargets", "phase\ntarget", "actions"))):
        x0, x1 = xs[i] + BW, xs[i + 1]
        arrow(ax, [(x0 + 4, yc), (x1 - 4, yc)], FG)
        txt(ax, (x0 + x1) / 2, yc + 12, lab, 14, MUTED, ha="center", va="bottom", ls=1.15)
    # lanes: the position loop (joint angles -> FK -> hand poses), the phase loop (finger openings) and the perception loop
    # (RGB-D); lower lanes drop from further right on the simulator so that no lines cross
    sx = xs[4]
    lanes = [  # (colour, y, drop x on the simulator, [tap x on the blocks], label, label x)
        (WHITE, y - 34, sx + 50, [xs[2] + 170, xs[3] + BW / 2], D["lane_pos"], xs[2] + 182),
        (C_ALT, y - 74, sx + 118, [xs[2] + 70], D["lane_phase"], xs[2] + 82),
        (C_FIT, y - 114, sx + 186, [xs[0] + BW / 2], D["lane_perc"], xs[0] + BW / 2 + 12),
    ]
    for c, ly, dx, taps, lab, lx in lanes:
        ax.plot([dx, dx, min(taps)], [y, ly, ly], color=c, lw=2.2, zorder=2, solid_joinstyle="miter")
        for tx in taps:
            arrow(ax, [(tx, ly), (tx, y + 2)], c, lw=2.2)
        ax.add_patch(Circle((dx, y), 4, fc=c, ec="none", zorder=3))
        txt(ax, lx, ly - 21, lab, 15, c)
    return y - 114 - 34


# ---- the phase machine panel ----------------------------------------------------------------------------------------
PH, PX = 40, 15  # pill height, pill font (px)
G0 = 36  # plain transition (px)
LX = 15  # transition label font (px)
BACK = 38  # spacing of the retry arcs below a row


def pill_cat(name):
    if "|" in name:
        name, cat = name.split("|")
        return name, cat
    return name, category(name, False)


def row_layout(row, x0):
    """x extents of each pill and each transition label of a row"""
    items, x = [], x0
    toks = row["seq"]
    if row.get("start"):
        x += 40
    for i, tok in enumerate(toks):
        if isinstance(tok, str):
            name, cat = pill_cat(tok)
            w = max(78, tw(name, PX) + 26)
            items.append(dict(kind="pill", name=name, cat=cat, x=x, w=w))
            x += w
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if isinstance(nxt, str):
                x += G0
        else:
            lab, c = tok[0], tok[1]
            w = tw(lab, LX) + (40 if len(tok) > 2 and tok[2] == "eye" else 22)
            items.append(dict(kind="guard", lab=lab, c=c, eye=len(tok) > 2 and tok[2] == "eye", x=x + 14, w=w))
            x += w + 28
    return items, x


def phase_panel(ax, top, D):
    rows = D["rows"]
    txt(ax, M + 18, top - 36, "Phase machine", 20, C_ALT, bold=True)
    # legends: phase kinds (the videos' timeline colours) and what a phase change waits for
    x = M + 18 + tw("Phase machine", 20, True) + 40
    for key in ("free", "push", "grasp", "hold", "release", "idle"):
        lab, c = CAT[key]
        ax.add_patch(FancyBboxPatch((x, top - 44), 22, 14, boxstyle="round,pad=0,rounding_size=4", fc=c, ec="none"))
        txt(ax, x + 30, top - 42, lab, 14, MUTED)
        x += 30 + tw(lab, 14) + 26
    x = W - M - 18
    for k, lab in (("h", "hand pose or timer"), ("a", "the other arm"), ("p", "perception"), ("g", "finger opening")):
        x -= tw(lab, 14)
        txt(ax, x, top - 42, lab, 14, GUARD[k])
        x -= 22
    txt(ax, x - 4, top - 42, "Phase changes wait for:", 14, MUTED, ha="right")
    if D.get("panel_note"):
        txt(ax, M + 18, top - 68, D["panel_note"], 14, MUTED)
    x0 = M + 18 + 168
    y = top - (84 if D.get("panel_note") else 70)
    pos = {}
    for r, row in enumerate(rows):
        items, xend = row_layout(row, x0)
        up = 34 if row.get("self") else 0
        hh = max([PH / 2] + [(PH + 12 * it["lab"].count("\n")) / 2 + 4 for it in items if it["kind"] == "guard"])
        y -= up + hh
        yc = y
        txt(ax, M + 18, yc, row["label"], 15, MUTED, bold=True, va="center", ls=1.2)
        if row.get("start"):
            ax.add_patch(Circle((x0 + 6, yc), 6, fc=FG, ec="none", zorder=3))
            arrow(ax, [(x0 + 14, yc), (x0 + 38, yc)], FG)
            if isinstance(row["start"], str):
                txt(ax, x0 + 6, yc - 16, row["start"], 13, WHITE, ha="center", va="top", ls=1.15)
        prev = None
        for it in items:
            if it["kind"] == "pill":
                fc = CAT[it["cat"]][1]
                ax.add_patch(FancyBboxPatch((it["x"], yc - PH / 2), it["w"], PH, boxstyle="round,pad=0,rounding_size=9",
                                            fc=fc, ec="none", alpha=0.95, zorder=3))
                txt(ax, it["x"] + it["w"] / 2, yc, it["name"], PX, BG if it["cat"] == "grasp" else FG, bold=True,
                    ha="center", va="center")
                pos[(r, it["name"])] = (it["x"], it["x"] + it["w"], yc)
                if prev is not None:
                    arrow(ax, [(prev, yc), (it["x"] - 2, yc)], FG, lw=1.8)
                prev = it["x"] + it["w"]
            else:
                c = GUARD[it["c"]]
                cx = it["x"] + it["w"] / 2
                arrow(ax, [(prev, yc), (it["x"] + 6, yc)], FG, lw=1.8, head=False)
                ax.add_patch(FancyBboxPatch((it["x"], yc - PH / 2 - 6 * it["lab"].count("\n")), it["w"],
                                            PH + 12 * it["lab"].count("\n"), boxstyle="round,pad=0,rounding_size=7",
                                            fc=BG, ec=c, lw=1.4, ls=(0, (3, 2)), zorder=3))
                tx = it["x"] + (32 if it["eye"] else 11)
                if it["eye"]:
                    eye(ax, it["x"] + 17, yc)
                txt(ax, tx, yc, it["lab"], LX, c, va="center", ls=1.15)
                prev = it["x"] + it["w"]
        if row.get("end"):
            ex = prev + 22
            arrow(ax, [(prev, yc), (ex - 10, yc)], FG, lw=1.8)
            ax.add_patch(Circle((ex, yc), 9, fc="none", ec=FG, lw=2, zorder=3))
            ax.add_patch(Circle((ex, yc), 5, fc=FG, ec="none", zorder=3))
            txt(ax, ex + 16, yc, row["end"], 14, WHITE, va="center")
        # loops on one phase (above the pill) and retries back to an earlier phase (below the row)
        for name, lab, c in row.get("self", []):
            a, b, _ = pos[(r, name)]
            yt = yc + PH / 2
            arrow(ax, [(b - 18, yt), (b - 18, yt + 16), (a + 18, yt + 16), (a + 18, yt + 2)], GUARD[c], lw=1.6)
            txt(ax, a + 18, yt + 22, lab, 14, GUARD[c])
        yb = yc - PH / 2
        for j, (src, dst, lab, c, *opt) in enumerate(row.get("back", [])):
            a, b, _ = pos[(r, src)]
            a2, b2, _ = pos[(r, dst)]
            dy = 20 + BACK * j
            xs, xd = (a + b) / 2 + 8 * j, (a2 + b2) / 2 - 8 * j
            arrow(ax, [(xs, yb), (xs, yb - dy), (xd, yb - dy), (xd, yb - 2)], GUARD[c], lw=1.6)
            lx = xd + 12  # under its own arc, clear of the deeper arcs' drops
            if opt and opt[0] == "eye":
                eye(ax, lx + 8, yb - dy - 13, r=6)
                lx += 22
            txt(ax, lx, yb - dy - 5, lab, 14, GUARD[c], va="top")
        y = min(yb, yc - hh) - (20 + BACK * (len(row["back"]) - 1) + 30 if row.get("back") else 0) - 26
        if any(src[0] == r for src, *_ in D.get("links", [])):
            y -= 34  # room for the link to the next row
        row["_bottom"] = y
    for (r0, n0), (r1, n1), lab, c in D.get("links", []):
        a, b, y0 = pos[(r0, n0)]
        a2, b2, y1 = pos[(r1, n1)]
        xs, xd = (a + b) / 2, (a2 + b2) / 2
        ym = y0 - PH / 2 - 30
        arrow(ax, [(xs, y0 - PH / 2), (xs, ym), (xd, ym), (xd, y1 + PH / 2 + 2)], GUARD[c], lw=1.8)
        eye(ax, xs + 20, ym, r=6)
        txt(ax, xs + 34, ym, lab, 14, GUARD[c], va="center")
    y += 8
    ax.add_patch(FancyBboxPatch((M, y), W - 2 * M, top - y, boxstyle="round,pad=0,rounding_size=12", fc="none",
                                ec=LINE, lw=1.5, zorder=0))
    return y


def figure(D, path, panel=True):
    """y runs down from 0 at the top; the height is set once everything is drawn (1 unit = 1 px)"""
    fig = plt.figure(figsize=(W / 100, 10), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    txt(ax, M, -48, D["title"], 32, FG, bold=True)
    txt(ax, M + tw(D["title"], 32, True) + 18, -48, D["sub"], 18, MUTED)
    bottom = loop(ax, -84, D)
    if panel:
        bottom = phase_panel(ax, bottom - 6, D)
    H = int(round(-bottom + 24))
    ax.set_xlim(0, W)
    ax.set_ylim(-H, 0)
    fig.set_size_inches(W / 100, H / 100)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
    print(f"{path.name}: {W} x {H}")


# ---- content (from submission/policy/hybrid_*.py; docs/controllers.md has the numbers) ------------------------------
POSITION = "joint angles  →  forward kinematics  →  hand poses, every step"
PHASE = "finger openings: holding or not, every step"

COMMON = dict(
    title="The loop every controller runs", sub="20 Hz; it reads only the observations, never the simulator state",
    perceive=["depth pixels → 3-D points", "fit the object's known", "shape → object pose"],
    plan=["arm roles, grasp and push", "points, paths"],
    phases=["one per arm: approach,", "grasp, lift, ..., release", "+ a retry path"],
    motion=["saturated P-control,", "minimum-jerk paths or", "contact feed-forward"],
    sim=["two Panda arms + scene", "14 numbers per step:", "hand deltas + grippers"],
    lane_pos="position loop:  " + POSITION,
    lane_phase="phase loop:  " + PHASE,
    lane_perc="perception loop:  RGB-D cameras; once, every step or after set events, depending on the task",
)

TASKS = {
    "StackTwoBlocks": dict(
        perceive=["red pixels → 3-D points", "2-means split: two blocks", "top face → centre, yaw"],
        plan=["cheapest arm/block roles", "A: helper grips the base", "B: relay via mid-table"],
        plan_note="at the start; again after a miss",
        phases=["one machine per arm", "+ the other arm's phase"],
        motion=["saturated P-control", "dp = clip(Kp·e, ±v_max)", "yaw: P-control, capped"],
        sim=["two Panda arms", "+ two red blocks"],
        lane_perc="RGB-D, head + front camera: every step, for each block not in a hand",
        panel_note="Mode A shown (the stacking arm reaches both blocks). Mode B: each arm picks its own block; the "
                   "helper sets its block down mid-table.",
        rows=[dict(label="stacking\narm", start=True, end="success",
                   seq=["approach", "descend", "grasp", ("holding", "g"), "lift", "wait|hold",
                        ("helper clear,\nbase seen again", "a", "eye"), "carry", "lower",
                        ("rests on\nthe base", "h"), "release"],
                   back=[("grasp", "approach", "missed: reopen, look again", "g", "eye")]),
              dict(label="helper\narm", start=True,
                   seq=["approach", "descend", "grasp", ("holding", "g"), "hold",
                        ("stacker holds\nits block 3 steps", "a"), "release", "rise", "home"])],
    ),
    "RotateValve": dict(
        perceive=["valve model (21,911 pts)", "fit to depth: x, y, yaw", "per valve → wheel centre"],
        perceive_note="after the valves settle",
        plan=["left arm: wheel at y > 0", "both reach in the same", "time (minimum jerk)"],
        phases=["one machine per arm,", "both arms at once"],
        motion=["minimum-jerk feed-forward", "+ bounded correction", "while the path is still"],
        sim=["two Panda arms", "+ two valves"],
        lane_perc="RGB-D, head + front camera: every step until the valves stop moving, then the fit, once",
        rows=[dict(label="each arm", start=True, end="success (valve past 27°)",
                   self=[("hold", "not over after 1.2 s: turn 20° more (3x)", "h")],
                   seq=["settle|idle", ("depth still:\nfit valves", "p", "eye"), "reach",
                        ("above wheel,\nfingers open", "h"), "descend", "close",
                        ("rim between pads\n(0.45-0.92)", "g"), "turn", "hold"],
                   back=[("close", "descend", "miss: open, rise, again (2x)", "g")])],
    ),
    "PackBox": dict(
        perceive=["depth 0.70-0.95 m: walls", "min-area rectangle → yaw", "shell extents → centre"],
        perceive_note="once: the box is fixed",
        plan=["push point on each hinge", "parallel or sequenced", "hand roll if reach is far"],
        phases=["one machine per arm,", "each arm its own flap"],
        motion=["feed-forward on flap arc", "+ servo-lag feed-forward", "+ orientation feedback"],
        sim=["two Panda arms", "+ box with two flaps"],
        lane_perc="RGB-D, head + front camera: the first step only",
        lane_pos="joint angles  →  forward kinematics  →  hand poses, lag and tilt, every step",
        rows=[dict(label="first arm", start=True,
                   self=[("push", "aim +2.5° per step; pauses while the hand lags > 8 mm or tilts > 4°", "h")],
                   seq=["transit", "descend", "push", ("path end\n(180.8°)", "h"), "hold",
                        ("sequenced\nmode only", "h"), "release", "away"]),
              dict(label="second arm", start=True, end="success (both flaps closed)",
                   seq=["transit", "descend", ("parallel: at once\nsequenced: first\npush past 130°", "a"),
                        "push", ("path end", "h"), "hold"])],
    ),
    "CubeHandover": dict(
        perceive=["red pixels → 3-D points", "top face → rod centre, axis", "held: the rod's free end"],
        perceive_note="free end gated by kinematics",
        plan=["giver: arm on rod's side", "grasp 6.5 cm from centre", "handover at x = 0.40 m"],
        phases=["giver and receiver", "+ the other arm's phase"],
        motion=["saturated P-control,", "half StackTwoBlocks' speed", "yaw: P-control, capped"],
        sim=["two Panda arms", "+ red rod (3 kg)"],
        lane_perc="RGB-D, head + front camera: at the start and after a miss; the held rod's free end during the handover",
        rows=[dict(label="giver", start=True, end="success",
                   seq=["approach", "descend", "grasp", ("holding\n2 steps", "g"), "lift", "hold",
                        ("receiver holds\n2 steps", "a"), "release"],
                   back=[("grasp", "approach", "missed, blocked or dropped: reopen, look again", "g", "eye")]),
              dict(label="receiver", start=True,
                   seq=["approach|idle", ("rod at the\nhandover pose", "a"), "insert",
                        ("at the seen\nfree end", "p", "eye"), "grasp", ("holding", "g"), "hold"],
                   back=[("insert", "approach", "blocked or missed: back out, retry", "h")])],
    ),
    "LiftTray": dict(
        perceive=["tray mesh, 3 mm voxels", "fit to depth points,", "coarse to fine: x, y, axis"],
        perceive_note="hands masked on a retry",
        plan=["each arm: its end wall", "lift: pull 20 cm toward", "the robot, rise 12 cm"],
        phases=["one machine for both", "arms: they move as one"],
        motion=["saturated P-control,", "the arm with less to go", "slowed: arrive together"],
        sim=["two Panda arms", "+ 34 kg tray"],
        lane_perc="RGB-D, head + front camera: the first step, and again on every retry",
        rows=[dict(label="both arms", start=True, end="success",
                   self=[("lift", "55 steps without success: set down, pull again", "h")],
                   seq=["approach", ("both within\n1 cm, 4°", "h"), "descend", "grasp",
                        ("both holding\n2 steps", "g"), "lift"],
                   back=[("grasp", "approach", "missed: both open, rise, perceive again", "g", "eye"),
                         ("lift", "approach", "a wall slipped: set down, open, perceive again", "g", "eye")])],
    ),
    "PickSingleBookFromTable": dict(
        perceive=["head depth: the top face", "outline template + ribbon", "fit → x, y, yaw"],
        perceive_note="side faces pin the near edge",
        plan=["overhang enough? else push", "pinch: hand tilted 40°", "planning IK checks paths"],
        phases=["pusher, then the", "pinching arm"],
        motion=["minimum-jerk segments,", "sent as feed-forward", "+ bounded correction"],
        sim=["two Panda arms", "+ book on a counter"],
        lane_perc="RGB-D, head camera only: at the start, after the push, on a retry",
        rows=[dict(label="pusher\n(if needed)", start=True,
                   seq=["p_reach", "p_down", "p_push", "p_off", "p_up", "p_clear"]),
              dict(label="pinching\narm", start="no push\nneeded", end="success",
                   seq=["g_reach", "g_insert", "g_close", ("closed on the book\n(opening ≥ 0.12)", "g"), "g_lift",
                        "g_hold"],
                   back=[("g_close", "g_reach", "missed: back off, perceive, re-plan", "g", "eye")])],
        links=[((0, "p_clear"), (1, "g_reach"), "out of view: book seen again, pinch planned", "p")],
    ),
    "StackSingleBookShelf": dict(
        perceive=["head depth: top outline", "template + spine face", "fit → x, y, yaw"],
        perceive_note="the spine edge is pinned",
        plan=["arm on the book's side", "push if overhang < 15 mm", "IK + hand box vs shelf"],
        phases=["one working arm;", "the other waits"],
        motion=["minimum-jerk Bezier paths", "sent as feed-forward", "+ bounded correction"],
        sim=["two Panda arms", "+ book, counter, shelf"],
        lane_perc="RGB-D, head camera only: at the start, after the push, on a retry",
        rows=[dict(label="push\n(if needed)", start=True,
                   self=[("p_push", "seen again: pushed further while the overhang is short", "p")],
                   seq=["p_go", "p_down", "p_push"]),
              dict(label="pinch,\ncarry, place", start="no push\nneeded", end="success",
                   seq=["g_go", "g_insert", "g_close", ("holding", "g"), "c_go", "c_lower",
                        ("descent\nstalls", "h"), "c_release"],
                   back=[("g_close", "g_go", "missed or dropped: open, back off, perceive", "g", "eye")]),
              dict(label="other arm", start=True, seq=["idle"])],
        links=[((0, "p_push"), (1, "g_go"), "overhang ≥ 15 mm (seen)", "p")],
    ),
}


def main():
    figure(dict(COMMON, fwd=("object\npose", "roles,\ntargets", "phase\ntarget", "actions")), CTL / "common_loop.png",
           panel=False)
    for task, D in TASKS.items():
        D = dict(D, title={"PickSingleBookFromTable": "PickBook", "StackSingleBookShelf": "ShelfBook"}.get(task, task),
                 sub=GOAL[task])
        D.setdefault("lane_pos", POSITION)
        D.setdefault("lane_phase", PHASE)
        figure(D, CTL / f"{task}_loop.png")


if __name__ == "__main__":
    main()
