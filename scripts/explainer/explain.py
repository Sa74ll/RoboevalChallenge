"""Recompute each controller's perception step by step from a recorded episode (step 2 of 3: trace -> explain -> compose).

Reads the observation the controller planned from (trace.py's obs.npz) and replays its perception with the controller's
own functions (submission/policy), keeping the intermediate results: the selected pixels and 3-D points, the search
objective over candidate poses, and the final fit. Each final estimate is checked against the one the controller
computed during the episode (snaps.pkl), so the video shows exactly what the controller did.

  python -B scripts/explainer/explain.py /dev/shm/ctrlvid/RotateValve_900000
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "submission" / "policy"))
import hybrid_s2b as hs  # noqa: E402  (camera constants CAMS, forward kinematics)

RNG = np.random.default_rng(0)


def obs_at(O, t):
    """the observation dict the controller saw at act() call t"""
    return {"rgb_head": O["rgb_head"][t].transpose(2, 0, 1), "rgb_front": O["rgb_front"][t].transpose(2, 0, 1),
            "depth_head": O["depth_head"][t], "depth_front": O["depth_front"][t]}


def full_cloud(obs, n_max=200000):
    """every depth pixel of both cameras back-projected (robot frame), with its colour, cropped to the workspace"""
    P, C = [], []
    for n in ("head", "front"):
        c = hs.CAMS[n]
        dep = np.asarray(obs[f"depth_{n}"], np.float64)
        h, w = dep.shape
        ys, xs = np.mgrid[0:h, 0:w]
        f = (h / 2) / np.tan(np.radians(c["fovy"]) / 2)
        X = np.stack([(xs + 0.5 - w / 2) / f * dep, -(ys + 0.5 - h / 2) / f * dep, -dep], -1) @ np.array(c["mat"]).T
        P.append((X + np.array(c["pos"])).reshape(-1, 3))
        C.append(np.asarray(obs[f"rgb_{n}"]).transpose(1, 2, 0).reshape(-1, 3))
    P, C = np.concatenate(P), np.concatenate(C)
    m = (P[:, 0] > -0.1) & (P[:, 0] < 1.25) & (np.abs(P[:, 1]) < 0.95) & (P[:, 2] > 0.55) & (P[:, 2] < 1.6)
    P, C = P[m], C[m]
    i = RNG.choice(len(P), min(n_max, len(P)), replace=False)
    return P[i].astype(np.float32), C[i]


def same(a, b, tol=1e-6):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)


# ---- StackTwoBlocks ------------------------------------------------------------------------------------------------
def explain_s2b(O, S):
    t = 0
    obs = obs_at(O, t)
    masks, pts = {}, []
    for n in hs.CAMS:
        rgb = np.asarray(obs[f"rgb_{n}"]).astype(np.int16)
        masks[n] = (rgb[0] > 90) & (rgb[1] < 70) & (rgb[2] < 70)
        pts.append(hs._points(obs, n))
    P = np.concatenate(pts)
    P = P[(P[:, 2] > 0.9) & (P[:, 2] < 1.4)]
    a = P[np.argmax(np.linalg.norm(P[:, :2] - P[:, :2].mean(0), axis=1))]
    b = P[np.argmax(np.linalg.norm(P[:, :2] - a[:2], axis=1))]
    for _ in range(10):  # the 2-means split of locate()
        lab = np.linalg.norm(P[:, :2] - a[:2], axis=1) > np.linalg.norm(P[:, :2] - b[:2], axis=1)
        a, b = P[~lab].mean(0), P[lab].mean(0)
    blocks = []
    for Q in (P[~lab], P[lab]):
        top = Q[:, 2].max()
        T = Q[Q[:, 2] > top - 0.006]
        c = T[:, :2] - T[:, :2].mean(0)
        ang = np.arange(0, 90, 1.0)
        area = [np.ptp(c @ np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]]), 0).prod() for x in np.radians(ang)]
        blocks.append(dict(Q=Q, T=T, ang=ang, area=np.array(area), est=hs._block(Q)))
    ref = hs.locate(obs)
    for bl, r, s in zip(blocks, ref, S["snaps"][0]["blk"]):
        assert same(bl["est"]["xy"], r["xy"]) and same(bl["est"]["xy"], s["xy"]) and same(bl["est"]["yaw"], s["yaw"])
    s0 = S["snaps"][0]
    return dict(t=t, masks=masks, P=P, lab=lab, blocks=blocks,
                plan=dict(mode=s0["mode"], s=s0["s"], b=s0["b"], P=s0["P"], B=s0["B"], M=s0["M"]))


# ---- RotateValve ---------------------------------------------------------------------------------------------------
def explain_rv(O, S):
    import rv_perception as rvp
    t = next(i for i, s in enumerate(S["snaps"]) if s.get("arms"))
    s0 = S["snaps"][t]
    obs = obs_at(O, t)
    changes = list(s0["changes"])
    moved = any(c > 300 for c in changes[2:])
    model = rvp.ValveModel()
    P = rvp.scene_points(obs)
    view = rvp.DepthView(obs)
    est = rvp.locate_valves(obs, model, z_root=rvp.Z_ROOT_SETTLED, wide=moved)
    for e, r in zip(est, s0["est"]):
        assert same(e["wheel"], r["wheel"]) and same(e["yaw"], r["yaw"])
    heat = []
    for slot in (0, 1):  # the coarse pass of fit_valve (spawn-range search), as a score map over wheel positions
        box = (0.006, 0.006)
        yl, yh = rvp.ROOT_Y[slot] - rvp.ROOT_DY - box[1], rvp.ROOT_Y[slot] + rvp.ROOT_DY + box[1]
        off = np.array([rvp._rot(x) @ rvp.WHEEL_LOCAL[:2] for x in np.linspace(rvp.YAW0 - rvp.DYAW, rvp.YAW0 + rvp.DYAW, 13)])
        wxl, wxh = rvp.ROOT_X - rvp.ROOT_DX - box[0] + off[:, 0].min(), rvp.ROOT_X + rvp.ROOT_DX + box[0] + off[:, 0].max()
        wyl, wyh = yl + off[:, 1].min(), yh + off[:, 1].max()
        Q = P[(P[:, 1] > yl - 0.2) & (P[:, 1] < yh + 0.42)]
        rs = np.random.default_rng(0)
        Qc = Q[rs.choice(len(Q), 1500, replace=False)] if len(Q) > 1500 else Q
        yaws = np.arange(rvp.YAW0 - rvp.DYAW - np.radians(2), rvp.YAW0 + rvp.DYAW + np.radians(2.1), np.radians(3.0))
        wx, wy = np.arange(wxl, wxh + 1e-9, 0.01), np.arange(wyl, wyh + 1e-9, 0.01)
        G = rvp._score_grid(model, Qc, wx, wy, yaws, slot, view, 3.0 * len(Qc) / max(len(Q), 1), rvp.Z_ROOT_SETTLED, box)
        heat.append(dict(wx=wx, wy=wy, S=G.max(0), yaw_best=yaws[np.unravel_index(np.argmax(G), G.shape)[0]]))
    M = np.load(REPO / "submission" / "policy" / "rv_valve_pts.npy").astype(float)
    M = M[np.random.default_rng(0).choice(len(M), 4000, replace=False)]
    posed, inl = [], np.zeros(len(P), bool)
    for e in est:
        R = rvp._rot(e["yaw"])
        root = np.asarray(e["wheel"][:2]) - R @ rvp.WHEEL_LOCAL[:2]
        posed.append(np.c_[M[:, :2] @ R.T + root, M[:, 2] + rvp.Z_ROOT_SETTLED])
        L = np.c_[(P[:, :2] - root) @ R, P[:, 2] - rvp.Z_ROOT_SETTLED]
        inl |= model.count(L[:, None, :]) > 0
    return dict(t=t, changes=changes, moved=moved, P=P, inl=inl, heat=heat, posed=posed,
                est=[dict(wheel=e["wheel"], yaw=e["yaw"], inliers=e["inliers"], n=e["n"]) for e in est])


# ---- PackBox -------------------------------------------------------------------------------------------------------
def explain_pb(O, S):
    import hybrid_pb as pb
    t = 0
    obs = obs_at(O, t)
    P = np.concatenate([pb._cloud(obs, n) for n in ("head", "front")])
    P = P[(P[:, 0] > 0.12) & (P[:, 0] < 1.15) & (np.abs(P[:, 1] + 0.05) < 0.8)]
    wall = (P[:, 2] > pb.BAND_WALL[0]) & (P[:, 2] < pb.BAND_WALL[1])
    flap = (P[:, 2] > pb.BAND_FLAP[0]) & (P[:, 2] < pb.BAND_FLAP[1])
    W = P[wall, :2]
    ang = np.arange(52.0, 128.01, 1.0)
    area = []
    for x in np.radians(ang):
        ul, uh, vl, vh = pb._extent(W, x)
        area.append((uh - ul) * (vh - vl))
    box = pb.locate_box(obs)
    s0 = S["snaps"][0]
    assert same(box["xy"], s0["box"]["xy"]) and same(box["yaw"], s0["box"]["yaw"])
    ul, uh, vl, vh = pb._extent(W, box["yaw"])
    c, s = np.cos(box["yaw"]), np.sin(box["yaw"])
    ex, ey = np.array([c, s]), np.array([-s, c])
    rect = np.array([ex * u + ey * v for u, v in ((ul, vl), (uh, vl), (uh, vh), (ul, vh), (ul, vl))])
    return dict(t=t, P=P, wall=wall, flap=flap, ang=ang, area=np.array(area), box=box, rect=rect,
                plan=dict(mode=s0["mode"], first=s0["first"], reach=s0["reach"],
                          s=[A["path"]["s"] for A in s0["arms"]], paths=S["plans"][0]))


# ---- CubeHandover --------------------------------------------------------------------------------------------------
def explain_ch(O, S):
    import rod_perc as rp
    t = 0
    obs = obs_at(O, t)
    masks = {}
    for n in hs.CAMS:
        rgb = np.asarray(obs[f"rgb_{n}"]).astype(np.int16)
        masks[n] = (rgb[0] > 90) & (rgb[1] < 70) & (rgb[2] < 70)
    P = rp.rod_points(obs)
    top = rp._top(P[:, 2])
    T = P[(P[:, 2] > top - 0.006) & (P[:, 2] < top + 0.003)]
    rod = rp.locate_rod(obs, P)
    s0 = S["snaps"][0]
    assert same(rod["c"][:2], s0["rod"]["c"][:2], 1e-3) and same(rod["u"], s0["rod"]["u"], 1e-3)
    u = rod["u"]
    n = np.array([-u[1], u[0]])
    a, b = T[:, :2] @ u, T[:, :2] @ n
    a0, a1 = np.percentile(a, [0.5, 99.5])
    b0, b1 = np.percentile(b, [0.5, 99.5])
    rect = np.array([u * x + n * y for x, y in ((a0, b0), (a1, b0), (a1, b1), (a0, b1), (a0, b0))])
    hz = np.histogram(P[:, 2], bins=np.arange(0.93, 1.03, 0.002))
    return dict(t=t, masks=masks, P=P, T=T, top=top, rod=rod, rect=rect, hist=hz,
                plan=dict(G=s0["G"], R=s0["R"], rod=s0["rod"]))


# ---- LiftTray ------------------------------------------------------------------------------------------------------
def explain_lt(O, S):
    import lt_perc as lp
    t = 0
    obs = obs_at(O, t)
    occ_c, occ_m, occ_f = lp._model()
    P0 = lp.cloud(obs)
    hz, be = np.histogram(P0[:, 2], bins=np.arange(lp.Z_TABLE, 1.06, 0.002))
    zf = be[int(np.argmax(hz))] + 0.001
    z0 = zf - lp.FLOOR
    P = P0[(P0[:, 2] > z0 + 0.012) & (P0[:, 2] < z0 + 0.105)]
    rng = np.random.default_rng(0)
    Ps = P[rng.choice(len(P), min(len(P), 1500), replace=False)]
    cx0, cy0 = (float(v) for v in np.median(P[:, :2], 0))
    ths = np.pi / 2 + np.radians(3) * np.arange(-30, 31)  # the coarse pass of locate_tray
    offs = 0.015 * np.arange(-6, 7)
    X, Y = np.meshgrid(cx0 + offs, cy0 + offs, indexing="ij")
    G = np.array([lp._score(Ps, X.ravel(), Y.ravel(), np.full(X.size, th), z0, occ_c) for th in ths]).reshape(len(ths), *X.shape)
    est = lp.locate_tray(obs)
    s0 = S["snaps"][0]
    assert same(est["c"], s0["est"][0]["c"]) and same(est["th"], s0["est"][0]["th"])
    return dict(t=t, P0=P0, P=P, hist=(hz, be), z0=z0, zf=zf, ths=ths, offs=offs, c0=(cx0, cy0), G=G, est=est,
                tray=s0["tray"])


# ---- the two book tasks --------------------------------------------------------------------------------------------
def explain_book(O, S, mod_name):
    mod = __import__(mod_name)
    t = 0
    obs = obs_at(O, t)
    P = mod.cloud(obs, "head")
    R = mod.REGION
    reg = (P[:, 0] > R["x"][0]) & (P[:, 0] < R["x"][1]) & (np.abs(P[:, 1]) < R["y"])
    band = reg & (P[:, 2] > mod.Z_COUNTER + 0.012) & (P[:, 2] < mod.Z_COUNTER + 0.045)
    hz = P[band, 2]
    hist = np.histogram(hz, bins=np.arange(mod.Z_COUNTER + 0.012, mod.Z_COUNTER + 0.046, 0.002))
    est = mod.locate_book(obs)
    s0 = S["snaps"][0]
    assert est["ok"] and same(est["xy"], s0["book"]["xy"]) and same(est["yaw"], s0["book"]["yaw"])
    h_top = est["h_top"]
    top = band & (np.abs(P[:, 2] - h_top) < 0.003)
    side = reg & (P[:, 2] > h_top - 0.021) & (P[:, 2] < h_top - 0.007)
    sc = mod._Scorer(P[top], P[side], np.asarray(obs["depth_head"], np.float64), h_top)
    parts = sc(np.r_[est["xy"], est["yaw"]])  # (score, top points inside, seen-through template points, side points on
    # the edge, side points deep inside)
    bx, by = (mod.BOOK_X, mod.BOOK_Y) if hasattr(mod, "BOOK_X") else (mod.TOP_X, mod.TOP_Y)
    outline = dict(block=(bx, by), ribbon=(mod.RIB_X, mod.RIB_Y))
    return dict(t=t, P=P, reg=reg, band=band, top=top, side=side, hist=hist, est=est, parts=parts, outline=outline,
                plan=s0.get("plan"), push=s0.get("push"), events=S["snaps"][-1].get("events"))


EXPLAIN = {"StackTwoBlocks": explain_s2b, "RotateValve": explain_rv, "PackBox": explain_pb, "CubeHandover": explain_ch,
           "LiftTray": explain_lt, "PickSingleBookFromTable": lambda O, S: explain_book(O, S, "pk_perception"),
           "StackSingleBookShelf": lambda O, S: explain_book(O, S, "ss_perc")}


def main():
    d = Path(sys.argv[1])
    meta = json.loads((d / "meta.json").read_text())
    O = dict(np.load(d / "obs.npz"))
    with open(d / "snaps.pkl", "rb") as f:
        S = pickle.load(f)
    ex = EXPLAIN[meta["task"]](O, S)
    ex["cloud"] = full_cloud(obs_at(O, ex["t"]))
    with open(d / "explain.pkl", "wb") as f:
        pickle.dump(ex, f)
    print(f"{meta['task']}: perception at step {ex['t']}, estimate matches the controller's", flush=True)


if __name__ == "__main__":
    main()
