"""Record one controller episode step by step, for the explainer videos (step 1 of 3: trace -> explain -> compose).

The episode runs as in the official evaluator (same episode loop, policy seeding and interface from metadata.yaml, the
evaluator's pinned math settings), with the submitted bundle. Per control step it saves
  * the head camera rendered at --res (a separate MuJoCo renderer: rendering does not change the simulation),
  * the observation's head and front RGB-D images (what the controller sees, 256 x 256),
  * the action, proprioception and gripper openings,
  * a copy of the controller's attributes after act() (its estimates, plan and phase names; introspection only),
  * the controller's planned path ahead of each hand, where it has one (minimum-jerk segments, PackBox's flap paths).
The controller never reads anything recorded here. Training seeds only (>= 10000).

  MUJOCO_GL=egl python -B scripts/explainer/trace.py RotateValve 900000 --out /dev/shm/ctrlvid/RotateValve_900000
"""
import os

# the evaluator's pinned math settings (roboeval.challenge.determinism.PINNED_ENV), set before numpy loads
PINNED = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MKL_CBWR": "COMPATIBLE"}
os.environ.update(PINNED)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from roboeval.challenge import results as res  # noqa: E402
from roboeval.challenge import spec  # noqa: E402
from roboeval.challenge.determinism import PINNED_ENV  # noqa: E402
from roboeval.challenge.evaluate import _seed_policy_rngs  # noqa: E402
from roboeval.challenge.submission import (EpisodeSpec, Interface, load_metadata, load_policy,  # noqa: E402
                                           make_env, task_instruction)

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
PAD_C = 0.1034  # flange -> finger pad centre along the hand's approach axis (Panda hand)


def sanitize(v, depth=0):
    """picklable copy of plain data (numbers, strings, arrays, containers, objects' attributes); no modules or functions"""
    if depth > 4:
        return None
    if isinstance(v, (bool, int, float, str, type(None), np.integer, np.floating, np.bool_)):
        return v
    if isinstance(v, np.ndarray):
        return v.copy() if v.size <= 50000 else ("ndarray", v.shape, str(v.dtype))
    if isinstance(v, dict):
        return {str(k): sanitize(x, depth + 1) for k, x in list(v.items())[:400]}
    if isinstance(v, (list, tuple)):
        return [sanitize(x, depth + 1) for x in list(v)[:2000]]
    if hasattr(v, "__dict__") and not isinstance(v, type) and type(v).__module__ not in ("builtins",):
        if callable(v) and not hasattr(v, "phase"):
            return None
        return {"__class__": type(v).__name__, **{k: sanitize(x, depth + 1) for k, x in vars(v).items()}}
    return None


def _pad(p, r):
    """finger pad centre from a flange pose (orientation matrix, or a yaw angle with the hand pointing down)"""
    r = np.asarray(r)
    if r.shape == (3, 3):
        return np.asarray(p) + PAD_C * r[:, 2]
    return np.asarray(p) - np.array([0.0, 0.0, PAD_C])


def planned_paths(ctl, t):
    """per arm, the pad path the controller has planned from step t to the end of its current segment (or None)"""
    segs = None
    if isinstance(getattr(ctl, "segs", None), list):  # PickBook, ShelfBook
        segs = ctl.segs
    elif isinstance(getattr(ctl, "arms", None), list) and ctl.arms and isinstance(ctl.arms[0], dict):  # RotateValve
        segs = [A.get("seg") for A in ctl.arms]
    out = [None, None]
    if segs:
        for k, s in enumerate(segs[:2]):
            if s is None or not hasattr(s, "at"):
                continue
            e = int(s.end())
            js = np.unique(np.linspace(t, max(t, e), min(60, max(2, e - t + 1))).round().astype(int))
            out[k] = np.array([_pad(*s.at(int(j))) for j in js])
    if isinstance(getattr(ctl, "arms", None), list) and ctl.arms and hasattr(ctl.arms[0], "path"):  # PackBox
        for k, A in enumerate(ctl.arms[:2]):
            if A.path is not None and hasattr(A.path, "point"):
                th = np.radians(np.linspace(np.degrees(A.path.f["th0"]) - 12.0, 180.8, 60))
                out[k] = np.array([A.path.point(x) for x in th])
    return out


def flaps(ctl):
    """PackBox: each flap as the controller models it (hinge line and tip at its current aim angle), 4 corners each"""
    if not (isinstance(getattr(ctl, "arms", None), list) and ctl.arms and hasattr(ctl.arms[0], "path")):
        return None
    out = []
    for A in ctl.arms[:2]:
        P = A.path
        if P is None:
            out.append(None)
            continue
        f = P.f
        th = max(float(A.th), float(f["th0"]))
        X, Z = f["L"] * np.cos(th), f["L"] * np.sin(th)
        pts = [P.c + P.R @ np.array([f["ax"] + f["o"] * x, s, f["az"] + z]) for x, z, s in
               ((0, 0, -0.21), (X, Z, -0.21), (X, Z, 0.21), (0, 0, 0.21))]
        out.append(np.array(pts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("seed", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--submission", default=str(REPO / "submission"))
    ap.add_argument("--res", type=int, default=768)
    a = ap.parse_args()
    assert PINNED == PINNED_ENV, "pinned math settings differ from the evaluator's"
    assert a.seed >= 10000, "training seeds (>= 10000) only"
    out = Path(a.out)
    (out / "hi").mkdir(parents=True, exist_ok=True)
    sub = Path(a.submission)
    md = load_metadata(sub)
    itf = Interface.from_dict(md["interface"])
    env = make_env(a.task, "PositionAndOrientation", itf)
    pol = load_policy(sub, md)
    u = env.unwrapped
    model, data = u._mojo.model, u._mojo.data
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, a.res)
    model.vis.global_.offheight = max(model.vis.global_.offheight, a.res)
    rnd = mujoco.Renderer(model, a.res, a.res)
    cam = u._cameras_map["head"][0]
    max_steps = spec.max_steps(a.task, itf.control_frequency)
    ep = EpisodeSpec(family=a.task, variation="PositionAndOrientation", env_name=type(env).__name__,
                     instruction=task_instruction(a.task), episode_index=0, seed=a.seed,
                     policy_seed=int(md.get("policy_seed", 0)), control_frequency=itf.control_frequency,
                     max_steps=max_steps, action_space=env.action_space)
    rec = {k: [] for k in ("act", "q", "grip", "succ")}
    O = {k: [] for k in ("rgb_head", "depth_head", "rgb_front", "depth_front")}
    snaps, plans, flp = [], [], []

    def grab(t, obs):
        rnd.update_scene(data, camera=cam)
        Image.fromarray(rnd.render()).save(out / "hi" / f"{t:04d}.jpg", quality=93)
        for k in O:
            x = np.asarray(obs[k])
            O[k].append(x.transpose(1, 2, 0).copy() if x.ndim == 3 else x.astype(np.float32).copy())
        rec["q"].append(np.asarray(obs["proprioception"], np.float64).copy())
        rec["grip"].append(np.asarray(obs["proprioception_grippers"], np.float64).copy())

    t0 = time.time()
    obs, info = env.reset(seed=a.seed)
    _seed_policy_rngs(ep.policy_seed, ep.seed)
    pol.reset(ep)
    ctl = pol._cache[a.task]
    grab(0, obs)
    termination = "budget"
    for t in range(max_steps):
        act = np.asarray(pol.act(obs), dtype=np.float32)
        snaps.append(sanitize(vars(ctl)))
        plans.append(planned_paths(ctl, int(getattr(ctl, "t", t + 1))))
        flp.append(flaps(ctl))
        obs, _, terminated, truncated, info = env.step(act)
        rec["act"].append(act)
        rec["succ"].append(bool(env.success))
        grab(t + 1, obs)
        if env.success:
            termination = "success"
            break
        if terminated:
            termination = "fail"
            break
        if truncated:
            termination = "unhealthy"
            break
    steps = len(rec["act"])
    raw = res.to_jsonable(info)
    cams = {n: {"fovy": float(model.cam_fovy[u._cameras_map[n][0]])} for n in ("head", "front")}
    meta = dict(task=a.task, seed=a.seed, termination=termination, steps=steps, max_steps=max_steps, res=a.res,
                metrics=res.episode_metrics(raw, steps, itf.control_frequency), controller=type(ctl).__name__,
                cams=cams, wall_s=round(time.time() - t0, 1))
    np.savez_compressed(out / "trace.npz", act=np.array(rec["act"]), q=np.array(rec["q"]), grip=np.array(rec["grip"]),
                        succ=np.array(rec["succ"]))
    np.savez_compressed(out / "obs.npz", **{k: np.array(v) for k, v in O.items()})
    with open(out / "snaps.pkl", "wb") as f:
        pickle.dump(dict(snaps=snaps, plans=plans, flaps=flp), f)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    print(f"{a.task} {a.seed}: {termination} in {steps} steps ({meta['wall_s']} s)", flush=True)


if __name__ == "__main__":
    main()
