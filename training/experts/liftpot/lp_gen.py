"""Generate synthetic LiftPot demonstrations with the scripted expert (lp_expert.py; 38/40 successes, median 56 steps on
unseen training seeds; the incumbent: ~90%, 70-80 steps). Copy of s2b_gen.py for LiftPot (200-step budget, both arms). Runs the expert in the evaluator's environment with the bundle's interface (five 256x256
cameras), records every step after the start no-op as (observation, expert action) — the same episode format as
replay_saved.py / replay_mirror.py, so to_lerobot.py converts it — and keeps successful episodes only.
--noise P (DART): execute the expert's action plus Gaussian noise on both arms' positions (P m per step) and rotations
(5 P rad per step) during the approach and descent only (the expert corrects position and full orientation there), but
record the clean action.
Usage: python lp_gen.py OUT_DIR FIRST_SEED N --bundle BUNDLE [--workers W] [--noise P]   (TRAINING seeds only)"""
import argparse
import io
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

_W = {}


def _init(bundle):
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from roboeval.challenge.submission import load_interface, make_env
    from lp_expert import Expert

    _W["env"] = make_env("LiftPot", "PositionAndOrientation", load_interface(bundle))
    _W["ex"] = Expert(_W["env"])


def _png(chw):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(np.transpose(chw, (1, 2, 0)))).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def run(job):
    env, ex = _W["env"], _W["ex"]
    obs, _ = env.reset(seed=job["seed"])
    ex.reset()
    rng = np.random.default_rng(job["seed"])
    cams = sorted(k for k in obs if k.startswith("rgb_"))
    frames, term = [], "budget"
    for t in range(200):
        a = ex.act()
        if t > 0:  # skip the start no-op, as collect.py does
            frames.append({"png": {c.removeprefix("rgb_"): _png(obs[c]) for c in cams},
                           "state": np.asarray(obs["proprioception"], np.float32),
                           "gripper": np.asarray(obs["proprioception_grippers"], np.float32),
                           "action": np.asarray(a, np.float32)})
        x = a
        if job["noise"] > 0 and t > 0 and ex.phase in ("approach", "descend"):
            x = a.copy()
            for off in (0, 6):
                x[off:off + 3] += rng.normal(0, job["noise"], 3)
                x[off + 3:off + 6] += rng.normal(0, 5 * job["noise"], 3)
        obs, _, terminated, truncated, _ = env.step(x)
        if env.success:
            term = "success"; break
        if terminated or truncated:
            term = "fail"; break
    out = {"seed": job["seed"], "term": term, "steps": t + 1, "phase": ex.phase, "file": None}
    if term == "success":
        f = Path(job["out_dir"]) / f"ep_expert_{job['seed']}.pkl"
        with open(f.with_suffix(".tmp"), "wb") as fh:
            pickle.dump({"seed": job["seed"], "sigma": job["noise"], "steps": t + 1, "expert": True, "frames": frames}, fh, protocol=5)
        os.replace(f.with_suffix(".tmp"), f)
        out["file"] = str(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("first", type=int); ap.add_argument("n", type=int)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.0)
    a = ap.parse_args()
    assert a.first >= 10_000, "training seeds only"
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    jobs = [dict(seed=s, out_dir=str(out), noise=a.noise) for s in range(a.first, a.first + a.n)]
    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.bundle,))
    log = open(out / "gen_log.jsonl", "a")
    n = kept = 0
    for r in pool.imap_unordered(run, jobs, chunksize=1):
        n += 1; kept += r["file"] is not None
        log.write(json.dumps(r) + "\n"); log.flush()
        if n % 25 == 0 or n == len(jobs):
            print(f"{time.strftime('%H:%M')} {n}/{len(jobs)} generated, {kept} successes kept", flush=True)
    pool.close()
    print("GEN_DONE", n, kept, flush=True)


if __name__ == "__main__":
    main()
