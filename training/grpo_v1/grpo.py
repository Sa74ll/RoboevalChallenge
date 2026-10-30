"""GRPO fine-tuning of one ACT task model in the RoboEval simulator (driver; RoboEval env python).

Each iteration: S fresh training seeds x G noisy rollouts per seed (same initial state, different
exploration noise) -> reward per episode -> advantage = (R - group mean) / group std -> one PPO-clip
update in the learner process (CUDA torch, see learner.py). Every `dev_every` iterations the
deterministic policy (no noise) is scored on fixed dev seeds and saved when it is the best so far.

Seeds: training seeds are drawn from [10000, 1e6); dev seeds are 3000+. Neither overlaps the official
seeds (1000-1049, never used) or our validation seeds (2000-2049, kept for the final comparison).
The reward reads the env's own success and subtask_progress outputs; the benchmark is not modified.

Usage: python grpo.py --ckpt CKPT_DIR --bundle BUNDLE_DIR --family StackTwoBlocks --out OUT_DIR
"""
import argparse
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout  # noqa: E402


# Shaping from the env's own end-state distances: (key, weight); the reward grows with the fraction
# by which the distance shrank from the start of the episode. Only where "smaller" means closer to done.
SHAPING = {"StackTwoBlocks": [("inter-block distance", 0.2)]}


def reward(ep, family):
    """Benchmark task score of this episode + extras (see the module notes in PROGRESS.md 4.10).

    1. The official per-task score (roboeval.challenge.scoring.task_scores) on this episode's 14
       metrics against the human reference: outcome (0.7 success + 0.3 progress), safety, efficiency
       (incl. completion time) and coordination. The leaderboard applies it to the 50-episode means.
    2. +0.5 on success, so a smoother failure never outranks a success.
    3. Up to +0.2 for closing the task's key distance, a finer signal than the 3-step progress.
    4. -0.25 when the simulator went unstable.
    """
    from roboeval.challenge.scoring import task_scores

    r = task_scores(family, ep["metrics"])["score"] + 0.5 * ep["success"]
    for key, w in SHAPING.get(family, []):
        d0, d1 = ep["dist0"].get(key), ep["dist_end"].get(key)
        if d0 and d1 is not None:
            r += w * float(np.clip((d0 - d1) / d0, 0.0, 1.0))
    if ep["term"] == "unhealthy":
        r -= 0.25
    return r


def leaderboard_score(eps, family):
    """The leaderboard's task score for this set of episodes (task_scores on the metric means)."""
    from roboeval.challenge.scoring import task_scores

    keys = eps[0]["metrics"].keys()
    return float(task_scores(family, {k: np.mean([e["metrics"][k] for e in eps]) for k in keys})["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bundle", required=True, help="bundle dir whose metadata.yaml gives the interface")
    ap.add_argument("--family", required=True)
    ap.add_argument("--variation", default="PositionAndOrientation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--seeds-per-iter", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--n-exec", type=int, default=8)
    ap.add_argument("--noop", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dev-seeds", type=int, default=32)
    ap.add_argument("--dev-every", type=int, default=5)
    ap.add_argument("--learner-python", required=True)
    ap.add_argument("--shm", default="/dev/shm/grpo")
    ap.add_argument("--init-weights", default=None)
    ap.add_argument("--start-iter", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=2)
    a = ap.parse_args()

    out, shm = Path(a.out), Path(a.shm)
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(shm, ignore_errors=True)
    shm.mkdir(parents=True)
    log = open(out / "log.jsonl", "a")
    here = Path(__file__).resolve().parent

    cmd = [a.learner_python, str(here / "learner.py"), "--ckpt", a.ckpt, "--shm", str(shm),
           "--sigma", str(a.sigma), "--lr", str(a.lr), "--beta", str(a.beta), "--epochs", str(a.epochs)]
    if a.init_weights:
        cmd += ["--init-weights", a.init_weights]
    learner = subprocess.Popen(cmd, cwd=here, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def ask(line=None):
        if line:
            learner.stdin.write(line + "\n")
            learner.stdin.flush()
        while True:
            resp = learner.stdout.readline()
            if not resp:
                raise RuntimeError("learner exited")
            if resp.startswith("{"):
                return json.loads(resp)

    cur = ask()
    print(f"learner ready: {cur}", flush=True)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(a.workers, initializer=rollout.init_worker,
                    initargs=(a.ckpt, a.bundle, a.family, a.variation, a.threads))
    from roboeval.challenge import spec
    from roboeval.challenge.submission import load_interface

    max_steps = spec.max_steps(a.family, load_interface(a.bundle).control_frequency)
    rng = np.random.default_rng(12345 + a.start_iter)
    dev_seeds = list(range(3000, 3000 + a.dev_seeds))
    best = -1.0

    def dev_eval(it):
        nonlocal best
        jobs = [dict(seed=s, g=0, noise_seed=0, sigma=0.0, n_exec=a.n_exec, noop=a.noop,
                     weights=cur["weights"], ver=cur["ver"], out=None) for s in dev_seeds]
        eps = pool.map(rollout.run_episode, jobs, chunksize=1)
        succ = float(np.mean([e["success"] for e in eps]))
        rds = leaderboard_score(eps, a.family)
        rec = {"dev": True, "iter": it, "ver": cur["ver"], "rds": rds, "success": succ,
               "tp": float(np.mean([e["tp"] for e in eps])),
               "steps": float(np.mean([e["steps"] for e in eps])),
               "reward": float(np.mean([reward(e, a.family) for e in eps])),
               "errors": sum(e["term"] == "error" for e in eps), "time": time.strftime("%m-%d %H:%M")}
        if rds > best:  # select on the leaderboard formula, not success alone
            best = rds
            ask(f"save {out / 'best'}")
            rec["saved_best"] = True
        ask(f"save {out / 'latest'}")
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"DEV it {it}: rds {rds:.3f} success {succ:.3f} tp {rec['tp']:.3f} steps {rec['steps']:.0f} "
              f"reward {rec['reward']:.3f} {'(best)' if rec.get('saved_best') else ''}", flush=True)

    dev_eval(a.start_iter)
    for it in range(a.start_iter, a.iters):
        t0 = time.time()
        itdir = shm / f"it{it}"
        itdir.mkdir()
        seeds = rng.integers(10_000, 1_000_000, a.seeds_per_iter)
        jobs = [dict(seed=int(s), g=g, noise_seed=int(rng.integers(2**31)), sigma=a.sigma,
                     n_exec=a.n_exec, noop=a.noop, weights=cur["weights"], ver=cur["ver"],
                     out=str(itdir / f"ep_{s}_{g}.npz"))
                for s in seeds for g in range(a.group)]
        eps = list(pool.imap_unordered(rollout.run_episode, jobs, chunksize=1))
        t_roll = time.time() - t0
        groups = {}
        for e in eps:
            e["R"] = reward(e, a.family)
            groups.setdefault(e["seed"], []).append(e)
        live = 0
        for g in groups.values():
            rs = np.array([e["R"] for e in g if e["term"] != "error"])
            sd = rs.std() if len(rs) > 1 else 0.0
            for e in g:
                e["adv"] = 0.0 if (e["term"] == "error" or sd < 1e-6) else float((e["R"] - rs.mean()) / (sd + 1e-6))
            live += sd >= 1e-6
        (itdir / "summary.json").write_text(json.dumps(eps))
        cur = ask(f"update {itdir}")
        shutil.rmtree(itdir, ignore_errors=True)
        rec = {"iter": it, "ver": cur["ver"], "rds": leaderboard_score(eps, a.family), "success": float(np.mean([e["success"] for e in eps])),
               "tp": float(np.mean([e["tp"] for e in eps])), "steps": float(np.mean([e["steps"] for e in eps])),
               "reward": float(np.mean([e["R"] for e in eps])), "live_groups": int(live),
               "errors": sum(e["term"] == "error" for e in eps), "t_roll": round(t_roll),
               "t_total": round(time.time() - t0), **{k: v for k, v in cur.items() if k not in ("ok", "weights")}}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"it {it}: rds {rec['rds']:.3f} succ {rec['success']:.3f} tp {rec['tp']:.3f} steps {rec['steps']:.0f} R {rec['reward']:.3f} "
              f"live {live}/{len(groups)} samples {cur.get('samples')} kl {cur.get('kl', 0):.3f} "
              f"ratio {cur.get('ratio', 1):.3f} clip {cur.get('clipfrac', 0):.3f} ({rec['t_total']}s)", flush=True)
        if (it + 1) % a.dev_every == 0:
            dev_eval(it + 1)
    learner.stdin.write("quit\n")
    learner.stdin.close()
    learner.wait()
    pool.close()
    print("GRPO_DONE", flush=True)


if __name__ == "__main__":
    main()
