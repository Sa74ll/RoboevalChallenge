"""GRPO v2 driver (Oct 22): grpo.py plus opt-in options from recent GRPO variants; defaults reproduce
grpo.py exactly (it then runs learner2.py, whose defaults reproduce learner.py).
  --adv group-std | batch-std | mean
                   group-std: (R - group mean) / group std (GRPO, as before). mean: R - group mean
                   (Dr. GRPO: no per-group std, which inflates noise in groups whose rewards barely
                   differ, e.g. all successes). batch-std: R - group mean, scaled by one std over the
                   whole iteration (removes the per-group bias, keeps the step size comparable).
  --dyn-sample     DAPO / SimpleVLA-RL dynamic sampling: only groups with mixed outcomes
                   (0 < successes < group) train; --dyn-oversample F draws F x seeds-per-iter initial
                   states and keeps at most seeds-per-iter mixed groups (in draw order).
  --time-penalty W, --time-window F
                   DAPO-style soft length penalty: -W * clip((steps - (1-F) max_steps) / (F max_steps), 0, 1),
                   so episodes that drift towards the step budget (StackTwoBlocks' main failure) score lower.
  --clip-high, --seq-ratio, --seq-clip-low, --seq-clip-high, --ep-per-mb: passed to learner2.py
                   (DAPO clip-higher; GSPO sequence-level ratio).

GRPO fine-tuning of one ACT task model in the RoboEval simulator (driver; RoboEval env python).

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


def reward(ep, family, max_steps=None, time_penalty=0.0, time_window=0.25, shaping=True, subgoal=None):
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
    for key, w in (SHAPING.get(family, []) if shaping else []):
        d0, d1 = ep["dist0"].get(key), ep["dist_end"].get(key)
        if d0 and d1 is not None:
            r += w * float(np.clip((d0 - d1) / d0, 0.0, 1.0))
    if subgoal and ep.get("sg_off") is not None:  # --subgoal-shaping (StackTwoBlocks): centred first grasp, adequate lift
        w_off, w_lift = subgoal
        r += w_off * float(np.clip(1.0 - ep["sg_off"] / 0.03, 0.0, 1.0)) + w_lift * float(np.clip(ep["sg_lift"] / 0.06, 0.0, 1.0))
    if ep["term"] == "unhealthy":
        r -= 0.25
    if time_penalty and max_steps:
        start = (1.0 - time_window) * max_steps
        r -= time_penalty * float(np.clip((ep["steps"] - start) / (time_window * max_steps), 0.0, 1.0))
    return r


def leaderboard_score(eps, family):
    """The leaderboard's task score for this set of episodes (task_scores on the metric means)."""
    from roboeval.challenge.scoring import task_scores

    keys = eps[0]["metrics"].keys()
    return float(task_scores(family, {k: np.mean([e["metrics"][k] for e in eps]) for k in keys})["score"])


def save_rollouts(eps, sdir, it, ver):
    """Seed + executed actions of these episodes, one .npz (actions, offsets) + .json per round; with pinned BLAS
    env.reset(seed) and these actions reproduce each episode exactly (camera frames can be re-rendered)."""
    if not eps:
        return
    sdir.mkdir(exist_ok=True)
    lens = [len(e["actions"]) for e in eps]
    np.savez_compressed(sdir / f"it{it}.npz", actions=np.concatenate([e["actions"] for e in eps]),
                        offsets=np.cumsum([0] + lens))
    meta = [{"iter": it, "ver": ver, "seed": e["seed"], "gkey": e["gkey"], "g": e["g"], "prefix_t": e["prefix_t"],
             "term": e["term"], "steps": e["steps"], "success": e["success"], "tp": e["tp"], "R": e["R"], "n_actions": n,
             "sg_off": e.get("sg_off"), "sg_lift": e.get("sg_lift")}
            for e, n in zip(eps, lens)]
    (sdir / f"it{it}.json").write_text(json.dumps(meta))


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
    ap.add_argument("--adv", choices=["group-std", "batch-std", "mean"], default="group-std")
    ap.add_argument("--branch-bank", default=None, help="JSON list of {npz, seed, t, kind}: logged branch points")
    ap.add_argument("--branch-frac", type=float, default=0.0, help="fraction of each round's starts drawn from the bank")
    ap.add_argument("--no-shaping", action="store_true", help="drop the distance-closing reward term (report: it may pay for low carries)")
    ap.add_argument("--subgoal-shaping", default=None, metavar="W_OFF,W_LIFT",
                    help="StackTwoBlocks: + W_OFF * (1 - offset/3 cm) for the policy's first grasp and + W_LIFT * (lift/6 cm) for "
                         "that block's highest lift while held (both clipped to [0, 1]); Oct 24, from the grasp-offset analysis")
    ap.add_argument("--noise-hold", action="store_true", help="one exploration noise vector per executed chunk instead of per step")
    ap.add_argument("--critic", type=float, default=None, metavar="LAMBDA",
                    help="StackTwoBlocks: per-chunk advantages from a privileged-state critic, GAE(LAMBDA), values shifted per "
                         "group to its mean return (critic.py); kept groups as without it; Oct 24, research report")
    ap.add_argument("--critic-rounds", type=int, default=8, help="rounds of chunk states the critic is fit on")
    ap.add_argument("--critic-warmup", type=int, default=2, help="fits before its advantages are used (episode advantages until then)")
    ap.add_argument("--action-scale", type=float, default=1.0,
                    help="multiply the arms' pose deltas (not the grippers) at execution, in training and dev rollouts, as the "
                         "bundle's inference.json action_scale (ShelfBook is evaluated at 0.85; Oct 24)")
    ap.add_argument("--save-successes", action="store_true",
                    help="keep every successful rollout (seed + executed actions, replayable exactly) in OUT/successes/")
    ap.add_argument("--save-failures", action="store_true", help="keep the other rollouts the same way in OUT/failures/")
    ap.add_argument("--dyn-sample", action="store_true")
    ap.add_argument("--dyn-oversample", type=float, default=1.5)
    ap.add_argument("--time-penalty", type=float, default=0.0)
    ap.add_argument("--time-window", type=float, default=0.25)
    ap.add_argument("--clip-high", type=float, default=None)
    ap.add_argument("--seq-ratio", action="store_true")
    ap.add_argument("--seq-clip-low", type=float, default=0.03)
    ap.add_argument("--seq-clip-high", type=float, default=0.04)
    ap.add_argument("--ep-per-mb", type=int, default=8)
    a = ap.parse_args()

    out, shm = Path(a.out), Path(a.shm)
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(shm, ignore_errors=True)
    shm.mkdir(parents=True)
    log = open(out / "log.jsonl", "a")
    here = Path(__file__).resolve().parent

    cmd = [a.learner_python, str(here / "learner2.py"), "--ckpt", a.ckpt, "--shm", str(shm),
           "--sigma", str(a.sigma), "--lr", str(a.lr), "--beta", str(a.beta), "--epochs", str(a.epochs)]
    if a.clip_high is not None:
        cmd += ["--clip-high", str(a.clip_high)]
    if a.seq_ratio:
        cmd += ["--seq-ratio", "--seq-clip-low", str(a.seq_clip_low), "--seq-clip-high", str(a.seq_clip_high),
                "--ep-per-mb", str(a.ep_per_mb)]
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
    subgoal = tuple(float(x) for x in a.subgoal_shaping.split(",")) if a.subgoal_shaping else None
    R = lambda e: reward(e, a.family, max_steps, a.time_penalty, a.time_window, not a.no_shaping, subgoal)  # noqa: E731
    print(f"v2 options: adv={a.adv} dyn_sample={a.dyn_sample} (x{a.dyn_oversample}) time_penalty={a.time_penalty}"
          f"/{a.time_window} clip_high={a.clip_high} seq_ratio={a.seq_ratio} ({a.seq_clip_low}/{a.seq_clip_high})",
          flush=True)
    dev_seeds = list(range(3000, 3000 + a.dev_seeds))
    best = -1.0
    bank = json.loads(Path(a.branch_bank).read_text()) if a.branch_bank else []
    if bank:
        print(f"branch bank: {len(bank)} logged branch points, {a.branch_frac:.0%} of each round's starts", flush=True)
    print(f"reward shaping: {not a.no_shaping}; noise held per chunk: {a.noise_hold}; subgoal shaping: {subgoal}; "
          f"action scale: {a.action_scale}; critic: {a.critic} ({a.critic_rounds} rounds, warm-up {a.critic_warmup})", flush=True)
    crit = None
    if a.critic is not None:
        import torch

        from critic import Critic, gae
        torch.set_num_threads(4)

    def dev_eval(it):
        nonlocal best
        jobs = [dict(seed=s, g=0, noise_seed=0, sigma=0.0, n_exec=a.n_exec, noop=a.noop, action_scale=a.action_scale,
                     weights=cur["weights"], ver=cur["ver"], out=None) for s in dev_seeds]
        eps = pool.map(rollout.run_episode, jobs, chunksize=1)
        succ = float(np.mean([e["success"] for e in eps]))
        rds = leaderboard_score(eps, a.family)
        rec = {"dev": True, "iter": it, "ver": cur["ver"], "rds": rds, "success": succ,
               "tp": float(np.mean([e["tp"] for e in eps])),
               "steps": float(np.mean([e["steps"] for e in eps])),
               "reward": float(np.mean([R(e) for e in eps])),
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
        n_seeds = int(np.ceil(a.seeds_per_iter * a.dyn_oversample)) if a.dyn_sample else a.seeds_per_iter
        n_bank = int(round(a.branch_frac * n_seeds)) if bank else 0
        starts = [dict(gkey=int(s), seed=int(s)) for s in rng.integers(10_000, 1_000_000, n_seeds - n_bank)]
        for i in (rng.choice(len(bank), n_bank, replace=False) if n_bank else []):
            b = bank[int(i)]
            starts.append(dict(gkey=f"b{int(i)}", seed=int(b["seed"]), prefix_npz=b["npz"], prefix_t=int(b["t"])))
        starts = [starts[i] for i in rng.permutation(len(starts))]  # dynamic sampling keeps groups in draw order
        jobs = [dict(seed=st["seed"], gkey=st["gkey"], prefix_npz=st.get("prefix_npz"), prefix_t=st.get("prefix_t", 0),
                     g=g, noise_seed=int(rng.integers(2**31)), sigma=a.sigma, noise_hold=a.noise_hold, subgoal=bool(subgoal),
                     priv=a.critic is not None,
                     n_exec=a.n_exec, noop=a.noop, action_scale=a.action_scale, weights=cur["weights"], ver=cur["ver"],
                     out=str(itdir / f"ep_{st['gkey']}_{g}.npz"))
                for st in starts for g in range(a.group)]
        eps = list(pool.imap_unordered(rollout.run_episode, jobs, chunksize=1))
        t_roll = time.time() - t0
        groups = {}
        for e in eps:
            e["R"] = R(e)
            groups.setdefault(e["gkey"], []).append(e)
        played = [e for e in eps if e.get("actions") is not None]
        if a.save_successes:  # Oct 23 (user): save GRPO's successful episodes for later training
            save_rollouts([e for e in played if e["term"] == "success"], out / "successes", it, cur["ver"])
        if a.save_failures:  # the rest too: a value model for chunk selection needs failures as well
            save_rollouts([e for e in played if e["term"] != "success"], out / "failures", it, cur["ver"])
        for e in eps:
            e.pop("actions", None)  # not JSON-serializable; the learner does not need them
        # draw order of the starts, so dynamic sampling keeps the first mixed groups, not the fastest ones
        order = [st["gkey"] for st in starts]
        mixed, kept = 0, []
        for sd_ in order:
            g = [e for e in groups.get(sd_, []) if e["term"] != "error"]
            ns = sum(e["success"] for e in g)
            is_mixed = 0 < ns < len(g)
            mixed += is_mixed
            if a.dyn_sample and (not is_mixed or len(kept) >= a.seeds_per_iter):
                continue
            kept.append(sd_)
        for e in eps:
            e["adv"] = 0.0
        devs = {}
        for sd_ in kept:
            g = groups[sd_]
            rs = np.array([e["R"] for e in g if e["term"] != "error"])
            if len(rs) < 2 or rs.std() < 1e-6:
                continue
            for e in g:
                if e["term"] != "error":
                    devs[id(e)] = (e, e["R"] - rs.mean(), rs.std())
        if devs:
            bstd = float(np.std([d for _, d, _ in devs.values()])) if a.adv == "batch-std" else 1.0
            for e, d, gsd in devs.values():
                e["adv"] = float(d / (gsd + 1e-6)) if a.adv == "group-std" else float(d / (bstd + 1e-6) if a.adv == "batch-std" else d)
        live = len({e["gkey"] for e, _, _ in devs.values()})
        live_bank = len({e["gkey"] for e, _, _ in devs.values() if str(e["gkey"]).startswith("b")})
        crit_rec = {}
        withp = [e for e in eps if e.get("priv") is not None and e["term"] != "error"]
        if a.critic is not None and withp:
            # --critic: the same episodes train, each chunk with its own advantage (critic.py)
            X = np.concatenate([e["priv"] for e in withp])
            y = np.concatenate([np.full(len(e["priv"]), e["R"]) for e in withp])
            gm = {k: np.mean([e["R"] for e in withp if e["gkey"] == k]) for k in {e["gkey"] for e in withp}}
            yg = np.concatenate([np.full(len(e["priv"]), gm[e["gkey"]]) for e in withp])
            crit_rec["crit_ev_group"] = float(1.0 - np.var(y - yg) / (np.var(y) + 1e-9))  # what a per-start baseline explains
            if crit is None:
                crit = Critic(X.shape[1], rounds=a.critic_rounds)
            if crit.fits:  # out of sample: this round is not in the buffer yet
                crit_rec["crit_ev"] = float(1.0 - np.var(y - crit.predict(X)) / (np.var(y) + 1e-9))
            if crit.fits >= a.critic_warmup:
                by_g = {}
                for e, _, gsd in devs.values():
                    if e.get("priv") is not None and e["file"]:
                        by_g.setdefault(e["gkey"], []).append((e, gsd))
                chunks = []
                for g in by_g.values():
                    vs = [crit.predict(e["priv"]) for e, _ in g]
                    off = np.mean([e["R"] for e, _ in g]) - np.mean([v[0] for v in vs])
                    for (e, gsd), v in zip(g, vs):
                        e["adv_chunks"] = gae(v + off, e["R"], a.critic)
                        if a.adv == "group-std":
                            e["adv_chunks"] = e["adv_chunks"] / (gsd + 1e-6)
                        chunks.append(e["adv_chunks"])
                if chunks:
                    sd = float(np.std(np.concatenate(chunks)))
                    for g in by_g.values():
                        for e, _ in g:
                            e["adv_chunks"] = [float(x / (sd + 1e-6) if a.adv == "batch-std" else x) for x in e["adv_chunks"]]
                    crit_rec.update(crit_used=True, crit_adv_std=sd, crit_eps=len(chunks))
            crit_rec.update(crit_loss=crit.add_and_fit(X, y), crit_rows=sum(len(b[1]) for b in crit.buf))
        for e in eps:
            e.pop("priv", None)
        (itdir / "summary.json").write_text(json.dumps(eps))
        cur = ask(f"update {itdir}")
        shutil.rmtree(itdir, ignore_errors=True)
        grasped = [e for e in eps if e.get("sg_off") is not None]  # --subgoal-shaping: episodes with a policy grasp
        sg_rec = {"sg_n": len(grasped), "sg_off": float(np.mean([e["sg_off"] for e in grasped])) if grasped else None,
                  "sg_lift": float(np.mean([e["sg_lift"] for e in grasped])) if grasped else None} if subgoal else {}
        rec = {"iter": it, "ver": cur["ver"], "rds": leaderboard_score(eps, a.family), "success": float(np.mean([e["success"] for e in eps])),
               "tp": float(np.mean([e["tp"] for e in eps])), "steps": float(np.mean([e["steps"] for e in eps])),
               "reward": float(np.mean([e["R"] for e in eps])), "live_groups": int(live), "live_bank": int(live_bank),
               "bank_starts": int(n_bank), **sg_rec, **crit_rec,
               "mixed_groups": int(mixed), "seeds_drawn": len(order),
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
