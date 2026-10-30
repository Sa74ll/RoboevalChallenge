"""Simulator rollouts for GRPO: one env + CPU policy per pool worker (RoboEval env, CPU torch).

An episode mirrors the evaluator's loop (roboeval/challenge/evaluate.py run_episode) with the
submission's execution settings: `noop` zero actions with open grippers first (start fix), then a
fresh chunk every `n_exec` steps. With sigma > 0, Gaussian noise is added to the executed chunk in
normalized action units and the (observation, noisy normalized chunk) pairs are saved for the update.
"""
import os
import time
import traceback

import numpy as np

_W = {}


def init_worker(ckpt_dir, bundle_dir, family, variation, threads):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch

    torch.set_num_threads(threads)
    from roboeval.challenge import spec
    from roboeval.challenge.submission import load_interface, make_env

    from act_common import ACTNet

    interface = load_interface(bundle_dir)
    _W.update(
        make=lambda: make_env(family, variation, interface),
        max_steps=spec.max_steps(family, interface.control_frequency),
        freq=interface.control_frequency,
        net=ACTNet(ckpt_dir, "cpu"),
        ver=None,
    )
    _W["env"] = _W["make"]()


def run_episode(job):
    import torch

    net, env, max_steps = _W["net"], _W["env"], _W["max_steps"]
    if job["ver"] != _W["ver"]:
        net.load_weights(job["weights"])
        _W["ver"] = job["ver"]
    rng = np.random.default_rng(job["noise_seed"])
    sigma, n_exec = job["sigma"], job["n_exec"]
    rec = {"imgs": [], "state": [], "act": [], "nexec": []}
    steps, term, err, info, dist0 = 0, "budget", None, {}, {}
    start = time.time()
    try:
        obs, info = env.reset(seed=job["seed"])
        done = False

        def step(a):
            nonlocal obs, info, steps, term, done
            obs, _, terminated, truncated, info = env.step(np.asarray(a, dtype=np.float32))
            steps += 1
            if env.success:
                term, done = "success", True
            elif terminated:
                term, done = "fail", True
            elif truncated:
                term, done = "unhealthy", True

        for _ in range(job["noop"]):
            if not done and steps < max_steps:
                step(np.zeros(net.action_dim, dtype=np.float32))  # 0 = grippers open
        dist0 = dict(info.get("target_distance") or {})  # start-of-episode distances, for shaping
        while not done and steps < max_steps:
            imgs, state = net.obs_arrays(obs)
            with torch.no_grad():
                mu = net.mu(torch.from_numpy(imgs)[None], torch.from_numpy(state)[None])[0, :n_exec].numpy()
            a_n = mu + rng.normal(0.0, sigma, mu.shape).astype(np.float32) if sigma > 0 else mu
            a = net.unnorm_action(torch.from_numpy(a_n)).numpy()
            k = 0
            while k < n_exec and steps < max_steps and not done:
                step(a[k])
                k += 1
            if sigma > 0:
                rec["imgs"].append(imgs)
                rec["state"].append(state)
                rec["act"].append(a_n.astype(np.float32))
                rec["nexec"].append(k)
    except Exception:
        term, err = "error", traceback.format_exc(limit=6)
        _W["env"] = _W["make"]()  # a broken simulator must not poison the next episode
    from roboeval.challenge import results as res

    raw = res.to_jsonable(info)
    if term == "error":
        raw["success"] = 0.0
    out = {
        "seed": job["seed"], "g": job["g"], "term": term, "error": err, "steps": steps,
        "success": float(term == "success"), "tp": float(info.get("subtask_progress", 0.0) or 0.0),
        # the 14 leaderboard metrics of this episode, exactly as the evaluator derives them
        "metrics": res.episode_metrics(raw, steps, _W["freq"]),
        "dist0": dist0, "dist_end": dict(info.get("target_distance") or {}),
        "wall": time.time() - start, "file": None,
    }
    if sigma > 0 and term != "error" and rec["act"]:
        f = job["out"]
        np.savez(f, imgs=np.stack(rec["imgs"]), state=np.stack(rec["state"]),
                 act=np.stack(rec["act"]), nexec=np.array(rec["nexec"], dtype=np.int16))
        out["file"] = f
    return out
