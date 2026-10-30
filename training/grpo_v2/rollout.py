"""Simulator rollouts for GRPO: one env + CPU policy per pool worker (RoboEval env, CPU torch).

An episode mirrors the evaluator's loop (roboeval/challenge/evaluate.py run_episode) with the
submission's execution settings: `noop` zero actions with open grippers first (start fix), then a
fresh chunk every `n_exec` steps. With sigma > 0, Gaussian noise is added to the executed chunk in
normalized action units and the (observation, noisy normalized chunk) pairs are saved for the update.
job["action_scale"] (default 1) scales the arms' pose deltas at execution, as the bundle does.
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
        family=family,
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
    executed = []  # every action sent to the simulator (start no-op, branch prefix, policy): exact replay from the seed
    # StackTwoBlocks subgoals (Oct 24, opt-in): the first grasp the policy makes (after any replayed prefix): horizontal
    # gripper-block offset, then that block's highest lift while that hand holds it. Training seeds: grasps < 1 cm off
    # centre stack 92%, > 3 cm 46%; lifts < 2 cm 19%, > 6 cm 91% (PROGRESS.md, Oct 24).
    # A grasp counts only if it starts after the policy took over: blocks already in hand at the end of a replayed prefix
    # are ignored until that hand lets go (so lifting an already stacked block earns nothing).
    sg = ({"off": None, "lift": None, "held": None, "z0": 0.0, "pre": set()}
          if job.get("subgoal") and _W.get("family") == "StackTwoBlocks" else None)
    # --critic (Oct 24, StackTwoBlocks and ShelfBook): simulator state at the start of every recorded chunk, for a value model that
    # shapes training advantages only (it never runs at test time)
    priv_on = bool(job.get("priv")) and sigma > 0 and _W.get("family") in ("StackTwoBlocks", "StackSingleBookShelf")
    rec["priv"] = []
    free_from = int(job.get("prefix_t") or 0) if job.get("prefix_npz") else int(job["noop"])
    try:
        obs, info = env.reset(seed=job["seed"])
        done = False
        u = getattr(env, "unwrapped", env)

        objs = getattr(u, "blocks", None) or getattr(u, "books", None) or []  # StackTwoBlocks' blocks or ShelfBook's book

        def holding():
            return {(s, b) for s in u.robot.grippers for b, blk in enumerate(objs) if u.robot.is_gripper_holding_object(blk, s)}

        def subgoals():
            h = holding()
            sg["pre"] &= h  # a pre-held block that was let go can count when grasped again
            if sg["off"] is None:  # waiting for the policy's first grasp
                for s, b in sorted(h - sg["pre"], key=str):
                    g, p = u.robot.grippers[s].body.get_position(), u.blocks[b].body.get_position()
                    sg.update(off=float(np.linalg.norm(g[:2] - p[:2])), lift=0.0, held=(s, b), z0=float(p[2]))
                    return
            elif sg["held"] is not None:  # that block's highest lift until this hand first lets go
                if sg["held"] in h:
                    sg["lift"] = max(sg["lift"], float(u.blocks[sg["held"][1]].body.get_position()[2]) - sg["z0"])
                else:
                    sg["held"] = None

        def privileged(state):  # time, progress, object poses, gripper positions and offsets, holding flags, proprioception
            sides = sorted(u.robot.grippers, key=str)
            bp = [np.asarray(b.body.get_position(), dtype=np.float32) for b in objs]
            bq = [np.asarray(b.body.get_quaternion(), dtype=np.float32) for b in objs]
            gp = [np.asarray(u.robot.grippers[s].body.get_position(), dtype=np.float32) for s in sides]
            h = holding()
            if _W["family"] == "StackTwoBlocks":
                rel = bp[1] - bp[0]  # block to block
            else:  # ShelfBook: the book touching the lower / upper shelf, its height above the start
                rel = [float(objs[0].is_colliding(u.book_shelf.lower_shelf_body)), float(objs[0].is_colliding(u.book_shelf.upper_shelf_body)),
                       float(bp[0][2] - u.book_init_height)]
            return np.concatenate([[steps / max_steps, float(info.get("subtask_progress", 0.0) or 0.0)], *bp, *bq, rel,
                                   *gp, *[g - b for g in gp for b in bp], [float((s, k) in h) for s in sides for k in range(len(bp))],
                                   state]).astype(np.float32)

        def step(a):
            nonlocal obs, info, steps, term, done
            executed.append(np.asarray(a, dtype=np.float32))
            obs, _, terminated, truncated, info = env.step(np.asarray(a, dtype=np.float32))
            steps += 1
            if sg is not None:
                if steps == free_from:
                    sg["pre"] = holding()
                elif steps > free_from:
                    subgoals()
            if env.success:
                term, done = "success", True
            elif terminated:
                term, done = "fail", True
            elif truncated:
                term, done = "unhealthy", True

        if job.get("prefix_npz"):
            # Branch start (Oct 23): replay a logged episode's first prefix_t executed actions exactly (they include its
            # start no-op; bit-exact with pinned BLAS), then the policy takes over. Only the policy's chunks are recorded.
            prefix = np.load(job["prefix_npz"])["actions"][: int(job["prefix_t"])].astype(np.float32)
            step(prefix[0])
            dist0 = dict(info.get("target_distance") or {})
            for x in prefix[1:]:
                if done or steps >= max_steps:
                    break
                step(x)
        else:
            for _ in range(job["noop"]):
                if not done and steps < max_steps:
                    step(np.zeros(net.action_dim, dtype=np.float32))  # 0 = grippers open
            dist0 = dict(info.get("target_distance") or {})  # start-of-episode distances, for shaping
        while not done and steps < max_steps:
            imgs, state = net.obs_arrays(obs)
            p = privileged(state) if priv_on else None
            with torch.no_grad():
                mu = net.mu(torch.from_numpy(imgs)[None], torch.from_numpy(state)[None])[0, :n_exec].numpy()
            if sigma > 0 and job.get("noise_hold"):  # one perturbation held over the chunk (report: per-step white noise is too weak)
                a_n = mu + rng.normal(0.0, sigma, (1, mu.shape[1])).astype(np.float32)
            else:
                a_n = mu + rng.normal(0.0, sigma, mu.shape).astype(np.float32) if sigma > 0 else mu
            a = net.unnorm_action(torch.from_numpy(a_n)).numpy()
            if job.get("action_scale", 1.0) != 1.0:  # as the bundle's inference.json action_scale (ShelfBook 0.85):
                a = a.copy()  # the arms' pose deltas, not the grippers; the update still sees the unscaled a_n
                a[:, :-2] *= job["action_scale"]
            k = 0
            while k < n_exec and steps < max_steps and not done:
                step(a[k])
                k += 1
            if sigma > 0:
                rec["imgs"].append(imgs)
                rec["state"].append(state)
                rec["act"].append(a_n.astype(np.float32))
                rec["nexec"].append(k)
                if p is not None:
                    rec["priv"].append(p)
    except Exception:
        term, err = "error", traceback.format_exc(limit=6)
        _W["env"] = _W["make"]()  # a broken simulator must not poison the next episode
    from roboeval.challenge import results as res

    raw = res.to_jsonable(info)
    if term == "error":
        raw["success"] = 0.0
    out = {
        "seed": job["seed"], "gkey": job.get("gkey", job["seed"]), "g": job["g"], "term": term, "error": err, "steps": steps,
        "success": float(term == "success"), "tp": float(info.get("subtask_progress", 0.0) or 0.0),
        # the 14 leaderboard metrics of this episode, exactly as the evaluator derives them
        "metrics": res.episode_metrics(raw, steps, _W["freq"]),
        "dist0": dist0, "dist_end": dict(info.get("target_distance") or {}),
        "wall": time.time() - start, "file": None,
        # executed actions (Oct 23, user: save GRPO successes for training); replayable from the seed with pinned BLAS
        "actions": np.stack(executed).astype(np.float32) if term != "error" and executed else None,
        "prefix_t": int(job.get("prefix_t") or 0) if job.get("prefix_npz") else 0,  # leading steps replayed from a logged episode
        "sg_off": sg["off"] if sg else None, "sg_lift": sg["lift"] if sg else None,
        "priv": np.stack(rec["priv"]) if rec["priv"] and term != "error" else None,  # one row per recorded chunk
    }
    if sigma > 0 and term != "error" and rec["act"]:
        f = job["out"]
        np.savez(f, imgs=np.stack(rec["imgs"]), state=np.stack(rec["state"]),
                 act=np.stack(rec["act"]), nexec=np.array(rec["nexec"], dtype=np.int16))
        out["file"] = f
    return out
