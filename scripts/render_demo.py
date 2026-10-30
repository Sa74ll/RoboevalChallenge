"""Render the bundle solving each task: one animated GIF with all tasks side by side (the README's demo).

Episodes run exactly as in the official evaluator (same episode loop, same policy seeding, same interface from
metadata.yaml). Frames come from a separate MuJoCo renderer pointed at the head camera; rendering does not change the
simulation. The default seed, 2000, is the first seed of our validation set, used for every task.

  MUJOCO_GL=egl python scripts/render_demo.py                                # all 8 tasks -> assets/demo.gif
  MUJOCO_GL=egl python scripts/render_demo.py --tasks LiftPot --seed 2001 --out lp.gif
"""
import os

# the evaluator's pinned math settings (roboeval.challenge.determinism.PINNED_ENV), set before numpy loads
PINNED = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MKL_CBWR": "COMPATIBLE"}
os.environ.update(PINNED)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from roboeval.challenge import spec  # noqa: E402
from roboeval.challenge.determinism import PINNED_ENV  # noqa: E402
from roboeval.challenge.evaluate import _seed_policy_rngs  # noqa: E402
from roboeval.challenge.submission import (EpisodeSpec, Interface, load_metadata, load_policy,  # noqa: E402
                                           make_env, task_instruction)

REPO = Path(__file__).resolve().parent.parent
TASKS = ["StackTwoBlocks", "RotateValve", "PackBox", "CubeHandover",
         "LiftTray", "PickSingleBookFromTable", "StackSingleBookShelf", "LiftPot"]
SHORT = {"PickSingleBookFromTable": "PickBook", "StackSingleBookShelf": "ShelfBook"}


def run_episode(task, seed, sub, md, itf, res, camera, every):
    """One episode as the evaluator runs it; returns (frames, termination, steps). A frame every `every` steps."""
    env = make_env(task, "PositionAndOrientation", itf)
    policy = load_policy(sub, md)
    u = env.unwrapped
    model, data = u._mojo.model, u._mojo.data
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, res)
    model.vis.global_.offheight = max(model.vis.global_.offheight, res)
    renderer = mujoco.Renderer(model, res, res)
    cam = u._cameras_map[camera][0]
    ep = EpisodeSpec(family=task, variation="PositionAndOrientation", env_name=type(env).__name__,
                     instruction=task_instruction(task), episode_index=0, seed=seed,
                     policy_seed=int(md.get("policy_seed", 0)), control_frequency=itf.control_frequency,
                     max_steps=spec.max_steps(task, itf.control_frequency), action_space=env.action_space)

    frames = []

    def grab():
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    obs, _ = env.reset(seed=seed)
    _seed_policy_rngs(ep.policy_seed, ep.seed)
    policy.reset(ep)
    grab()
    termination, steps = "budget", 0
    for steps in range(1, ep.max_steps + 1):
        obs, _, terminated, truncated, _ = env.step(np.asarray(policy.act(obs), dtype=np.float32))
        if steps % every == 0:
            grab()
        if env.success:
            termination = "success"
            break
        if terminated:
            termination = "fail"
            break
        if truncated:
            termination = "unhealthy"
            break
    if steps % every:
        grab()
    renderer.close()
    env.close()
    return frames, termination, steps


def label(frame, title, t_s, done):
    """Task name top left, sim time top right; a result banner once the episode has ended."""
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    s = im.width
    font = ImageFont.load_default(size=max(12, s // 16))
    small = ImageFont.load_default(size=max(10, s // 20))
    d.text((8, 6), title, font=font, fill="white", stroke_width=3, stroke_fill="black")
    d.text((s - 8, 8), f"{t_s:4.1f} s", font=small, fill="white", stroke_width=3, stroke_fill="black", anchor="ra")
    if done is not None:
        ok = done == "success"
        d.rectangle((0, s - s // 8, s, s), fill=(22, 128, 60) if ok else (170, 40, 40))
        d.text((s // 2, s - s // 16), "success" if ok else done, font=font, fill="white", anchor="mm")
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default=str(REPO / "submission"))
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--camera", default="head")
    ap.add_argument("--res", type=int, default=256, help="tile size in pixels")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--every", type=int, default=2, help="keep every k-th control step (20 Hz / k frames per second)")
    ap.add_argument("--hold", type=float, default=1.5, help="seconds to hold the final frame")
    ap.add_argument("--colors", type=int, default=128, help="GIF palette size")
    ap.add_argument("--out", default=str(REPO / "assets" / "demo.gif"))
    a = ap.parse_args()
    assert PINNED == PINNED_ENV, "the evaluator's pinned math settings changed; update PINNED"
    if 1000 <= a.seed <= 1049:
        raise SystemExit("seeds 1000-1049 are the official evaluation seeds; pick another seed")

    sub = Path(a.submission)
    md = load_metadata(sub)
    itf = Interface.from_dict(md["interface"])
    dt = a.every / itf.control_frequency
    tiles = []
    for task in a.tasks:
        frames, termination, steps = run_episode(task, a.seed, sub, md, itf, a.res, a.camera, a.every)
        print(f"{task} seed {a.seed}: {termination} after {steps} steps ({steps / itf.control_frequency:.1f} s)",
              flush=True)
        name = SHORT.get(task, task)
        tiles.append([label(f, name, min(i * dt, steps / itf.control_frequency), None) for i, f in enumerate(frames)]
                     + [label(frames[-1], name, steps / itf.control_frequency, termination)])

    n = max(len(t) for t in tiles) + int(a.hold / dt)
    blank = np.zeros_like(tiles[0][0])
    cols = min(a.cols, len(tiles))
    grid_tiles = tiles + [[blank]] * (-len(tiles) % cols)
    grid = []
    for i in range(n):
        rows = [np.concatenate([t[min(i, len(t) - 1)] for t in grid_tiles[r:r + cols]], axis=1)
                for r in range(0, len(grid_tiles), cols)]
        grid.append(np.concatenate(rows, axis=0))
    # one palette for the whole animation (no colour flicker between frames, smaller file)
    palette = Image.fromarray(np.concatenate(grid[::max(1, n // 6)], axis=0)).quantize(colors=a.colors)
    out_frames = [Image.fromarray(g).quantize(palette=palette, dither=Image.Dither.NONE) for g in grid]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:], duration=int(1000 * dt), loop=0,
                       optimize=True)
    print(f"wrote {out} ({len(out_frames)} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
