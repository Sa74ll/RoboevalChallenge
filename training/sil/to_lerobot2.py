"""Turn collect.py episodes into a LeRobot v3 dataset with the demo dataset's features, then (optionally)
merge it with the demos into a training dataset (roboeval-train env, lerobot 0.4.4).

usage: python to_lerobot.py DEMO_ROOT EPISODES_DIR NEW_ROOT [MERGED_ROOT] [--shards N] [--max-episodes M]
With --shards N the episodes are converted by N processes into NEW_ROOT_p0..p{N-1} (about 30 s per episode
in one process), which are merged with the demos into MERGED_ROOT (required then) and deleted.
The demo dataset is only read. NEW_ROOT and MERGED_ROOT must not exist."""
import argparse
import io
import pickle
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

DEFAULT = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def convert(demo_root, files, new_root):
    meta = LeRobotDatasetMetadata(demo_root.name, root=demo_root)
    task = meta.tasks.index[0]
    features = {k: v for k, v in meta.features.items() if k not in DEFAULT}
    ds = LeRobotDataset.create(new_root.name, fps=meta.fps, root=new_root, robot_type=meta.robot_type,
                               features=features, use_videos=False, image_writer_threads=4)
    cams = [k for k, v in features.items() if v["dtype"] == "image"]
    for f in files:
        ep = pickle.load(open(f, "rb"))
        uid = uuid.uuid4().hex
        n = len(ep["frames"])
        if n == 0:  # (Oct 24) e.g. a mirrored branch-start replay that succeeded before its branch point
            print(f"{f.name}: 0 frames, skipped", flush=True)
            continue
        for i, fr in enumerate(ep["frames"]):
            frame = {k: np.asarray(Image.open(io.BytesIO(fr["png"][k.rsplit(".", 1)[1]])).convert("RGB")) for k in cams}
            frame.update({"observation.state": fr["state"], "observation.gripper": fr["gripper"], "action": fr["action"],
                          "reward": np.array([1.0 if i == n - 1 else 0.0], np.float32),
                          "is_terminal": np.array([i == n - 1]), "truncate": np.array([False]),
                          "uuid": uid, "task": task})
            ds.add_frame(frame)
        ds.save_episode()
        print(f"{f.name}: {n} frames", flush=True)
    ds.finalize()
    print("NEW", new_root, ds.meta.total_episodes, "episodes", ds.meta.total_frames, "frames", flush=True)


def merge(roots, merged_root):
    # merge_datasets() would load every input and the result through the HF datasets cache (2x the data on
    # disk) only to read their paths; the file-level aggregation underneath needs no cache.
    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(repo_ids=[r.name for r in roots], aggr_repo_id=merged_root.name, roots=roots,
                       aggr_root=merged_root)
    m = LeRobotDatasetMetadata(merged_root.name, root=merged_root)
    print("MERGED", merged_root, m.total_episodes, "episodes", m.total_frames, "frames", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo_root", type=Path); ap.add_argument("ep_dir", type=Path); ap.add_argument("new_root", type=Path)
    ap.add_argument("merged_root", type=Path, nargs="?")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--part", type=int, default=-1, help="internal: convert only files[part::shards]")
    ap.add_argument("--max-episodes", type=int, default=0, help="random subset of this many (seed 0); 0 = all")
    ap.add_argument("--no-demos", action="store_true", help="(Oct 24) merge only the new episodes (the demo set still gives the features)")
    a = ap.parse_args()
    files = sorted(a.ep_dir.glob("ep_*.pkl"))
    if a.max_episodes and len(files) > a.max_episodes:
        files = sorted(np.random.default_rng(0).choice(files, a.max_episodes, replace=False))
    if a.part >= 0:
        return convert(a.demo_root, files[a.part::a.shards], a.new_root)
    if a.shards == 1:
        convert(a.demo_root, files, a.new_root)
        if a.merged_root:
            merge([a.demo_root, a.new_root], a.merged_root)
        return
    assert a.merged_root, "--shards needs MERGED_ROOT"
    parts = [a.new_root.with_name(f"{a.new_root.name}_p{k}") for k in range(a.shards)]
    procs = [subprocess.Popen([sys.executable, __file__, str(a.demo_root), str(a.ep_dir), str(p), "--shards",
                               str(a.shards), "--part", str(k), "--max-episodes", str(a.max_episodes)])
             for k, p in enumerate(parts)]
    assert all(p.wait() == 0 for p in procs), "a shard failed"
    merge(([] if a.no_demos else [a.demo_root]) + parts, a.merged_root)
    for p in parts:
        shutil.rmtree(p)


if __name__ == "__main__":
    main()
