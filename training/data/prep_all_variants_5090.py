"""Aggregate all variants of a family into work/e2/data/<Fam>ALL (read-only on roboeval_data/).

roboeval_data/ already has LeRobot-standard keys and was repaired + verified on 2026-10-17
(repair_source_indices.py / repair_task_index.py). This re-checks each source's indices against
its meta table before aggregating, and checks the output for dead (padding-masked) targets.

Usage: conda run -n roboeval-train python work/e2/prep_all.py <Family> [<Family> ...]
"""
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "roboeval_data"
DST = ROOT / "work" / "e2" / "data"
VARIANTS = ("", "Position", "Orientation", "PositionAndOrientation")
CHUNK = 16


def frames(d: Path) -> pd.DataFrame:
    return pd.concat(pq.read_table(f, columns=["episode_index", "index"]).to_pandas()
                     for f in sorted(d.glob("data/*/*.parquet")))


def meta(d: Path) -> pd.DataFrame:
    return pd.concat(pq.read_table(f, columns=["episode_index", "length", "dataset_from_index", "dataset_to_index"])
                     .to_pandas() for f in sorted(d.glob("meta/episodes/*/*.parquet"))).sort_values("episode_index")


def check(d: Path) -> float:
    """Assert indices match meta; return the fraction of frames whose 16-step chunk is fully valid-free."""
    e, m = frames(d), meta(d)
    assert list(e["index"]) == list(range(len(e))), f"{d.name}: index not 0..N-1"
    assert list(m.episode_index) == list(range(len(m))), f"{d.name}: meta episode_index not 0..M-1"
    b = m.set_index("episode_index").loc[e.episode_index]
    assert (e["index"].values >= b.dataset_from_index.values).all() and \
           (e["index"].values < b.dataset_to_index.values).all(), f"{d.name}: frames outside meta episode bounds"
    # dead target = every step of the chunk is past the episode end (only the episode's last frame
    # has CHUNK-1 padded steps; anything worse means index/meta disagree)
    return float((e["index"].values + 1 >= b.dataset_to_index.values).mean())


def build(fam: str) -> None:
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = [f"{fam}{v}" for v in VARIANTS if (SRC / f"{fam}{v}" / "meta" / "info.json").exists()]
    for n in names:
        check(SRC / n)
    out = DST / f"{fam}ALL"
    if not (out / "meta" / "info.json").exists():
        aggregate_datasets(names, f"{fam}ALL", roots=[SRC / n for n in names], aggr_root=out)
    last = check(out)
    ds = LeRobotDataset(f"{fam}ALL", root=out, revision="local",
                        delta_timestamps={"action": [i / 20 for i in range(CHUNK)]})
    pad = sum(ds[i]["action_is_pad"].float().mean().item() for i in range(0, len(ds), len(ds) // 50)) / 51
    print(f"{fam}ALL: variants={len(names)} episodes={ds.meta.total_episodes} frames={ds.meta.total_frames} "
          f"last-frame share={last:.4f} sampled pad={pad:.3f}", flush=True)


if __name__ == "__main__":
    DST.mkdir(parents=True, exist_ok=True)
    for fam in sys.argv[1:]:
        build(fam)
