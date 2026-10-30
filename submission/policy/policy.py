"""ACT policy adapter exposing the challenge interface.

make_policy(checkpoint_dir, metadata) -> policy with .reset(episode) and .act(obs).
obs keys: "proprioception" (36,), "proprioception_grippers" (2,),
"rgb_<camera>" (3, H, W) uint8 for each camera declared in metadata, "depth_<camera>" (H, W) float32
(meters) for cameras declared with depth: true (used by the hybrid controllers only).

Per-task models are permitted by the challenge rules: each task family has its
own checkpoint under checkpoint_dir/<family>/. The evaluator calls make_policy
once per task and then reset(episode) before every episode, so the family
checkpoint is loaded lazily on first use and cached per family.
"""

import json

import numpy as np
from pathlib import Path


try:
    from .inference import ACTInference  # package context (local tests)
except ImportError:
    from inference import ACTInference  # harness context: policy/ is on sys.path
try:  # hand-designed controllers (classical RGB-D perception + scripted control), selected by checkpoint/<family>/hybrid.json
    from .hybrid_s2b3 import HybridS2B3
    from .hybrid_rv import HybridRV
    from .hybrid_pb import HybridPB
    from .hybrid_ch import HybridCH
    from .hybrid_lt import HybridLT
    from .hybrid_pk import HybridPK
    from .hybrid_ss import HybridSS
except ImportError:
    from hybrid_s2b3 import HybridS2B3
    from hybrid_rv import HybridRV
    from hybrid_pb import HybridPB
    from hybrid_ch import HybridCH
    from hybrid_lt import HybridLT
    from hybrid_pk import HybridPK
    from hybrid_ss import HybridSS
HYBRIDS = {"hybrid_s2b3": HybridS2B3, "hybrid_rv": HybridRV, "hybrid_pb": HybridPB, "hybrid_ch": HybridCH, "hybrid_lt": HybridLT, "hybrid_pk": HybridPK, "hybrid_ss": HybridSS}
class _Policy:
    def __init__(self, checkpoint_dir):
        self._checkpoint_dir = Path(checkpoint_dir)
        self._cache = {}
        self._last_family = None

    def reset(self, episode) -> None:
        # The evaluator calls make_policy per task, but select the family model on
        # every reset so a single policy instance can never serve the wrong task.
        family = getattr(episode, "family", None)
        if family is None:
            raise ValueError(
                "episode.family is missing; per-task checkpoints require an explicit family"
            )
        if family not in self._cache:
            ckpt = self._checkpoint_dir / family
            if (ckpt / "hybrid.json").is_file():  # {"policy": name in HYBRIDS, "cfg": {constructor options}}
                spec = json.loads((ckpt / "hybrid.json").read_text())
                self._cache[family] = HYBRIDS[spec["policy"]](**spec.get("cfg", {}))
            else:
                if not (ckpt / "config.json").is_file():
                    raise ValueError(f"no checkpoint for family {family!r} under {self._checkpoint_dir}")
                self._cache[family] = ACTInference(ckpt)
        # Clear the per-episode execution state (action queue / ensembling buffer) so an episode
        # never starts by replaying actions predicted for the previous one. A task can opt out with
        # {"carry_queue_across_episodes": true} in its inference.json: the original submission never
        # reset the queue, and on StackTwoBlocks that measured better (16 vs 8 successes / 50).
        if not self._cache[family].carry_queue:
            self._cache[family].reset(episode)
        self._last_family = family

    def act(self, obs: dict) -> np.ndarray:
        # act() is only called after a reset() that fixed the current family.
        family = self._last_family
        return self._cache[family].act(obs)


def make_policy(checkpoint_dir, metadata) -> _Policy:
    return _Policy(checkpoint_dir)
