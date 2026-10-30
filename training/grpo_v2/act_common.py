"""Shared ACT helpers for GRPO fine-tuning (rollout workers on CPU, learner on GPU).

Loads a lerobot ACT checkpoint the same way the submission's inference.py does (vendored
modeling_act.py, MEAN_STD stats from the normalizer safetensors) and turns evaluator observations
into the model's normalized input batch, so rollouts see exactly what the submitted policy sees.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "policy_v2"))
import modeling_act as ma  # noqa: E402
from inference import EPS, _Config  # noqa: E402


class ACTNet:
    def __init__(self, ckpt_dir, device="cpu"):
        self.ckpt = Path(ckpt_dir)
        self.device = torch.device(device)
        self.cfg = _Config(json.loads((self.ckpt / "config.json").read_text()))
        self.cams = list(self.cfg.image_features)
        self.obs_names = [f"rgb_{k.rsplit('.', 1)[1]}" for k in self.cams]  # evaluator keys
        self.state_key = next(
            k for k, v in self.cfg.input_features.items() if v is self.cfg.robot_state_feature
        )
        self.action_dim = int(self.cfg.output_features["action"].shape[0])
        self.stats = {}
        st = load_file(str(self.ckpt / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
        for key, t in st.items():
            if key.endswith(".mean") or key.endswith(".std"):
                feat, stat = key.rsplit(".", 1)
                self.stats.setdefault(feat, {})[stat] = t.to(self.device)
        self.model = ma.ACT(self.cfg).to(self.device)
        self.load_weights(self.ckpt / "model.safetensors")
        self.model.eval()  # no dropout, latent z = 0: the deterministic path used at evaluation

    def load_weights(self, path):
        raw = load_file(str(path), device=str(self.device))
        self.model.load_state_dict({k.removeprefix("model."): v for k, v in raw.items()}, strict=True)

    def save_weights(self, path):
        sd = {"model." + k: v.detach().cpu().clone().contiguous() for k, v in self.model.state_dict().items()}
        save_file(sd, str(path))

    def save_checkpoint(self, out_dir):
        """A checkpoint dir in the source's layout (config, normalizer stats), with the current weights."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for f in self.ckpt.iterdir():
            if f.is_file() and f.name != "model.safetensors":
                shutil.copy2(f, out / f.name)
        self.save_weights(out / "model.safetensors")

    def _norm(self, key, t):
        s = self.stats.get(key)
        return t if s is None else (t - s["mean"]) / (s["std"] + EPS)

    def unnorm_action(self, a):
        s = self.stats["action"]
        return a * (s["std"] + EPS) + s["mean"]

    def obs_arrays(self, obs):
        """Evaluator observation -> (uint8 (n_cams, 3, H, W), float32 (state_dim,))."""
        imgs = np.stack([np.ascontiguousarray(obs[n]) for n in self.obs_names])
        return imgs, np.asarray(obs["proprioception"], dtype=np.float32)

    def mu(self, imgs, state):
        """Normalized action chunk (B, chunk, action_dim) from uint8 images (B, n_cams, 3, H, W)
        and raw states (B, state_dim)."""
        imgs = imgs.to(self.device, non_blocking=True)
        batch = {k: self._norm(k, imgs[:, i].float() / 255.0) for i, k in enumerate(self.cams)}
        batch[self.state_key] = self._norm(self.state_key, state.to(self.device).float())
        batch[ma.OBS_IMAGES] = [batch[k] for k in self.cams]
        return self.model(batch)[0]
