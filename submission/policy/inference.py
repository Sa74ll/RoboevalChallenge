"""Self-contained ACT inference for the CoRL 2026 challenge submission.

Reproduces the lerobot 0.4.4 training-time pre/post-processing exactly:
- MEAN_STD normalization (x - mean) / (std + 1e-8) on images, robot state and
  action targets, using the dataset statistics stored in the checkpoint
  directory's normalizer safetensors (the same tensors the model trained on).
- ACT forward pass (vendored modeling_act.py).
- MEAN_STD unnormalization of the predicted chunk.
- Chunk execution: by default an action queue of size n_action_steps (open loop, as trained).
  An optional <checkpoint>/inference.json overrides it per task:
    {"n_action_steps": k}              replan every k steps (k <= chunk_size)
    {"temporal_ensemble_coeff": m}     query every step and average all overlapping chunk
                                       predictions with weights exp(-m * i), i = 0 for the
                                       oldest prediction (ACT temporal ensembling)
    {"carry_queue_across_episodes": true}  keep the original behaviour of not clearing the
                                       queue between episodes (see policy.py)
    {"ensemble": ["subdir", ...]}      also load these checkpoints (relative to this one) and
                                       average every model's unnormalized chunk
    {"action_scale": s}                multiply the arms' pose deltas (all but the last two,
                                       gripper, dims) by s before they are executed (s < 1: slower)
    {"unstick": {"window": w, "thr": t, "steps": n, "z_std": z}}
                                       when the arms' mean |pose delta| (summed over dims) over the
                                       last w executed steps is below t, replan now and for the next
                                       n steps use a latent drawn from N(0, z^2) (seeded by the
                                       episode) instead of ACT's zero latent: another mode of the
                                       learned action distribution, to leave a stalled state
"""

import fcntl
import json
import os
import tempfile
import time
from collections import deque
from pathlib import Path

GPU_SLOT_FILE = os.path.join(tempfile.gettempdir(), "roboeval_gpu_slot.lock")

import numpy as np
import torch
import torchvision
from safetensors.torch import load_file

try:
    from . import modeling_act as ma  # package context (local tests)
except ImportError:
    import modeling_act as ma  # harness context: policy/ is on sys.path

EPS = 1e-8


class _Feature:
    def __init__(self, shape, ftype):
        self.shape = tuple(shape)
        self.type = ftype


class _Config:
    """Runtime stand-in for ACTConfig built from the checkpoint's config.json."""

    def __init__(self, d: dict):
        for k, v in d.items():
            self.__dict__[k] = v
        self.input_features = {k: _Feature(v["shape"], v["type"]) for k, v in d["input_features"].items()}
        self.output_features = {k: _Feature(v["shape"], v["type"]) for k, v in d["output_features"].items()}
        self.image_features = [k for k, v in self.input_features.items() if v.type == "VISUAL"]
        states = [k for k, v in self.input_features.items() if v.type == "STATE"]
        self.robot_state_feature = self.input_features[states[0]] if states else None
        self.env_state_feature = (
            self.input_features["observation.env_state"] if "observation.env_state" in self.input_features else None
        )
        self.action_feature = self.output_features.get("action")
        # The backbone weights are overwritten by load_state_dict below anyway;
        # pass None so torchvision never tries to download them at inference time
        # (the organizer's evaluation machine may be offline).
        self.pretrained_backbone_weights = None


class ACTInference:
    def __init__(self, checkpoint_dir: str | Path, device: str | torch.device = "cuda"):
        dev = torch.device(device) if isinstance(device, str) else device
        if dev.type == "cuda" and not torch.cuda.is_available():
            # CPU-only torch or no usable driver: mem_get_info would raise AssertionError.
            dev = torch.device("cpu")
        if dev.type == "cuda":
            # Admit a worker to the GPU only with 2.5 GB free (a full worker's reservation is
            # ~0.6-1 GB including first-pass activations and cuDNN workspace, plus headroom for
            # the colocated vLLM service, whose preallocation fluctuates with its own workload).
            # The probe itself can raise: cudaMemGetInfo surfaces asynchronous OOMs from
            # sibling processes' failed allocations, and a poisoned context must not take the
            # worker down. Any probe failure means "not enough free" -> CPU.
            try:
                free, _ = torch.cuda.mem_get_info(dev)
            except (torch.OutOfMemoryError, torch.AcceleratorError, RuntimeError):
                free = 0
            if free < 2_500_000_000:
                dev = torch.device("cpu")
            elif self._acquire_gpu_slot(timeout_s=10.0) is None:
                # A sibling worker already holds the single GPU slot: with ~3 GB shared with a
                # colocated vLLM service and 8 per-task rendering contexts, admitting more than
                # one policy worker to CUDA turns simultaneous startup into an OOM cascade that
                # poisons sibling contexts. Every other worker runs its policy on CPU.
                dev = torch.device("cpu")
        self._ckpt = Path(checkpoint_dir)
        self._load(dev)
        if dev.type == "cuda":
            # The probe is only a pre-filter: N sibling workers can pass it simultaneously and
            # then over-allocate together. A dummy forward pass makes each CUDA worker build its
            # real allocator reservation before any sibling's probe is misled, and a worker whose
            # reservation OOMs falls back to CPU with a clean context instead of surfacing an
            # asynchronous CUDA kernel error mid-episode.
            try:
                self._warmup()
            except (torch.OutOfMemoryError, torch.AcceleratorError, RuntimeError):
                self._release()
                dev = torch.device("cpu")
                self._load(dev)
        self._action_queue = deque([], maxlen=self.n_action_steps)
        self._te_chunks = deque()
        self._t = 0
        self._noop_left = self.start_noop_steps
        self._init_unstick(None)

    @staticmethod
    def _acquire_gpu_slot(timeout_s: float = 10.0):
        """Acquire the single GPU slot, or return None if another live worker holds it.

        The slot is the lock file itself: creation is exclusive, and the file records the
        holder's pid so a file left behind by a dead holder is reclaimed (a crashed run never
        wedges the next one). At most one policy worker per run runs its model on CUDA.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fd = os.open(GPU_SLOT_FILE, os.O_CREAT | os.O_EXCL, 0o644)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return True
            except FileExistsError:
                try:
                    with open(GPU_SLOT_FILE) as f:
                        holder = int((f.read() or "").strip() or 0)
                    if holder and holder != os.getpid():
                        os.kill(holder, 0)  # alive (may be our own parent pool manager)
                except (ValueError, OSError, ProcessLookupError):
                    try:
                        os.unlink(GPU_SLOT_FILE)  # stale lock from a dead holder
                        continue
                    except OSError:
                        pass
                if time.monotonic() > deadline:
                    return None
                time.sleep(0.05)
            except OSError:
                if time.monotonic() > deadline:
                    return None
                time.sleep(0.05)
    def _warmup(self) -> None:
        with torch.no_grad():
            batch = {}
            for cam_key in self.cfg.image_features:
                shp = self.cfg.input_features[cam_key].shape
                batch[cam_key] = torch.zeros(1, *shp, device=self.device)
            shp = self.cfg.robot_state_feature.shape
            batch[self.state_key] = torch.zeros(1, *shp, device=self.device)
            if self.cfg.env_state_feature is not None:
                shp = self.cfg.env_state_feature.shape
                batch["observation.env_state"] = torch.zeros(1, *shp, device=self.device)
            batch[ma.OBS_IMAGES] = [batch[k] for k in self.cfg.image_features]
            self.model(batch)

    def _release(self) -> None:
        del self.model
        self.model = None
        self.stats = {}
        self.extra = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def _load(self, dev: torch.device) -> None:
        self.device = dev
        ckpt = self._ckpt
        self.cfg = _Config(json.loads((ckpt / "config.json").read_text()))
        self.n_action_steps = int(self.cfg.n_action_steps)
        self.te_coeff = None
        self.carry_queue = False
        self.start_noop_steps = 0
        exec_cfg = ckpt / "inference.json"
        ex = {}
        if exec_cfg.exists():
            ex = json.loads(exec_cfg.read_text())
            self.carry_queue = bool(ex.get("carry_queue_across_episodes", False))
            if ex.get("temporal_ensemble_coeff") is not None:
                self.te_coeff = float(ex["temporal_ensemble_coeff"])
            # The demos start one step after reset, with the fingers already opening; at reset the
            # proprioception reports closed, motionless fingers, which some checkpoints read as
            # "holding" and answer by closing the grippers for good. Hold still with open grippers
            # for this many steps first, so the policy's first observation matches the demos.
            self.start_noop_steps = int(ex.get("start_noop_open_steps", 0))
            if ex.get("n_action_steps") is not None:
                self.n_action_steps = int(ex["n_action_steps"])
        self.action_scale = float(ex.get("action_scale", 1.0))
        self.unstick = ex.get("unstick")
        self.state_key = self._state_key(self.cfg)
        self.stats = self._read_stats(ckpt, dev)
        self.model = self._read_model(self.cfg, ckpt, dev)
        # {"ensemble": ["subdir", ...]}: further checkpoints (paths relative to this one) whose
        # unnormalized chunks are averaged with this one's, each with its own normalization stats.
        self.extra = []
        for rel in ex.get("ensemble", []):
            cfg = _Config(json.loads((ckpt / rel / "config.json").read_text()))
            self.extra.append((cfg, self._state_key(cfg), self._read_stats(ckpt / rel, dev),
                               self._read_model(cfg, ckpt / rel, dev)))
        self.chunk_len = min([int(self.cfg.chunk_size)] + [int(m[0].chunk_size) for m in self.extra])
        assert 1 <= self.n_action_steps <= self.chunk_len

    @staticmethod
    def _state_key(cfg: _Config) -> str:
        return next(k for k, v in cfg.input_features.items() if v is cfg.robot_state_feature)

    @staticmethod
    def _read_stats(ckpt: Path, dev: torch.device) -> dict:
        stats = {}
        st = load_file(str(ckpt / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
        for key, tensor in st.items():
            if key.endswith(".mean") or key.endswith(".std"):
                feat, stat = key.rsplit(".", 1)
                stats.setdefault(feat, {})[stat] = tensor.to(dev)
        return stats

    @staticmethod
    def _read_model(cfg: _Config, ckpt: Path, dev: torch.device):
        model = ma.ACT(cfg).to(dev)
        raw = load_file(str(ckpt / "model.safetensors"))
        state = {k.removeprefix("model."): v for k, v in raw.items()}
        # Copies into the fp32 parameters (fp16 files load too). The VAE encoder only runs in training mode, so the
        # shipped checkpoints may leave its weights out (strip_vae.py); every other key must match exactly.
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("vae_encoder") for k in missing):
            raise RuntimeError(f"checkpoint {ckpt} does not match the model: missing {missing}, unexpected {unexpected}")
        model.eval()
        return model

    def _fallback_to_cpu(self) -> None:
        self._load(torch.device("cpu"))
        self._action_queue = deque([], maxlen=self.n_action_steps)

    def reset(self, episode) -> None:
        self._action_queue = deque([], maxlen=self.n_action_steps)
        self._te_chunks = deque()  # (start_step, chunk ndarray) for temporal ensembling
        self._t = 0
        self._noop_left = self.start_noop_steps
        self._init_unstick(episode)

    def _init_unstick(self, episode) -> None:
        if self.unstick:
            self._motion = deque(maxlen=int(self.unstick["window"]))
            self._kick_left = 0
            self._rng = np.random.default_rng([int(getattr(episode, "policy_seed", 0)), int(getattr(episode, "seed", 0))])
            self._set_latent(None)

    def _models(self):
        return [self.model] + [m[3] for m in self.extra]

    def _set_latent(self, z) -> None:
        for m in self._models():
            m.latent_override = None if z is None else torch.as_tensor(z, dtype=torch.float32, device=self.device)[None]

    def _watch_motion(self, a: np.ndarray) -> None:
        """unstick: count down an active kick, or start one when the arms have stalled."""
        u = self.unstick
        if self._kick_left > 0:
            self._kick_left -= 1
            if self._kick_left == 0:
                self._set_latent(None)
                self._motion.clear()
            return
        self._motion.append(float(np.abs(a[:-2]).sum()))
        if len(self._motion) == self._motion.maxlen and np.mean(self._motion) < float(u["thr"]):
            self._kick_left = int(u["steps"])
            self._set_latent(self._rng.normal(0.0, float(u.get("z_std", 1.0)), int(self.cfg.latent_dim)))
            self._action_queue.clear()
            self._te_chunks.clear()

    @staticmethod
    def _normalize(stats: dict, key: str, tensor: torch.Tensor) -> torch.Tensor:
        s = stats.get(key)
        if s is None:
            return tensor
        return (tensor - s["mean"]) / (s["std"] + EPS)

    @staticmethod
    def _unnormalize(stats: dict, key: str, tensor: torch.Tensor) -> torch.Tensor:
        s = stats.get(key)
        if s is None:
            return tensor
        return tensor * (s["std"] + EPS) + s["mean"]

    def _model_chunk(self, cfg, state_key, stats, model, obs: dict) -> np.ndarray:
        """One forward pass of one model -> unnormalized (chunk_size, action_dim) actions."""
        batch = {}
        for cam_key in cfg.image_features:
            name = cam_key.rsplit(".", 1)[1]  # "observation.images.head" -> "head" (obs key is rgb_head)
            img = np.ascontiguousarray(obs[f"rgb_{name}"])  # (3, H, W) uint8 — the evaluator's rgb layout
            t = torch.from_numpy(img).to(torch.float32) / 255.0
            t = t.unsqueeze(0).to(self.device)  # (1, 3, H, W)
            batch[cam_key] = self._normalize(stats, cam_key, t)
        state = torch.from_numpy(np.asarray(obs["proprioception"], dtype=np.float32))
        batch[state_key] = self._normalize(stats, state_key, state.unsqueeze(0).to(self.device))
        batch[ma.OBS_IMAGES] = [batch[k] for k in cfg.image_features]
        actions, _ = model(batch)  # forward returns (actions, latent_pdf_params)
        return self._unnormalize(stats, "action", actions[0]).cpu().numpy()

    def _predict_chunk(self, obs: dict) -> np.ndarray:
        """Unnormalized (chunk, action_dim) actions: this checkpoint's, or the mean over the ensemble."""
        chunk = self._model_chunk(self.cfg, self.state_key, self.stats, self.model, obs)
        if not self.extra:
            return chunk
        chunks = [chunk] + [self._model_chunk(*m, obs) for m in self.extra]
        return np.mean([c[: self.chunk_len] for c in chunks], axis=0)

    def _act_once(self, obs: dict) -> np.ndarray:
        if self.te_coeff is None:
            if len(self._action_queue) == 0:
                chunk = self._predict_chunk(obs)
                self._action_queue.extend(list(chunk[: self.n_action_steps]))
            return self._action_queue.popleft()
        # Temporal ensembling: a fresh chunk every step; average every stored prediction for the
        # current step, oldest first, weighted exp(-m * i).
        self._te_chunks.append((self._t, self._predict_chunk(obs)))
        while self._t - self._te_chunks[0][0] >= len(self._te_chunks[0][1]):
            self._te_chunks.popleft()
        preds = np.stack([c[self._t - s] for s, c in self._te_chunks])
        w = np.exp(-self.te_coeff * np.arange(len(preds)))
        self._t += 1
        return (w[:, None] * preds).sum(0) / w.sum()

    def act(self, obs: dict) -> np.ndarray:
        if self._noop_left > 0:
            self._noop_left -= 1
            return np.zeros(int(self.cfg.output_features["action"].shape[0]), dtype=np.float32)  # 0 = open
        while True:
            with torch.no_grad():
                try:
                    a = self._act_once(obs)
                    if self.action_scale != 1.0:
                        a = a.copy()
                        a[:-2] *= self.action_scale  # BimanualPanda ee deltas: 2 x (xyz, rotvec), then 2 grippers
                    if self.unstick:
                        self._watch_motion(a)
                    return a
                except (torch.OutOfMemoryError, torch.AcceleratorError):
                    # VRAM was taken by another process after the constructor probe.
                    if self.device.type == "cuda":
                        self._release()
                        self._load(torch.device("cpu"))
                        continue
                    raise
                except RuntimeError as e:
                    # Some asynchronous CUDA failures surface as plain RuntimeError. Only a
                    # CUDA-related one demotes the worker; anything else is a real bug.
                    if self.device.type == "cuda" and "CUDA" in str(e):
                        self._release()
                        self._load(torch.device("cpu"))
                        continue
                    raise
