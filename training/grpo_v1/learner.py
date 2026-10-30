"""GRPO learner (GPU process, run with a CUDA torch; the rollout env's torch is CPU-only).

Reads one command per line on stdin and answers one JSON line on stdout:
  update <iter_dir>   PPO-clip on the episodes listed in <iter_dir>/summary.json (group-normalized
                      advantages), then writes new weights and returns their path and version
  save <out_dir>      writes a checkpoint dir (same layout as the source checkpoint)
  quit
Policy: pi(a | s) = N(mu_theta(s), sigma^2) per executed step in normalized action units, with
mu_theta the deterministic ACT chunk (eval mode, z = 0). Only the first `nexec` actions of a chunk
were executed, so only those enter the ratio. The ResNet backbone is frozen. A KL penalty to the
starting policy (closed form for equal-variance Gaussians) keeps the update close to the demos.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from act_common import ACTNet


def reply(**kw):
    print(json.dumps(kw), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--init-weights", default=None, help="resume from these weights instead of the checkpoint's")
    ap.add_argument("--shm", required=True)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mb", type=int, default=128)
    ap.add_argument("--micro", type=int, default=32)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    net, ref = ACTNet(a.ckpt, "cuda"), ACTNet(a.ckpt, "cuda")
    if a.init_weights:
        net.load_weights(a.init_weights)
    for p in ref.model.parameters():
        p.requires_grad_(False)
    for p in net.model.backbone.parameters():
        p.requires_grad_(False)
    params = [p for p in net.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    shm = Path(a.shm)
    ver = 0
    w = shm / f"policy_v{ver}.safetensors"
    net.save_weights(w)
    reply(ok=True, weights=str(w), ver=ver, trainable=sum(p.numel() for p in params))
    s2 = a.sigma ** 2

    for line in sys.stdin:
        cmd, _, arg = line.strip().partition(" ")
        if cmd == "quit":
            break
        if cmd == "save":
            net.save_checkpoint(arg)
            reply(ok=True, saved=arg)
            continue
        if cmd != "update":
            reply(ok=False, error=f"unknown command {cmd}")
            continue
        summ = json.loads((Path(arg) / "summary.json").read_text())
        imgs, state, act, mask, adv = [], [], [], [], []
        for ep in summ:
            if not ep["file"] or ep["adv"] == 0.0:
                continue
            d = np.load(ep["file"])
            n = len(d["act"])
            imgs.append(d["imgs"]); state.append(d["state"]); act.append(d["act"])
            mask.append(np.arange(d["act"].shape[1])[None, :] < d["nexec"][:, None])
            adv.append(np.full(n, ep["adv"], dtype=np.float32))
        if not adv:
            reply(ok=True, weights=str(w), ver=ver, samples=0)
            continue
        imgs = torch.from_numpy(np.concatenate(imgs))
        state = torch.from_numpy(np.concatenate(state))
        act = torch.from_numpy(np.concatenate(act)).cuda()
        mask = torch.from_numpy(np.concatenate(mask)).float().cuda()
        adv = torch.from_numpy(np.concatenate(adv)).cuda()
        N, K = act.shape[0], act.shape[1]

        def logp_of(mu, idx):
            return -0.5 * ((act[idx] - mu) ** 2).sum(-1) / s2  # (B, K)

        with torch.no_grad():
            logp_old = torch.empty(N, K, device="cuda")
            mu_ref = torch.empty_like(act)
            for i in range(0, N, a.micro):
                idx = torch.arange(i, min(i + a.micro, N))
                logp_old[idx] = logp_of(net.mu(imgs[idx], state[idx])[:, :K], idx)
                mu_ref[idx] = ref.mu(imgs[idx], state[idx])[:, :K]
        stats = {"pg": [], "kl": [], "ratio": [], "clipfrac": []}
        for _ in range(a.epochs):
            perm = torch.randperm(N)
            for i in range(0, N, a.mb):
                idx_mb = perm[i:i + a.mb]
                n_tok = mask[idx_mb].sum()
                opt.zero_grad(set_to_none=True)
                acc = {"pg": 0.0, "kl": 0.0, "ratio": 0.0, "clip": 0.0}
                # gradient accumulation over micro-batches (the GPU is shared with the renderers)
                for j in range(0, len(idx_mb), a.micro):
                    idx = idx_mb[j:j + a.micro]
                    m = mask[idx]
                    mu = net.mu(imgs[idx], state[idx])[:, :K]
                    ratio = torch.exp(logp_of(mu, idx) - logp_old[idx])
                    A = adv[idx][:, None]
                    surr = torch.minimum(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A)
                    pg = -(surr * m).sum() / n_tok
                    kl = (((mu - mu_ref[idx]) ** 2).sum(-1) / (2 * s2) * m).sum() / n_tok
                    (pg + a.beta * kl).backward()
                    with torch.no_grad():
                        acc["pg"] += pg.item(); acc["kl"] += kl.item()
                        acc["ratio"] += ((ratio * m).sum() / n_tok).item()
                        acc["clip"] += ((((ratio - 1).abs() > a.clip).float() * m).sum() / n_tok).item()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                stats["pg"].append(acc["pg"]); stats["kl"].append(acc["kl"])
                stats["ratio"].append(acc["ratio"]); stats["clipfrac"].append(acc["clip"])
        old = w
        ver += 1
        w = shm / f"policy_v{ver}.safetensors"
        net.save_weights(w)
        if old.exists() and ver > 1:
            old.unlink()
        reply(ok=True, weights=str(w), ver=ver, samples=N, **{k: float(np.mean(v)) for k, v in stats.items()})


if __name__ == "__main__":
    main()
