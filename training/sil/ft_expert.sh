#!/usr/bin/env bash
# Oct 24 19:20: fine-tune INIT (orch/rl_runs/INIT) on the expert-only dataset sil/${TAG}_DATA (its normalization already set
# to INIT's by the data pipeline), with the s2b_exp1 recipe: peak lr 2e-5 (cosine to 5e-7, 200 warmup steps), no
# augmentation, batch 32, STEPS steps (default 6000), checkpoints every 1500. Waits (up to 3 h) for "DATA_DONE TAG" in
# sil/${TAG}_data.out. Env: TAG INIT GPU [STEPS] [PAUSE = units to stop first, e.g. a GRPO run on the same GPU].
set -uo pipefail
cd ~/roboeval_challenge/orch/sil
PYT=$HOME/miniconda3/envs/roboeval-train/bin
TAG=${TAG:?} INIT=${INIT:?} GPU=${GPU:?} STEPS=${STEPS:-6000}
dl=$((SECONDS + 3 * 3600))
until grep -q "DATA_DONE $TAG" ${TAG}_data.out 2>/dev/null; do
  [ $SECONDS -gt $dl ] && { echo "$(date +%T) TIMEOUT waiting for DATA_DONE $TAG"; exit 1; }
  systemctl --user -q is-active ${TAG}_data || { sleep 5; grep -q "DATA_DONE $TAG" ${TAG}_data.out || { echo "$(date +%T) ${TAG}_data ended without DATA_DONE"; exit 1; }; }
  sleep 30
done
[ -f ${TAG}_DATA/meta/info.json ] || { echo "$(date +%T) no dataset ${TAG}_DATA"; exit 1; }
[ -n "${PAUSE:-}" ] && systemctl --user stop $PAUSE && echo "$(date +%T) paused $PAUSE"
echo "$(date +%T) START $TAG gpu$GPU from $INIT ($(grep -o '"total_episodes": [0-9]*' ${TAG}_DATA/meta/info.json), $(grep -o '"total_frames": [0-9]*' ${TAG}_DATA/meta/info.json))"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_DATASETS_CACHE=/dev/shm/rbe_hfds_$TAG CUDA_VISIBLE_DEVICES=$GPU
$PYT/lerobot-train --policy.path=$PWD/../rl_runs/$INIT --policy.use_amp=true --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=${TAG}_DATA --dataset.root=$PWD/${TAG}_DATA --dataset.revision=local --dataset.image_transforms.enable=false \
  --use_policy_training_preset=false --optimizer.type=adamw --optimizer.lr=2e-5 --optimizer.weight_decay=1e-4 \
  --optimizer.grad_clip_norm=10 --scheduler.type=cosine_decay_with_warmup --scheduler.num_warmup_steps=200 \
  --scheduler.num_decay_steps=$STEPS --scheduler.peak_lr=2e-5 --scheduler.decay_lr=5e-7 \
  --output_dir=runs/$TAG --batch_size=32 --steps=$STEPS --num_workers=12 \
  --save_freq=1500 --eval_freq=0 --log_freq=250 --wandb.enable=false > logs/$TAG.log 2>&1
rc=$?; rm -rf /dev/shm/rbe_hfds_$TAG
echo "$(date +%T) END $TAG rc=$rc $(grep -oE "step:[0-9.]+K? .*loss:[0-9.]+" logs/$TAG.log | tail -1)"
