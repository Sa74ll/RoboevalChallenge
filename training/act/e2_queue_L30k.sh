#!/usr/bin/env bash
# E2 on the local RTX 5090 (freed from vLLM 2026-10-17): all-variant data (E1 arm B data),
# batch 32 (4x E1), 30k steps, fresh ImageNet init. Weakest tasks first. One job at a time:
# the machine is CPU/data-bound (~100 samples/s), so concurrent jobs would only split it.
# Waits for each family's dataset (prep_all.py) before training it. Never overwrites a run dir.
set -uo pipefail
cd "$(dirname "$0")/../.."
STEPS=${STEPS:-30000} BATCH=${BATCH:-32}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for fam in StackSingleBookShelf CubeHandover RotateValve LiftPot PickSingleBookFromTable LiftTray PackBox StackTwoBlocks; do
  until grep -qh "^${fam}ALL:" work/e2/logs/prep_*.log 2>/dev/null; do sleep 60; done
  out=work/e2/runs/b${BATCH}_${fam}
  if [ -e "$out" ]; then echo "$(date +%T) skip $fam: $out exists"; continue; fi
  echo "$(date +%T) START $fam"
  conda run --no-capture-output -n roboeval-train lerobot-train --policy.type=act --policy.chunk_size=16 \
    --policy.n_action_steps=16 --policy.use_amp=true --policy.device=cuda --policy.push_to_hub=false \
    --dataset.repo_id=${fam}ALL --dataset.root=work/e2/data/${fam}ALL --dataset.revision=local \
    --output_dir=$out --batch_size=$BATCH --steps=$STEPS --num_workers=${NW:-14} \
    --save_freq=10000 --eval_freq=0 --log_freq=500 --wandb.enable=false > work/e2/logs/train_${fam}.log 2>&1
  echo "$(date +%T) END $fam rc=$? $(grep -oE 'step:[0-9.]+K? .*loss:[0-9.]+' work/e2/logs/train_${fam}.log | tail -1)"
done
echo "$(date +%T) E2_QUEUE_DONE"
