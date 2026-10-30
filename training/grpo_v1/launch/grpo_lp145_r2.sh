#!/usr/bin/env bash
# GRPO LiftPot round 2 on the 5090 (Oct 22 11:40). The 5090's PackBox branch (grpo_pb1155) finished its 200 rounds at
# 11:24 and the tower's LiftPot round 2 was still queued behind RotateValve (grpo_chain4_lp, now dequeued), so it runs
# here instead: from round 145 (pick8's LiftPot model, SHA-1 97ecd8b6), which is also the KL reference, as on the
# tower; rounds numbered 1000-1200 so the training seeds differ from round 1. 5 workers under a 22 GB cap.
cd ~/local_models/work/rl
echo "$(date +%T) START grpo lp145_r2_5090"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=~/miniforge3/envs/roboeval/bin/python
$PY grpo.py --ckpt ../ens/members/glp145 --start-iter 1000 \
  --bundle ../pick6/sub_pick6 --family LiftPot --out ../rl_runs/lp145_r2_5090 --iters 1200 --workers 5 \
  --learner-python $PY --shm /dev/shm/grpo_lp145_r2
echo "$(date +%T) END grpo lp145_r2_5090 rc=$?"
