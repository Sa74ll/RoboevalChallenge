#!/usr/bin/env bash
# GRPO on the 5090 (Oct 21): LiftPot from pick6's model (pick4 L-30k), once LiftTray round 2 (unit sil_lt_r2b) is
# done with the GPU. LiftPot fails by timeouts (13/50 in pick6's 5070 run), the case where GRPO helped before.
# Same recipe as the tower runs (grpo_tower.sh); 10 workers to fit the 30 GB of RAM.
cd ~/local_models/work/rl
until grep -qs LT_ROUND2_DONE ../sil/lt_round2b.out || ! systemctl --user -q is-active sil_lt_r2b; do sleep 120; done
echo "$(date +%T) START grpo p6_LiftPot"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=~/miniforge3/envs/roboeval/bin/python
$PY grpo.py --ckpt ../pick6/sub_pick6/checkpoint/LiftPot --bundle ../pick6/sub_pick6 --family LiftPot \
  --out ../rl_runs/p6_LiftPot --iters 200 --workers 10 --learner-python $PY --shm /dev/shm/grpo_p6_LiftPot
echo "$(date +%T) END grpo p6_LiftPot rc=$?"
