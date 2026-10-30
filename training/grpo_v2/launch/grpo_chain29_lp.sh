#!/usr/bin/env bash
# Oct 25 05:25: run L2 = LiftPot GRPO from the expert-data fine-tune lp_exp1 at 3000 steps (68994b5d, frozen as
# rl_runs/lp_exp1_3000_ck + glp1150's inference.json; 5070: 45, 44, 44 = 44.3 vs the incumbent glp1150's 10 whole-bundle runs
# 44.1) on tower GPU 1, the slot run L1 (four checks 1.1-2.4 below the incumbent) held until 05:25. The pattern that lifted
# PickBook (K1: +2.9 validation, +2.3 hold-out) and ShelfBook (M2: +2.0 at its first check): GRPO right after a supervised
# stage, KL reference = the init. v2 settings (sigma 0.2, mean baseline, dynamic sampling x1.5, time penalty 0.2 / last 25%,
# clip-high 0.28), fresh starts, rounds 37000-37200 (offset 37000 marks L2), L1's 8 workers, rollouts saved.
# Successor: re-planned at its 37050 check.
cd ~/roboeval_challenge/orch/rl_branch
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2
export OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_CBWR=COMPATIBLE
PY=$HOME/miniconda3/envs/roboeval/bin/python
echo "$(date +%T) START lp_l2"
exec $PY grpo2.py --ckpt ../rl_runs/lp_exp1_3000_ck --bundle ../sub_pick4c --family LiftPot --out ../rl_runs/lp_l2 \
  --iters 37200 --start-iter 37000 --workers 8 --learner-python $PY --shm /dev/shm/grpo_lp_l2 \
  --sigma 0.2 --adv mean --dyn-sample --dyn-oversample 1.5 --time-penalty 0.2 --time-window 0.25 --clip-high 0.28 \
  --init-weights ../rl_runs/lp_exp1_3000_ck/model.safetensors --save-successes --save-failures
