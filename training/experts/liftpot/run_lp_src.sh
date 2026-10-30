#!/usr/bin/env bash
# Oct 24 14:50: LiftPot source rollouts for a mirror fine-tune. The incumbent (GRPO 1150, e8c70ddf; pick13's LiftPot) succeeds
# 95% with the pot on the robot's right third and 75% on the left third (pick11 + pick12 whole-bundle logs, 200 episodes).
# Deterministic evaluator runs on TRAINING seeds 330000-330299 (never used before), 3 processes, logs with actions.
cd ~/local_models/work/xtask
export MUJOCO_GL=egl OMP_NUM_THREADS=2 OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_CBWR=COMPATIBLE
for k in 0 1 2; do
  ~/miniforge3/envs/roboeval/bin/python -m roboeval.challenge.evaluate --submission ../pick13/sub_pick13 --tasks LiftPot \
    --seeds-file seeds_lp_$k.json --workers 1 --gpus 0 --output-dir lp_src_$k --overwrite > lp_src_$k.log 2>&1 &
done
wait
echo LP_SRC_DONE
