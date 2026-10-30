#!/usr/bin/env bash
# Oct 24 19:05: LiftPot synthetic demonstrations from the scripted expert (tag lp_exp1), data only (training is started
# separately once a GPU is free). The expert (xtask/lp_expert.py af79de0e) reads the pot's and grippers' poses in simulation
# and acts through the policy's interface; each arm copies one real grasp of the incumbent (pose in the pot's frame and full
# orientation, from xtask/lp_grasps.json: 520 grasps of 300 successful TRAINING-seed episodes replayed exactly, nearest pot
# yaw), then both lift together: 38/40 on unseen training seeds, median 56 steps (incumbent ~90%, 70-80 steps). Episodes
# on TRAINING seeds 410000-410419, successes only, expert-only dataset with glp1150's normalization; the pickles are deleted
# once the dataset is built (standing OK for pipeline intermediates).
set -uo pipefail
cd ~/roboeval_challenge/orch/sil
TAG=lp_exp1
PY=$HOME/miniconda3/envs/roboeval/bin/python
PYT=$HOME/miniconda3/envs/roboeval-train/bin
free_gb() { df --output=avail -BG ~ | tail -1 | tr -dc 0-9; }
[ "$(free_gb)" -ge 25 ] || { echo "$(date +%T) only $(free_gb) GB free, need 25"; exit 1; }
echo "$(date +%T) GENERATE $TAG ($(free_gb) GB free)"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_CBWR=COMPATIBLE
$PY ../xtask/lp_gen.py $TAG 410000 420 --bundle ../sub_pick4c --workers 8 2>&1 | grep -v "converge\|GLFW\|warn"
echo "$(date +%T) $(ls $TAG/*.pkl | wc -l) expert episodes, $(du -sh $TAG | cut -f1)"
[ "$(free_gb)" -ge 15 ] || { echo "$(date +%T) only $(free_gb) GB free before the conversion, need 15"; exit 1; }
echo "$(date +%T) CONVERT $TAG (expert-only)"
HF_DATASETS_CACHE=/dev/shm/rbe_hfc_$TAG $PYT/python ../rl/to_lerobot2.py data/LiftPotALL $TAG ${TAG}_new ${TAG}_DATA \
  --shards 8 --no-demos > conv_$TAG.out 2>&1
rc=$?; rm -rf /dev/shm/rbe_hfc_$TAG; echo "CONV_EXIT $rc" >> conv_$TAG.out; grep "MERGED\|skipped" conv_$TAG.out
[ $rc = 0 ] || exit 1
$PY - ../rl_runs/glp1150_ref ${TAG}_DATA <<'PYEOF'
import json, shutil, sys
from pathlib import Path
from safetensors.torch import load_file
init, data = Path(sys.argv[1]), Path(sys.argv[2])
st = load_file(str(init / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
p, orig = data / "meta" / "stats.json", data / "meta" / "stats.orig.json"
if not orig.exists():
    shutil.copy2(p, orig)
s = json.loads(orig.read_text())
for feat in ("action", "observation.state"):
    for m in ("mean", "std"):
        assert len(s[feat][m]) == st[f"{feat}.{m}"].numel()
        s[feat][m] = st[f"{feat}.{m}"].tolist()
p.write_text(json.dumps(s, indent=4))
print("normalization stats of", init, "->", p)
PYEOF
[ -f ${TAG}_DATA/meta/info.json ] && rm -rf -- ./$TAG ./${TAG}_new && echo "$(date +%T) deleted the episode pickles and shards ($(free_gb) GB free)"
echo "$(date +%T) DATA_DONE $TAG $(du -sh ${TAG}_DATA | cut -f1)"
