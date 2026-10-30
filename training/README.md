# Training the LiftPot model

LiftPot is the bundle's one learned policy. It was built in four stages; each stage starts from the previous one's
checkpoint.

| # | Stage | Code | Data | Result |
|---|---|---|---|---|
| 1 | ACT, behaviour cloning | `act/e2_queue_L30k.sh` (LeRobot 0.4.4 `lerobot-train`, chunk 16, batch 32, 30k steps); data from `data/prep_all_variants_5090.py` | all 421 released LiftPot demonstrations (all four variants) | step 30000 |
| 2 | GRPO v1, two runs | `grpo_v1/grpo.py`; launch: `grpo_v1/launch/grpo_5090_liftpot.sh`, then `grpo_lp145_r2.sh` from its round 145 | online rollouts on training seeds, 9,280 + 9,600 | round 1150 |
| 3 | Scripted-expert fine-tune | data: `sil/lp_exp1_data.sh` (`experts/liftpot/lp_gen.py`, `lp_expert.py`, grasp library `lp_grasps.json` from `lp_calib.py`), converted by `sil/to_lerobot2.py`; training: `sil/ft_expert.sh` | 355 successful expert episodes (of 420, training seeds 410000-410419) | step 3000 |
| 4 | GRPO v2 | `grpo_v2/grpo2.py`; launch: `grpo_v2/launch/grpo_chain29_lp.sh` | online rollouts on training seeds, 9,600 | round 37100, shipped |

**GRPO** here: each round draws fresh training seeds and runs 8 rollouts per seed, with Gaussian exploration noise
(sigma 0.2) on the normalized actions. The reward is the official task score of the episode plus 0.5 on success
(minus 0.25 if the simulation went unstable). v1 uses the advantage (R - group mean) / group std, v2 uses R - group mean;
the update is PPO-style clipping with a KL penalty to the starting model. Rollout workers run on the CPU
(`rollout.py`), the update in a separate learner process on the GPU (`learner.py`, `learner2.py`). v2 (as launched
for stage 4) adds dynamic sampling (only seeds with both successes and failures among their 8 rollouts train), a
penalty for episodes that run into the last 25% of the step budget, and a higher upper clip (0.28).
`act_common.py` loads a checkpoint exactly as the submitted `inference.py` does, using the modules in `policy_v2/`.

**Seeds.** Rollouts used training seeds (10,000 and above). Checkpoints were chosen on our dev seeds (3000-3031) and
validation seeds (2000-2049). The official seeds (1000-1049) were never used for training or selection.

**Simulator state.** The scripted expert reads the pot's and grippers' poses from the simulator to generate its
episodes, and the GRPO reward is computed by the evaluator from simulator state. The trained policy itself sees only
the observations.

**Logs.** `logs/sft/train_LiftPot.log` (stage 1), `logs/grpo/p6_LiftPot/` and `logs/grpo/lp145_r2_5090/` (stage 2),
`logs/sft/lp_exp1.log` (stage 3), `logs/grpo/lp_l2/` (stage 4). Each GRPO `log.jsonl` has one line per round
(success rate, task score, reward). The rollouts of stage 4 are saved (seed + executed actions) in the dataset
[Sa74ll/LocalModels-ACT-bimanual-rollouts](https://huggingface.co/datasets/Sa74ll/LocalModels-ACT-bimanual-rollouts)
(run `LiftPot/grpo_02`).

These are the scripts as they ran. They hard-code our machines' folder layout (names with `5090` or `tower` tell which
machine ran them), so paths need editing before reuse. Training used the unmodified LeRobot 0.4.4 package; RoboEval
at the challenge tag provides the environments.
