"""Calibration for a scripted LiftPot expert (Oct 24): replay successful TRAINING-seed episodes of the incumbent exactly and,
at each gripper's first grasp of the pot, record the gripper body's pose in the pot's frame (position, yaw relative to the
pot, roll/pitch), the step, and the pot's height at success.
Usage: python lp_calib.py LOGDIR N [OUT.json]   (LOGDIR/<run>/LiftPot_PositionAndOrientation/episode_*.json; OUT: every grasp)"""
import glob
import json
import sys

import numpy as np
from pyquaternion import Quaternion
from roboeval.challenge.submission import Interface, make_env

logdir, N = sys.argv[1], int(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else None
lib = []
env = make_env("LiftPot", "PositionAndOrientation", Interface(absolute=False, ee=True, control_frequency=20))
u = getattr(env, "unwrapped", env)
sides = list(u.robot.grippers)


def ypr(q):
    w, x, y, z = q
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    return yaw, pitch, roll


rows = {0: [], 1: []}
n = 0
for jf in sorted(glob.glob(f"{logdir}/*/LiftPot_PositionAndOrientation/episode_*.json")):
    j = json.load(open(jf))
    if j["termination"] != "success":
        continue
    a = np.load(jf[:-5] + ".npz")["actions"]
    env.reset(seed=j["seed"])
    pot = u.kitchenpot.body
    p0, q0 = np.array(pot.get_position()), Quaternion(pot.get_quaternion())
    got, succ_t, pre = {}, None, {}
    for t, x in enumerate(a):
        for k, sd in enumerate(sides):  # pose when the close command starts (before the fingers move the pot)
            if k not in pre and x[12 + k] > 0.5:
                g = u.robot.grippers[sd].body
                pp_, pq_ = np.array(pot.get_position()), Quaternion(pot.get_quaternion())
                rq_ = pq_.inverse * Quaternion(g.get_quaternion())
                pre[k] = (pq_.inverse.rotate(np.array(g.get_position()) - pp_), ypr(rq_.elements)[0], ypr(g.get_quaternion())[0], rq_.elements)
        env.step(x)
        pp, pq = np.array(pot.get_position()), Quaternion(pot.get_quaternion())
        for k, s in enumerate(sides):
            if k not in got and u.robot.is_gripper_holding_object(u.kitchenpot, s):
                g = u.robot.grippers[s].body
                gp, gq = np.array(g.get_position()), Quaternion(g.get_quaternion())
                rel = pq.inverse.rotate(gp - pp)
                rq = pq.inverse * gq
                got[k] = (t, rel, ypr(rq.elements), ypr(gq.elements))
        if env.success:
            succ_t = t
            break
    n += 1
    msg = f"seed {j['seed']} logged {j['steps']} replay success at {succ_t + 1 if succ_t is not None else None} | pot yaw {ypr(q0.elements)[0]:+.1f} pos {np.round(p0, 3)}"
    for k in (0, 1):
        if k in got:
            t, rel, rel_ypr, g_ypr = got[k]
            rows[k].append(np.r_[rel, rel_ypr, g_ypr, t])
            if k in pre:
                lib.append({"seed": j["seed"], "side": k, "t": int(t), "rel": [float(v) for v in pre[k][0]], "rel_yaw_deg": float(pre[k][1]),
                            "pot_yaw_deg": float(ypr(q0.elements)[0]), "world_yaw_deg": float(pre[k][2]),
                            "hold_rel": [float(v) for v in rel], "rel_quat": [float(v) for v in pre[k][3]]})
            msg += f" | {'LR'[k]} t {t} rel {np.round(rel * 100, 1)} cm rel ypr {np.round(rel_ypr, 0)}"
    print(msg, flush=True)
    if n >= N:
        break
for k in (0, 1):
    r = np.array(rows[k])
    if len(r):
        print(f"{'LR'[k]}: {len(r)} grasps; rel pos median {np.round(np.median(r[:, :3], 0) * 100, 1)} cm, IQR x {np.round(np.percentile(r[:, 0], [25, 75]) * 100, 1)} y {np.round(np.percentile(r[:, 1], [25, 75]) * 100, 1)} z {np.round(np.percentile(r[:, 2], [25, 75]) * 100, 1)}; "
              f"rel yaw/pitch/roll median {np.round(np.median(r[:, 3:6], 0), 0)}; world yaw/pitch/roll median {np.round(np.median(r[:, 6:9], 0), 0)}; grasp step median {np.median(r[:, 9]):.0f}")
if out:
    json.dump(lib, open(out, "w"), indent=1)
    print("wrote", len(lib), "grasps to", out)
