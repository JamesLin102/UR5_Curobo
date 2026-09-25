"""End-to-end check of Isaac-Grasp-Ur5Robotiq-v0, driven by an oracle, before any learning.

    python tools/check_grasp_env.py                          # 4 envs, 12 episodes
    python tools/check_grasp_env.py --num-envs 8 --episodes 40 --num-servers 2
    python tools/check_grasp_env.py --bank eval --no-noise

One command: it starts its own planner servers (--scene grasp --no-mapping:
training mode, each environment's cylinders told to the planner), makes the
environment the way Isaac Lab's scripts do (parse_env_cfg, gym.make), and
plays whole episodes with an oracle that knows from the bank which of the two
face-square grips works and aims it at the ESTIMATED cube -- with the same
error model the policy gets, so what it does not manage is what perception
error costs. Then it checks that Isaac Lab's rsl_rl wrapper takes the env.

Reports the oracle's success rate (first attempt and within the episode),
what each attempt came to (grasp.task.OUTCOMES), contacts, the returns home
that had to be teleports, where the gripper stopped holding and closed on
nothing (the holding threshold), and how long a step takes -- the number that
decides how long training will be.

Pass: every episode ends, the oracle succeeds in most (--min-success), no
return home was a teleport, and the wrapper accepts the env.
"""

import argparse
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from isaaclab.app import AppLauncher  # noqa: E402

import planner_servers  # noqa: E402
from lab import app as lab_app  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--num-envs", type=int, default=4)
ap.add_argument("--num-servers", type=int, default=0, help="0: one per env")
ap.add_argument("--batch", action="store_true",
                help="plan every env of a server in one batch (planner_server --batch)")
ap.add_argument("--episodes", type=int, default=12)
ap.add_argument("--bank", default="train")
ap.add_argument("--no-noise", action="store_true", help="the oracle sees the truth")
ap.add_argument("--min-success", type=float, default=0.8)
ap.add_argument("--policy", choices=("oracle", "random"), default="oracle",
                help="random: exercise every failure path (no success bar)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--solver-iterations", type=int, nargs=2, default=None,
                help="PhysX position/velocity iterations (default: rig.ROBOTS')")
ap.add_argument("--one-grip-fraction", type=float, default=None,
                help="share of episodes from layouts with one working grip (TaskCfg)")
AppLauncher.add_app_launcher_args(ap)
ARGS = ap.parse_args()
ARGS.headless = True
if ARGS.device == "cuda:0" and "--device" not in sys.argv:
    ARGS.device = "cpu"      # the faster at these sizes; --device cuda:0 to override


def log(msg):
    print(f"[grasp-env] {msg}", flush=True)


K = ARGS.num_servers or ARGS.num_envs
BATCH = ["--batch", str(-(-ARGS.num_envs // K))] if ARGS.batch else []
SERVERS = planner_servers.launch(K, "--scene", "grasp", "--no-mapping", *BATCH, log=log)
APP = lab_app.launch(ARGS)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import lab  # noqa: E402,F401  (registers the ids)
from grasp import lab_policy  # noqa: E402
from grasp import task as T  # noqa: E402
from lab.tasks import task_id  # noqa: E402


def main():
    tid = task_id("Grasp", "ur5_robotiq")
    cfg = parse_env_cfg(tid, device=ARGS.device, num_envs=ARGS.num_envs)
    cfg.cell.num_servers = ARGS.num_servers
    if ARGS.solver_iterations:
        cfg.cell.solver_iterations = tuple(ARGS.solver_iterations)
    if ARGS.batch:
        cfg.cell.planner_mode = "batch"
    cfg.task.bank = ARGS.bank
    cfg.task.noise = not ARGS.no_noise
    if ARGS.one_grip_fraction is not None:
        cfg.task.one_grip_fraction = ARGS.one_grip_fraction
    cfg.seed = ARGS.seed
    t0 = time.time()
    env = gym.make(tid, cfg=cfg)
    u = env.unwrapped
    log(f"{tid}: {u.num_envs} envs, {K} planner server(s), bank {ARGS.bank} "
        f"({len(u.bank)} layouts), built in {time.time() - t0:.0f} s")
    act = lab_policy.make(env, ARGS.policy, log=log)
    env.reset(seed=ARGS.seed)

    episodes, first_try, successes, steps = 0, 0, 0, 0
    outcomes, contacts = Counter(), 0
    held, empty = [], []
    step_times = []
    while episodes < ARGS.episodes:
        t = time.time()
        _, rew, term, trunc, extras = env.step(act(None))
        step_times.append(time.time() - t)
        steps += 1
        for e, a in enumerate(extras["attempt"]):
            outcomes[a["outcome"]] += 1
            contacts += a["contact"] > u.task.contact_force
            if a["contact"] > u.task.contact_force:
                log(f"  contact in env {e}: layout {a['layout']}, {a['contact']:.0f} N, "
                    f"{'lifted' if a['success'] else a['outcome']}, cube moved "
                    f"{1000 * a['moved']:.0f} mm, rose {1000 * a['rise']:.0f} mm, why {a['why']}")
            if a["success"]:
                held.append(a["gripper"])
            elif a["outcome"] == "empty":
                empty.append(a["gripper"])
            if bool(term[e]) or bool(trunc[e]):
                episodes += 1
                successes += a["success"]
                first_try += a["success"] and a["attempt"] == 0
        log(f"step {steps}: {step_times[-1]:.1f} s; " + ", ".join(
            f"env{e} {a['outcome']}{'+' if a['success'] else ''}" for e, a in enumerate(extras["attempt"])))

    per_attempt = np.mean(step_times) / u.num_envs
    prof = u.cell.runner.profile
    total = sum(v for k, v in prof.items() if k in ("planner", "physics", "ops"))
    log("where run() spent its time: " + ", ".join(
        f"{k} {v:.1f} s ({100 * v / total:.0f}%)" for k, v in prof.items()
        if k in ("planner", "physics", "ops")) + f"; {int(prof['ticks'])} ticks, "
        f"{1000 * prof['physics'] / max(prof['ticks'], 1):.1f} ms each "
        f"(write {1000 * prof['tick.write'] / max(prof['ticks'], 1):.1f}, "
        f"step {1000 * prof['tick.step'] / max(prof['ticks'], 1):.1f}, "
        f"read {1000 * prof['tick.read'] / max(prof['ticks'], 1):.1f})")
    log(f"{ARGS.policy}: {successes}/{episodes} episodes succeeded ({100 * successes / episodes:.0f}%), "
        f"{first_try} on the first attempt")
    log(f"attempts: {dict(outcomes)}; contacts {contacts}; returns home teleported {u.stuck}")
    if held:
        log(f"gripper holding: {min(held):+.3f}..{max(held):+.3f} rad")
    if empty:
        log(f"gripper closed on nothing: {min(empty):+.3f}..{max(empty):+.3f} rad "
            f"(holding threshold {u.task.hold_below:+.2f})")
    log(f"a step (one attempt in every env): {np.mean(step_times):.1f} s mean, "
        f"{np.max(step_times):.1f} s worst; {1 / per_attempt:.2f} attempts/s over {u.num_envs} envs")

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    wrapped = RslRlVecEnvWrapper(env)
    obs = wrapped.get_observations()
    shapes = {k: tuple(v.shape) for k, v in obs.items()}
    log(f"rsl_rl wrapper: observations {shapes}, {wrapped.num_actions} actions")

    bar = ARGS.min_success if ARGS.policy == "oracle" else 0.0
    ok = (successes / episodes >= bar and u.stuck == 0
          and shapes.get("policy") == (u.num_envs, T.OBS_DIM)
          and shapes.get("critic") == (u.num_envs, T.STATE_DIM))
    log("RESULT " + ("PASS" if ok else "FAIL"))
    env.close()
    return ok


if __name__ == "__main__":
    ok = False
    try:
        ok = main()
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        SERVERS.stop()
        sys.stdout.flush()
        os._exit(0 if ok else 1)
