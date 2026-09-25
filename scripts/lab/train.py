"""Train any registered task with PPO (rsl_rl): the planner servers it needs, then learn.

    python scripts/lab/train.py --task Grasp --smoke                          # 2 envs, 3 iterations
    python scripts/lab/train.py --task Grasp --num_envs 64 --num-servers 1 --planner batch --run-name first
    python scripts/lab/train.py --task Grasp --resume logs/rsl_rl/grasp/<run>/model_250.pt
    tensorboard --logdir logs/rsl_rl

The task is a lab.tasks.TASKS entry with an "rsl_rl" runner config. Its env
config decides the rest -- scene, mapping, solver iterations -- unless a flag
or --set says otherwise (lab/app.py). The planner servers follow from that
config (lab.app.start_servers): with mapping off, --num-servers of them (one
per env by default), --planner batch to plan each server's envs in one go.
Physics runs on --device, cpu by default, which is faster than GPU PhysX at
these sizes; the networks on --rl-device.

Checkpoints (every save_interval iterations) and tensorboard go to
logs/rsl_rl/<experiment_name>/<time>[_run-name]/. Whatever a task puts in
extras["log"] is plotted there too.
"""

import argparse
import os
import sys
import time
from datetime import datetime

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)

from lab import app as lab_app  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
lab_app.add_args(ap, task="Grasp")
ap.set_defaults(num_envs=8)
ap.add_argument("--max-iterations", type=int, default=None, help="default: the runner config's")
ap.add_argument("--run-name", default="")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--rl-device", default="cuda:0")
ap.add_argument("--resume", default=None, help="a checkpoint to continue from")
ap.add_argument("--no-servers", action="store_true", help="the planner servers are already running")
ap.add_argument("--smoke", action="store_true", help="2 envs, 3 iterations: does it run")
ARGS = ap.parse_args()
ARGS.headless = True
if "--device" not in sys.argv:
    ARGS.device = "cpu"
if ARGS.smoke:
    ARGS.num_envs, ARGS.max_iterations = 2, ARGS.max_iterations or 3


def log(msg):
    print(f"[train] {msg}", flush=True)


APP = lab_app.launch(ARGS)
SERVERS = None

import gymnasium as gym  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from lab.policy import runner_cfg  # noqa: E402


def main():
    global SERVERS
    tid, env_cfg = lab_app.make_env_cfg(ARGS)
    env_cfg.seed = ARGS.seed
    agent = runner_cfg(tid, ARGS.rl_device)
    agent.seed = ARGS.seed
    agent.run_name = ARGS.run_name
    if ARGS.max_iterations is not None:
        agent.max_iterations = ARGS.max_iterations

    log_dir = os.path.join(ROOT, "logs", "rsl_rl", agent.experiment_name,
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                           + (f"_{ARGS.run_name}" if ARGS.run_name else ""))
    env_cfg.log_dir = log_dir
    if not ARGS.no_servers:
        SERVERS = lab_app.start_servers(env_cfg, log=log)
    env = gym.make(tid, cfg=env_cfg)
    log(f"{tid}: {ARGS.num_envs} envs, scene {env_cfg.cell.scene}, mapping "
        f"{'on' if env_cfg.cell.mapping else 'off'}, planner {env_cfg.cell.planner_mode}, "
        f"physics on {ARGS.device}, networks on {ARGS.rl_device}")
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)

    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=log_dir, device=agent.device)
    if ARGS.resume:
        runner.load(ARGS.resume)
        log(f"resumed from {ARGS.resume}")
    log(f"{agent.max_iterations} iterations x {agent.num_steps_per_env} steps x "
        f"{ARGS.num_envs} envs; logs in {log_dir}")
    t0 = time.time()
    runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=False)
    log(f"done in {time.time() - t0:.0f} s; checkpoints in {log_dir}")
    env.close()
    return True


if __name__ == "__main__":
    ok = False
    try:
        ok = main()
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        if SERVERS is not None:
            SERVERS.stop()
        sys.stdout.flush()
        os._exit(0 if ok else 1)
