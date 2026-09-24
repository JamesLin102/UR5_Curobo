"""Train the grasp policy with PPO: Isaac-Grasp-Ur5Robotiq-v0, rsl_rl.

    python scripts/grasp/isaaclab_train.py --num_envs 8                  # starts its own servers
    python scripts/grasp/isaaclab_train.py --num_envs 16 --num-servers 4 --max-iterations 2000
    python scripts/grasp/isaaclab_train.py --smoke                       # 2 envs, 3 iterations

Training mode (docs/grasp_rl_plan.md §7): the planner is told each
environment's cylinders, nothing is rendered, and the policy sees the
synthesised perception estimate. Physics runs on --device (cpu by default,
which is faster at these sizes -- plan §9); the networks train on --rl-device.

Unless --no-servers, it starts planner_servers.py itself (--scene grasp
--no-mapping), one per --num-servers (default: one per env), and stops them
at the end. Logs and checkpoints go to logs/rsl_rl/grasp/<time>[_run-name]/,
tensorboard included; extras["log"] puts the success rate and what each
attempt came to in there as well.
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--num_envs", type=int, default=8)
ap.add_argument("--num-servers", type=int, default=0, help="0: one per env")
ap.add_argument("--no-servers", action="store_true", help="planner servers are already running")
ap.add_argument("--max-iterations", type=int, default=None)
ap.add_argument("--bank", default="train")
ap.add_argument("--run-name", default="")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--rl-device", default="cuda:0")
ap.add_argument("--resume", default=None, help="a checkpoint to start from")
ap.add_argument("--smoke", action="store_true", help="2 envs, 3 iterations: does it run")
AppLauncher.add_app_launcher_args(ap)
ARGS = ap.parse_args()
ARGS.headless = True
if ARGS.device == "cuda:0" and "--device" not in sys.argv:
    ARGS.device = "cpu"
if ARGS.smoke:
    ARGS.num_envs, ARGS.max_iterations = 2, ARGS.max_iterations or 3


def log(msg):
    print(f"[train] {msg}", flush=True)


def start_servers(k):
    logfile = os.path.join(tempfile.gettempdir(), f"grasp_train_servers_{os.getpid()}.log")
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(SCRIPTS, "planner_servers.py"), "--num", str(k),
         "--scene", "grasp", "--no-mapping"],
        stdout=open(logfile, "w"), stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"planner servers exited; see {logfile}")
        if "[servers] all" in open(logfile).read():
            log(f"{k} planner server(s) up; log {logfile}")
            return proc
        time.sleep(1)
    raise SystemExit(f"planner servers did not start; see {logfile}")


K = ARGS.num_servers or ARGS.num_envs
SERVERS = None if ARGS.no_servers else start_servers(K)
APP = AppLauncher(ARGS).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import lab  # noqa: E402,F401  (registers the ids)
from grasp.lab_rl_cfg import GraspPPORunnerCfg  # noqa: E402
from lab.tasks import task_id  # noqa: E402


def main():
    import importlib.metadata as md

    tid = task_id("Grasp", "ur5_robotiq")
    env_cfg = parse_env_cfg(tid, device=ARGS.device, num_envs=ARGS.num_envs)
    env_cfg.cell.num_servers = ARGS.num_servers
    env_cfg.task.bank = ARGS.bank
    env_cfg.seed = ARGS.seed

    agent = GraspPPORunnerCfg()
    agent.seed = ARGS.seed
    agent.device = ARGS.rl_device
    agent.run_name = ARGS.run_name
    if ARGS.max_iterations is not None:
        agent.max_iterations = ARGS.max_iterations
    agent = handle_deprecated_rsl_rl_cfg(agent, md.version("rsl-rl-lib"))

    log_dir = os.path.join(ROOT, "logs", "rsl_rl", agent.experiment_name,
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                           + (f"_{ARGS.run_name}" if ARGS.run_name else ""))
    env_cfg.log_dir = log_dir
    env = gym.make(tid, cfg=env_cfg)
    log(f"{tid}: {ARGS.num_envs} envs, {K} planner server(s), physics on {ARGS.device}, "
        f"networks on {ARGS.rl_device}, bank {ARGS.bank} ({len(env.unwrapped.bank)} layouts)")
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
    log(f"stuck returns home over the run: {env.unwrapped.stuck}")
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
        if SERVERS is not None and SERVERS.poll() is None:
            os.killpg(SERVERS.pid, signal.SIGINT)
            try:
                SERVERS.wait(40)
            except subprocess.TimeoutExpired:
                os.killpg(SERVERS.pid, signal.SIGKILL)
        sys.stdout.flush()
        os._exit(0 if ok else 1)
