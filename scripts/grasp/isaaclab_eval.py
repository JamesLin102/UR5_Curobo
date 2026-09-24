"""Evaluate a grasp policy -- or the oracle, or random actions -- on the eval bank.

    python scripts/grasp/isaaclab_eval.py --policy logs/rsl_rl/grasp/<run>/model_500.pt
    python scripts/grasp/isaaclab_eval.py --policy oracle --mode eval --num_envs 4
    python scripts/grasp/isaaclab_eval.py --policy random --episodes 200

--mode train: as training runs -- the planner told each environment's
    cylinders, the policy shown the truth through the error model. Fast.
--mode eval: as the real arm would -- the planner told nothing: the wrist
    camera scans at every reset, the map is built from it, the policy is
    shown grasp.perception's estimate from the scan's images, and the
    straight moves are checked against the map (docs/grasp_rl_plan.md §7).
    One mapping planner server per environment; rendering on; slower.

The difference between the two for the same policy is what the training
shortcuts cost. Reports the success rate with a 95% interval, what the
attempts came to, contacts, the cubes perception did not find, and the
returns home that had to be teleports.
"""

import argparse
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--policy", default="oracle", help="oracle, random, or a model_N.pt checkpoint")
ap.add_argument("--mode", choices=("train", "eval"), default="eval")
ap.add_argument("--num_envs", type=int, default=4)
ap.add_argument("--episodes", type=int, default=64)
ap.add_argument("--bank", default="eval")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-servers", action="store_true")
ap.add_argument("--rl-device", default="cuda:0")
AppLauncher.add_app_launcher_args(ap)
ARGS = ap.parse_args()
ARGS.headless = True
MAPPING = ARGS.mode == "eval"
if MAPPING:
    ARGS.enable_cameras = True
if ARGS.device == "cuda:0" and "--device" not in sys.argv:
    ARGS.device = "cpu"


def log(msg):
    print(f"[eval] {msg}", flush=True)


def start_servers(k):
    logfile = os.path.join(tempfile.gettempdir(), f"grasp_eval_servers_{os.getpid()}.log")
    cmd = [sys.executable, "-u", os.path.join(SCRIPTS, "planner_servers.py"), "--num", str(k),
           "--scene", "grasp"] + ([] if MAPPING else ["--no-mapping"])
    proc = subprocess.Popen(cmd, stdout=open(logfile, "w"), stderr=subprocess.STDOUT, cwd=ROOT,
                            start_new_session=True)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"planner servers exited; see {logfile}")
        if "[servers] all" in open(logfile).read():
            log(f"{k} planner server(s) ({'mapping' if MAPPING else 'told the cylinders'}); log {logfile}")
            return proc
        time.sleep(1)
    raise SystemExit(f"planner servers did not start; see {logfile}")


SERVERS = None if ARGS.no_servers else start_servers(ARGS.num_envs)
APP = AppLauncher(ARGS).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import lab  # noqa: E402,F401
from grasp import task as T  # noqa: E402
from lab.tasks import task_id  # noqa: E402


def make_policy(env):
    u = env.unwrapped
    if ARGS.policy == "random":
        return lambda obs: torch.rand((u.num_envs, T.ACT_DIM), device=u.device) * 2 - 1
    if ARGS.policy == "oracle":
        def oracle(obs):
            acts = []
            usable = u.bank.usable(u.task)
            for e in range(u.num_envs):
                feas = usable[u.layout[e]]
                face = 0 if feas[0] else 1
                acts.append(T.oracle_action(u.est[e].cube[2], face))
            return torch.tensor(np.array(acts), dtype=torch.float32, device=u.device)
        return oracle
    import importlib.metadata as md

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from grasp.lab_rl_cfg import GraspPPORunnerCfg
    agent = handle_deprecated_rsl_rl_cfg(GraspPPORunnerCfg(), md.version("rsl-rl-lib"))
    agent.device = ARGS.rl_device
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(ARGS.policy)
    policy = runner.get_inference_policy(device=agent.device)
    log(f"policy from {ARGS.policy}")

    def act(obs):
        with torch.inference_mode():
            # The networks may be on the GPU and the env's physics on the CPU.
            return policy(wrapped.get_observations().to(agent.device)).to(u.device).clamp(-1, 1)
    return act


def main():
    tid = task_id("Grasp", "ur5_robotiq")
    cfg = parse_env_cfg(tid, device=ARGS.device, num_envs=ARGS.num_envs)
    cfg.cell.mapping = MAPPING
    cfg.task.bank = ARGS.bank
    cfg.seed = ARGS.seed
    env = gym.make(tid, cfg=cfg)
    u = env.unwrapped
    log(f"{tid}: {ARGS.num_envs} envs, mode {ARGS.mode}, bank {ARGS.bank}, policy {ARGS.policy}")
    obs, _ = env.reset(seed=ARGS.seed)
    policy = make_policy(env)

    done, ok, contacts, t0 = 0, 0, 0, time.time()
    outcomes = Counter()
    while done < ARGS.episodes:
        obs, _, term, trunc, extras = env.step(policy(obs))
        for e, a in enumerate(extras["attempt"]):
            outcomes[a["outcome"]] += 1
            contacts += a["contact"] > u.task.contact_force
            if bool(term[e]) or bool(trunc[e]):
                done += 1
                ok += a["success"]
    p = ok / done
    half = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / done)
    log(f"success {ok}/{done} = {100 * p:.0f}% (95% interval {100 * max(0, p - half):.0f}-"
        f"{100 * min(1, p + half):.0f}%)")
    log(f"attempts: {dict(outcomes)}; contacts {contacts}; returns home teleported {u.stuck}"
        + (f"; cubes perception did not find {u.unseen}" if MAPPING else ""))
    log(f"{time.time() - t0:.0f} s for {done} episodes over {ARGS.num_envs} envs")
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
