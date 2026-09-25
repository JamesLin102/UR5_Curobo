"""Evaluate a policy -- a checkpoint, the task's oracle, or random actions -- on any registered task.

    python scripts/lab/eval.py --task Grasp --policy scripts/grasp/policies/first.pt --set task.bank=eval
    python scripts/lab/eval.py --task Grasp --policy oracle --mapping --num_envs 4 --set task.bank=eval
    python scripts/lab/eval.py --task Grasp --policy random --episodes 200
    python scripts/lab/eval.py --task Grasp --policy <model_N.pt> --mapping --num_envs 1 --viz
    python scripts/lab/eval.py --task Grasp --policy <model_N.pt> --num_envs 1 --gui --port 5699

The env runs as its config says unless the flags change it (lab/app.py): for
grasp, the default is training's conditions -- the planner told the cylinders,
nothing rendered -- and --mapping is the real arm's: the wrist camera scans,
the map is built from it, perception estimates from the images. The gap
between the two for one policy is what training's shortcuts cost.

Counts episodes to --episodes and reports the success rate with a 95%
interval, the mean return, what the steps came to, and the totals the task
logs (extras["log"] entries ending in _total). Success and outcomes are the
task's extras["success"] and extras["outcome"] (lab/tasks/base.py).

--gui opens the Isaac Sim window; --viz serves the point cloud view of one
env (lab/viz.py). While a training run holds the default planner port, give
--port another one.
"""

import argparse
import math
import os
import sys
import time
from collections import Counter

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)

from lab import app as lab_app  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
lab_app.add_args(ap, task="Grasp")
ap.set_defaults(num_envs=4)
ap.add_argument("--policy", default="oracle", help="a model_N.pt checkpoint, 'oracle' or 'random'")
ap.add_argument("--episodes", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rl-device", default="cuda:0")
ap.add_argument("--no-servers", action="store_true", help="the planner servers are already running")
ap.add_argument("--gui", action="store_true", help="open the Isaac Sim window to watch (a few envs)")
ARGS = ap.parse_args()
ARGS.headless = not ARGS.gui
if "--device" not in sys.argv:
    ARGS.device = "cpu"


def log(msg):
    print(f"[eval] {msg}", flush=True)


APP = lab_app.launch(ARGS)
SERVERS = None

import gymnasium as gym  # noqa: E402

from lab import policy as lab_policy  # noqa: E402
from lab.viz import CellViz  # noqa: E402


def main():
    global SERVERS
    tid, cfg = lab_app.make_env_cfg(ARGS)
    cfg.seed = ARGS.seed
    if not ARGS.no_servers:
        SERVERS = lab_app.start_servers(cfg, log=log)
    env = gym.make(tid, cfg=cfg)
    u = env.unwrapped
    log(f"{tid}: {ARGS.num_envs} envs, mapping {'on' if cfg.cell.mapping else 'off'}, "
        f"policy {ARGS.policy}")
    act = lab_policy.make(env, ARGS.policy, ARGS.rl_device, log=log)
    obs, _ = env.reset(seed=ARGS.seed)
    viz = CellViz(env, ARGS.viz_env, ARGS.viz_port, ARGS.viz_stride, log=log) if ARGS.viz else None
    if viz is not None:
        viz.update()

    done, ok, t0 = 0, 0, time.time()
    returns, ret = [], [0.0] * u.num_envs
    outcomes, totals = Counter(), {}
    while done < ARGS.episodes:
        obs, rew, term, trunc, extras = env.step(act(obs))
        success = extras.get("success", [False] * u.num_envs)
        outcomes.update(extras.get("outcome", []))
        totals.update({k: v for k, v in extras.get("log", {}).items() if k.endswith("_total")})
        for e in range(u.num_envs):
            ret[e] += float(rew[e])
            if bool(term[e]) or bool(trunc[e]):
                done += 1
                ok += bool(success[e])
                returns.append(ret[e])
                ret[e] = 0.0
        if viz is not None:
            viz.update(notes=[f"last step: {extras.get('outcome', ['?'] * u.num_envs)[viz.e]}"])

    p = ok / done
    half = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / done)
    log(f"success {ok}/{done} = {100 * p:.0f}% (95% interval {100 * max(0, p - half):.0f}-"
        f"{100 * min(1, p + half):.0f}%), mean return {sum(returns) / len(returns):.3f}")
    log(f"steps came to: {dict(outcomes)}")
    if totals:
        log("totals: " + ", ".join(f"{k} {v:g}" for k, v in sorted(totals.items())))
    log(f"{time.time() - t0:.0f} s for {done} episodes over {ARGS.num_envs} envs")
    if viz is not None:
        log(f"the viewer stays up at {viz.viewer.url}; Ctrl-C to quit")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        viz.close()
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
