"""Evaluate a grasp policy -- or the oracle, or random actions -- on the eval bank.

    python scripts/grasp/isaaclab_eval.py --policy scripts/grasp/policies/first.pt
    python scripts/grasp/isaaclab_eval.py --policy oracle --mode eval --num_envs 4
    python scripts/grasp/isaaclab_eval.py --policy random --episodes 200
    python scripts/grasp/isaaclab_eval.py --policy scripts/grasp/policies/first.pt --num_envs 1 --viz
    python scripts/grasp/isaaclab_eval.py --policy <model_N.pt> --mode train --num_envs 1 --gui --port 5699

--mode train: as training runs -- the planner told each environment's
    cylinders, the policy shown the truth through the error model. Fast.
--mode eval: as the real arm would -- the planner told nothing: the wrist
    camera scans at every reset, the map is built from it, the policy is
    shown grasp.perception's estimate from the scan's images, and the
    straight moves are checked against the map.
    One mapping planner server per environment; rendering on; slower.

The difference between the two for the same policy is what the training
shortcuts cost. Reports the success rate with a 95% interval, what the
attempts came to, contacts, the cubes perception did not find, and the
returns home that had to be teleports.
"""

import argparse
import math
import os
import sys
import time
from collections import Counter

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)

from isaaclab.app import AppLauncher  # noqa: E402

import planner_servers  # noqa: E402
from lab import app as lab_app  # noqa: E402
from lab import viz as lab_viz  # noqa: E402  (needs nothing from Isaac)

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--policy", default="oracle", help="oracle, random, or a model_N.pt checkpoint")
ap.add_argument("--mode", choices=("train", "eval"), default="eval")
ap.add_argument("--num_envs", type=int, default=4)
ap.add_argument("--episodes", type=int, default=64)
ap.add_argument("--bank", default="eval")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-servers", action="store_true")
ap.add_argument("--rl-device", default="cuda:0")
ap.add_argument("--port", type=int, default=None,
                help="first planner server port (default rig.PORT); another one while training runs")
ap.add_argument("--gui", action="store_true", help="open the Isaac Sim window to watch (a few envs)")
lab_viz.add_args(ap)
AppLauncher.add_app_launcher_args(ap)
ARGS = ap.parse_args()
ARGS.headless = not ARGS.gui
MAPPING = ARGS.mode == "eval"
if MAPPING:
    ARGS.enable_cameras = True
if ARGS.device == "cuda:0" and "--device" not in sys.argv:
    ARGS.device = "cpu"


def log(msg):
    print(f"[eval] {msg}", flush=True)


SERVERS = None if ARGS.no_servers else planner_servers.launch(
    ARGS.num_envs, "--scene", "grasp", *([] if MAPPING else ["--no-mapping"]),
    port=ARGS.port or planner_servers.PORT, log=log)
APP = lab_app.launch(ARGS)

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import lab  # noqa: E402,F401
from grasp import lab_policy  # noqa: E402
from grasp import viz as grasp_viz  # noqa: E402
from lab.tasks import task_id  # noqa: E402


def show(viz, u, extras=None):
    """The viewer on --viz-env: the scan and the estimate the next attempt will use."""
    e = viz.e
    notes = []
    if extras is not None:
        a = extras["attempt"][e]
        notes.append(f"last attempt: {a['outcome']}{' -- lifted' if a['success'] else ''} "
                     f"(the view is the next episode's)")
    views = u.scan_views[e]
    lines = viz.update(views=views or None,
                       labels=[f"scan{i}" for i in range(len(views))] if views else None,
                       notes=notes)
    cube = u._cube_xyyaw(u.cell.observe([e]).objects[u.cube_name][0])
    lines += grasp_viz.show_estimate(viz.viewer, u.est[e], cube, u.cylinders[e])
    viz.viewer.set_status("  \n".join(lines))


def main():
    tid = task_id("Grasp", "ur5_robotiq")
    cfg = parse_env_cfg(tid, device=ARGS.device, num_envs=ARGS.num_envs)
    cfg.cell.mapping = MAPPING
    if ARGS.port is not None:
        cfg.cell.port = ARGS.port
    cfg.task.bank = ARGS.bank
    cfg.seed = ARGS.seed
    env = gym.make(tid, cfg=cfg)
    u = env.unwrapped
    log(f"{tid}: {ARGS.num_envs} envs, mode {ARGS.mode}, bank {ARGS.bank}, policy {ARGS.policy}")
    obs, _ = env.reset(seed=ARGS.seed)
    policy = lab_policy.make(env, ARGS.policy, ARGS.rl_device, log=log)
    viz = lab_viz.CellViz(env, ARGS.viz_env, ARGS.viz_port, ARGS.viz_stride, log=log) \
        if ARGS.viz else None
    if viz is not None:
        show(viz, u)

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
        if viz is not None:
            show(viz, u, extras)
    p = ok / done
    half = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / done)
    log(f"success {ok}/{done} = {100 * p:.0f}% (95% interval {100 * max(0, p - half):.0f}-"
        f"{100 * min(1, p + half):.0f}%)")
    log(f"attempts: {dict(outcomes)}; contacts {contacts}; returns home teleported {u.stuck}"
        + (f"; cubes perception did not find {u.unseen}" if MAPPING else ""))
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
