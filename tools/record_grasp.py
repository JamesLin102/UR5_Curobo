"""Record the grasp policy on chosen layouts: Isaac Lab on the left, the point cloud view on the right.

    python tools/record_grasp.py --layouts 51 139 --dry-run    # which of these does it lift?
    python tools/record_grasp.py --layouts 51 139              # then record them

Evaluation mode, as the real arm would run: each layout starts with the wrist
camera's scan, the map and grasp.perception's estimate come from it, and the
policy (--policy, default the first run's last checkpoint) makes one attempt.
Starts its own planner server (mapping on). Writes --name.mp4 (real time) and
--name.gif (2x speed) into --out; the right view carries the perception layer
too -- the estimate against the truth. The video stops where the last
layout's episode ends.

--dry-run records nothing: it runs each layout once and says what the attempt
came to, to pick layouts the policy lifts.
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from lab import app as lab_app  # noqa: E402

FIRST = os.path.join(ROOT, "logs", "rsl_rl", "grasp", "2026-09-24_21-34-40_first", "model_499.pt")

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
lab_app.add_args(ap, task="Grasp")
ap.set_defaults(scene="grasp")
ap.add_argument("--layouts", type=int, nargs="+", required=True, help="bank rows, in order")
ap.add_argument("--bank", default="eval")
ap.add_argument("--policy", default=FIRST, help="a model_N.pt checkpoint, or 'oracle'")
ap.add_argument("--rl-device", default="cuda:0")
ap.add_argument("--dry-run", action="store_true", help="run the layouts, record nothing")
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "media"))
ap.add_argument("--name", default="grasp")
ap.add_argument("--every", type=int, default=2, help="physics steps per frame (60 Hz / 2 = 30 fps)")
ap.add_argument("--size", default="800x600", help="each view's WxH")
ap.add_argument("--eye", type=float, nargs=3, default=(1.4, 0.85, 1.0),
                help="both views' camera position, env-local")
ap.add_argument("--lookat", type=float, nargs=3, default=(0.4, 0.0, 0.24))
ARGS = ap.parse_args()
ARGS.headless, ARGS.viz = True, not ARGS.dry_run
if "--device" not in sys.argv:
    ARGS.device = "cpu"
W, H = (int(v) for v in ARGS.size.split("x"))


def say(msg):
    print(f"[record] {msg}", flush=True)


def start_server():
    log = os.path.join(tempfile.gettempdir(), f"record_grasp_server_{os.getpid()}.log")
    proc = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "scripts", "planner_server.py"),
                             "--scene", "grasp"], stdout=open(log, "w"),
                            stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"planner server exited; see {log}")
        if "listening" in open(log).read():
            return proc
        time.sleep(1)
    raise SystemExit(f"planner server did not start; see {log}")


os.makedirs(ARGS.out, exist_ok=True)
SERVER = start_server()
APP = lab_app.launch(ARGS)
signal.signal(signal.SIGINT, signal.default_int_handler)   # see grasp/isaaclab_eval.py

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from grasp import task as T  # noqa: E402
from grasp import viz as grasp_viz  # noqa: E402


def make_policy(env):
    """obs -> action, as isaaclab_eval.py does it."""
    u = env.unwrapped
    if ARGS.policy == "oracle":
        def oracle(obs):
            feas = u.bank.usable(u.task)[u.layout[0]]
            return torch.tensor(np.array([T.oracle_action(u.est[0].cube[2], 0 if feas[0] else 1)]),
                                dtype=torch.float32, device=u.device)
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

    def act(obs):
        with torch.inference_mode():
            return policy(wrapped.get_observations().to(agent.device)).to(u.device).clamp(-1, 1)
    return act


def main():
    tid, cfg = lab_app.make_env_cfg(ARGS)
    cfg.task.bank = ARGS.bank
    cfg.viewer.resolution = (W, H)
    cfg.viewer.eye, cfg.viewer.lookat = tuple(ARGS.eye), tuple(ARGS.lookat)
    env = gym.make(tid, cfg=cfg, render_mode=None if ARGS.dry_run else "rgb_array")
    u = env.unwrapped
    rows = list(ARGS.layouts)
    state = {"resets": 0}
    viz = rec = None
    if not ARGS.dry_run:
        from lab.viz import CellViz
        from recording import Recorder
        viz = CellViz(env, 0, ARGS.viz_port, ARGS.viz_stride, log=say)
        rec = Recorder(env, viz, ARGS.out, ARGS.name, (W, H), ARGS.every, log=say)

    # Every episode's reset goes through here: the video stops at the one after
    # the last layout, and the perception layer follows each new estimate.
    reset_cells = u.reset_cells

    def reset_and_show(env_ids, options):
        state["resets"] += 1
        k = state["resets"] - 1
        if rec is not None and k >= len(rows):
            rec.paused = True
        if viz is not None:
            viz.viewer.clear("perception")
        reset_cells(env_ids, options)
        if viz is not None and k < len(rows):
            cube = u._cube_xyyaw(u.cell.observe([0]).objects[u.cube_name][0])
            grasp_viz.show_estimate(viz.viewer, u.est[0], cube, u.cylinders[0])
    u.reset_cells = reset_and_show

    policy = make_policy(env)
    state["resets"] = 0            # rsl_rl's wrapper reset the env once already, above
    if rec is not None:
        rec.start()
    results = []
    try:
        obs, _ = env.reset(options={"layouts": [rows[0]]})
        for i, row in enumerate(rows):
            if i + 1 < len(rows):
                u.next_reset_options = {"layouts": [rows[i + 1]]}
            obs, _, _, _, extras = env.step(policy(obs))
            a = extras["attempt"][0]
            results.append((row, a["outcome"], bool(a["success"]), float(a["contact"])))
            say(f"{ARGS.bank} layout {row}: {len(T.layout_from_bank(u.bank, row)[1])} cylinders, "
                f"outcome {a['outcome']}, {'LIFTED' if a['success'] else 'not lifted'}, "
                f"contact {a['contact']:.1f} N")
    finally:
        if rec is not None:
            rec.stop()
    lifted = sum(r[2] for r in results)
    say(f"lifted {lifted} of {len(results)}: {[r[0] for r in results if r[2]]}")
    if rec is not None:
        rec.finish()
        viz.close()
    env.close()
    return lifted == len(results)


if __name__ == "__main__":
    ok = False
    try:
        ok = main()
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        if SERVER.poll() is None:
            os.killpg(SERVER.pid, signal.SIGINT)
            try:
                SERVER.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(SERVER.pid, signal.SIGKILL)
        sys.stdout.flush()
        os._exit(0 if ok else 1)
