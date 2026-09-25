"""Record the grasp policy on chosen layouts: Isaac Lab on the left, the point cloud view on the right.

    python tools/record_grasp.py --layouts 51 139 --dry-run    # which of these does it lift?
    python tools/record_grasp.py --layouts 51 139              # then record them

Evaluation mode, as the real arm would run: each layout starts with the wrist
camera's scan, the map and grasp.perception's estimate come from it, and the
policy (--policy, default grasp/policies/first.pt, the first run's last
checkpoint) makes one attempt.
Starts its own planner server (mapping on). Writes --name.mp4 (real time) and
--name.gif (2x speed) into --out; the right view carries the perception layer
too -- the estimate against the truth. The video stops where the last
layout's episode ends.

--dry-run records nothing: it runs each layout once and says what the attempt
came to, to pick layouts the policy lifts.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from lab import app as lab_app  # noqa: E402

FIRST = os.path.join(ROOT, "scripts", "grasp", "policies", "first.pt")

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
lab_app.add_args(ap, task="Grasp")
ap.set_defaults(mapping=True)      # evaluation mode: the scan, the map, perception
ap.add_argument("--layouts", type=int, nargs="+", required=True, help="bank rows, in order")
ap.add_argument("--bank", default="eval")
ap.add_argument("--policy", default=FIRST, help="a model_N.pt checkpoint, 'oracle' or 'random'")
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


os.makedirs(ARGS.out, exist_ok=True)
SERVER = None
APP = lab_app.launch(ARGS)

import gymnasium as gym  # noqa: E402

from lab import policy as lab_policy  # noqa: E402
from grasp import task as T  # noqa: E402


def main():
    global SERVER
    tid, cfg = lab_app.make_env_cfg(ARGS)
    cfg.task.bank = ARGS.bank
    cfg.viewer.resolution = (W, H)
    cfg.viewer.eye, cfg.viewer.lookat = tuple(ARGS.eye), tuple(ARGS.lookat)
    SERVER = lab_app.start_servers(cfg, log=say)
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
    # the last layout. (The perception layer is the env's own, GraspEnv.viz_draw.)
    reset_cells = u.reset_cells

    def reset_and_count(env_ids, options):
        state["resets"] += 1
        if rec is not None and state["resets"] > len(rows):
            rec.paused = True
        reset_cells(env_ids, options)
    u.reset_cells = reset_and_count

    policy = lab_policy.make(env, ARGS.policy, ARGS.rl_device, log=say)
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
        if SERVER is not None:
            SERVER.stop()
        sys.stdout.flush()
        os._exit(0 if ok else 1)
