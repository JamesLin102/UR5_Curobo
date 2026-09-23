"""The pick-and-place demo on Isaac Lab: isaacsim_client.py, on the other backend.

Same loop (demo_loop), same planner server, same scene. The cell comes from
the registered gym task, so what runs here is exactly what a learning loop
would get from gymnasium.make().

This process runs Isaac Lab ONLY and must never import cuRobo -- see lab/.

Start planner_server.py first, then:
    python scripts/lab/isaaclab_client.py --device cpu
    python scripts/lab/isaaclab_client.py --headless --no-mapping

Several cells at once: one planner server per cell (each holds its own map),
then the same client with --num_envs. Every cell shuttles its own block,
chosen by an oracle through the gym env's step(), one leg per step, in
lockstep:
    python scripts/planner_servers.py --num 4
    python scripts/lab/isaaclab_client.py --num_envs 4 --camera-class tiled
"""

import argparse
import os
import sys

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)

import scenes  # noqa: E402
from lab import app as lab_app  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
lab_app.add_args(_ap)
_ap.add_argument("--static", action="store_true",
                 help="hold the arm at HOME; no planning, just look")
ARGS = _ap.parse_args()

# Checked before the simulator starts, so a scene without anything to pick
# fails in a second, not a minute.
if not scenes.load(ARGS.scene).payload:
    raise SystemExit(f"scene {ARGS.scene!r} has no payload; this client only "
                     f"runs pick-and-place")

APP = lab_app.launch(ARGS)

import gymnasium as gym  # noqa: E402

import demo_loop  # noqa: E402
from cell_api import ResetOptions  # noqa: E402


def say(msg):
    print(f"[demo] {msg}", flush=True)


def run_many(env):
    """N cells, one leg each per env step: pick where the block is, place on the goal.

    The oracle reads each cell's goal off the task, so a cell whose leg
    failed simply tries again on the next step. A delivered block ends that
    cell's episode; the auto-reset leaves it where it was delivered (and, with
    mapping on, rescans that cell), and the shuttle carries on from there.
    """
    import torch

    u = env.unwrapped
    n = u.num_envs
    targets = u.scene_spec.targets
    grasp_z = targets[0][2] - u.scorer.descend
    env.reset(options={"block_on": [e % 2 for e in range(n)]})
    say(f"{n} cells shuttling their blocks; one env step = one leg in every cell")
    step = 0
    while u.cell.running:
        step += 1
        holding = u.cell.observe(range(n)).holding
        goal = u.scorer.goal.copy()
        at = [goal[e] if holding[e] else 1 - goal[e] for e in range(n)]
        u.next_reset_options = {"block_on": [int(g) for g in goal]}
        action = torch.tensor([[targets[i][0], targets[i][1], grasp_z, 0.0] for i in at])
        _, _, term, _, extras = env.step(action)
        legs = extras["leg"]
        done = [e for e in range(n) if bool(term[e])]
        failed = [f"env {e}: {legs[e]['failed']}" for e in range(n) if "failed" in legs[e]]
        say(f"step {step}: " + " ".join(f"{legs[e]['leg']}@{at[e]}" for e in range(n))
            + (f" | delivered in env {done}" if done else "")
            + (f" | {'; '.join(failed)}" if failed else ""))


def main():
    tid, cfg = lab_app.make_env_cfg(ARGS)
    env = gym.make(tid, cfg=cfg)
    if ARGS.num_envs > 1:
        if ARGS.static:
            raise SystemExit("--static drives one cell; leave out --num_envs")
        run_many(env)
        env.close()
        return
    cell = env.unwrapped.cell.view(0)
    # The map is empty on a fresh server; nothing to clear.
    cell.reset(ResetOptions(block_on=0, clear_map=False, scan=True))

    demo_loop.banner(cell, body_path=lambda name: f"/World/envs/env_0/{name}")
    if ARGS.static:
        demo_loop.static_loop(cell)
    else:
        demo_loop.run_forever(cell)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # A Kit app often hangs on the way out after an exception, holding the
        # GPU. Report and leave.
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    APP.close()
