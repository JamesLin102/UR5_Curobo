"""The pick-and-place demo on Isaac Lab: isaacsim_client.py, on the other backend.

Same loop (demo_loop), same planner server, same scene. The cell comes from
the registered gym task, so what runs here is exactly what a learning loop
would get from gymnasium.make().

This process runs Isaac Lab ONLY and must never import cuRobo -- see lab/.

Start planner_server.py first, then:
    python scripts/lab/isaaclab_client.py
    python scripts/lab/isaaclab_client.py --headless --no-mapping
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
if ARGS.num_envs != 1:
    raise SystemExit("the demo drives one cell; use --num_envs 1")

APP = lab_app.launch(ARGS)

import gymnasium as gym  # noqa: E402

import demo_loop  # noqa: E402
from cell_api import ResetOptions  # noqa: E402


def main():
    tid, cfg = lab_app.make_env_cfg(ARGS)
    env = gym.make(tid, cfg=cfg)
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
