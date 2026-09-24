"""UR5 + 2F-85 + wrist D435i: pick and place, planned by cuRobo 0.8 against a live map.

The demo, and nothing else: the cell itself -- stage, cameras, gripper, the
motion primitives -- is sim_env.SimEnv, and the loop is demo_loop, shared with
the Isaac Lab client. It shuttles the scene's payload between its targets
forever, which is what the README's numbers measure.

The scene's `unmapped` bodies are never described to the planner. The arm can
only discover them through the cameras, so avoiding them is proof the map is
actually feeding the planner. Each example's scene is its scene.py, and both
processes must be started with the same --scene.

This process runs Isaac Sim ONLY and must never import cuRobo -- see sim_env.

Start planner_server.py first, then:
    python scripts/pick_place/isaacsim_client.py
"""

import argparse
import os
import sys

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import scenes  # noqa: E402
from pick_place import demo_loop  # noqa: E402
from rig import DEFAULT_ROBOT, ROBOTS  # noqa: E402
from sim import sim_env  # noqa: E402

# planner_server.py flushes every line; without the same here, this side's
# output sits in the stdout buffer whenever it is redirected to a file, and the
# demo looks hung next to a server that is visibly working.
sys.stdout.reconfigure(line_buffering=True)

_ap = argparse.ArgumentParser()
_ap.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
_ap.add_argument("--scene", default=scenes.DEFAULT, choices=scenes.available(),
                 help="an example with a scene.py under scripts/; must match the server")
_ap.add_argument("--no-mapping", action="store_true")
_ap.add_argument("--no-overhead", action="store_true",
                 help="wrist camera only, for A/B against the fixed camera")
_ap.add_argument("--map-every", type=int, default=6, help="fuse a frame every N sim steps")
_ap.add_argument("--static", action="store_true",
                 help="hold the arm at HOME; no planning, just look")
_ap.add_argument("--depth-lag", type=int, default=2,
                 help="sim steps the depth annotator trails the physics by")
_ap.add_argument("--headless", action="store_true",
                 help="no window; renders only if the cameras need it")
ARGS = _ap.parse_args()

# The only loop this demo has is pick-and-place. Checked before Isaac Sim
# starts, so a scene without anything to pick fails in a second, not a minute.
if not scenes.load(ARGS.scene).payload:
    raise SystemExit(f"scene {ARGS.scene!r} has no payload; this client only "
                     f"runs pick-and-place")


def main():
    sim_env.launch(headless=ARGS.headless)
    env = sim_env.SimEnv(sim_env.EnvCfg(
        scene=ARGS.scene, robot=ARGS.robot, mapping=not ARGS.no_mapping,
        overhead=not ARGS.no_overhead, map_every=ARGS.map_every,
        depth_lag=ARGS.depth_lag))
    # The map is empty on a fresh server; nothing to clear.
    env.reset(sim_env.ResetOptions(block_on=0, clear_map=False, scan=True))

    demo_loop.banner(env)
    if ARGS.static:
        demo_loop.static_loop(env)
    else:
        demo_loop.run_forever(env)
    env.close()


if __name__ == "__main__":
    main()
