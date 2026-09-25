"""Record the pick_place demo: Isaac Lab on the left, the point cloud view on the right.

    python tools/record_pick_place.py --out docs/media        # then open the URL it prints

Starts its own planner server (mapping on), runs the start-up scan and two
round trips (block 0 -> 1 -> 0, twice), and writes

    pick_place.mp4   real time, 30 fps, both views side by side
    pick_place.gif   2x speed, 10 fps, smaller, for the README

Frames are taken on simulation time, every --every physics steps (60 Hz), so
the video plays at the arm's real speed however slowly it was recorded. The
left view is the env's own render (the viewer camera, cfg.viewer); the right
one is cloud_viewer.render() of the same layers --viz serves, from the same
eye, look-at and field of view. No browser is needed; the --viz page is up
while it records, to watch.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from lab import app as lab_app  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
lab_app.add_args(ap)
ap.add_argument("--out", default=os.path.join(ROOT, "docs", "media"))
ap.add_argument("--round-trips", type=int, default=2)
ap.add_argument("--every", type=int, default=2, help="physics steps per frame (60 Hz / 2 = 30 fps)")
ap.add_argument("--size", default="800x600", help="each view's WxH")
ap.add_argument("--no-server", action="store_true", help="a planner server is already running")
ap.add_argument("--eye", type=float, nargs=3, default=(1.55, 0.95, 1.05),
                help="both views' camera position, env-local")
ap.add_argument("--lookat", type=float, nargs=3, default=(0.3, 0.0, 0.2))
ARGS = ap.parse_args()
ARGS.headless, ARGS.viz, ARGS.device = True, True, ("cpu" if "--device" not in sys.argv else ARGS.device)
W, H = (int(v) for v in ARGS.size.split("x"))


def say(msg):
    print(f"[record] {msg}", flush=True)


os.makedirs(ARGS.out, exist_ok=True)
SERVER = None
APP = lab_app.launch(ARGS)

import gymnasium as gym  # noqa: E402

from cell_api import ResetOptions  # noqa: E402
from lab.viz import CellViz  # noqa: E402
from pick_place import demo_loop  # noqa: E402
from recording import Recorder  # noqa: E402


def main():
    global SERVER
    tid, cfg = lab_app.make_env_cfg(ARGS)
    if not ARGS.no_server:
        SERVER = lab_app.start_servers(cfg, log=say)
    cfg.viewer.resolution = (W, H)
    cfg.viewer.eye, cfg.viewer.lookat = tuple(ARGS.eye), tuple(ARGS.lookat)
    env = gym.make(tid, cfg=cfg, render_mode="rgb_array")
    viz = CellViz(env, 0, ARGS.viz_port, ARGS.viz_stride, log=say)
    rec = Recorder(env, viz, ARGS.out, "pick_place", (W, H), ARGS.every, log=say)
    cell = env.unwrapped.cell.view(0)

    t0 = time.time()
    rec.start()
    try:
        cell.reset(ResetOptions(block_on=0, clear_map=True, scan=True))
        done, tries, src, dst = 0, 0, 0, 1
        while done < 2 * ARGS.round_trips and tries < 4 * ARGS.round_trips:
            tries += 1
            if demo_loop.pick_place_cycle(cell, tries, src, dst):
                done += 1
                src, dst = dst, src
        cell.idle(30)
    finally:
        rec.stop()
    say(f"{done} of {2 * ARGS.round_trips} pick-and-place cycles in {tries} tries, "
        f"recorded in {time.time() - t0:.0f} s")
    rec.finish()
    viz.close()
    env.close()
    return done == 2 * ARGS.round_trips


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
