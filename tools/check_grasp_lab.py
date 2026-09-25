"""Does the gripper actually lift the cube off the table, wherever the grasp scene puts it?

    python tools/check_grasp_lab.py                 # 20 grips, empty table, headless
    python tools/check_grasp_lab.py --trials 50 --seed 3

The physics half of the question tools/grasp_reach.py answers kinematically.
One command: it starts its own planner server (--scene grasp --no-mapping),
builds the grasp scene on Isaac Lab with the cylinders taken off the table
(tools/check_grasp_env.py is the check with them, through the gym env),
and for each trial puts the cube somewhere in CUBE_XY at a random yaw and
runs the grip an oracle would -- legs.leg_ops, tool square to the cube's faces:
plan to above it, straight down, close, lift.

Per grip it reports whether it lifted (the cube rose by at least half of
lift_m and is still up after holding 30 steps), how far the fingers pushed it
before it rose, and where the gripper's leader joint stopped. That last number
is what "holding" will be read from on the real arm:
closed on a 45 mm cube it stalls short of the commanded angle, closed on
nothing it goes all the way.

Pass: every grip that was planned lifts the cube. Grips the planner refuses
are counted separately; tools/grasp_reach.py says which those are and why.
"""

import argparse
import math
import os
import signal
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--trials", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(ap)
ARGS = ap.parse_args()
ARGS.headless = True
if ARGS.device == "cuda:0" and "--device" not in sys.argv:
    ARGS.device = "cpu"      # the faster at these sizes; --device cuda:0 to override


def log(msg):
    print(f"[grasp] {msg}", flush=True)


def start_server():
    logfile = os.path.join(tempfile.gettempdir(), "check_grasp_server.log")
    out = open(logfile, "w")
    proc = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "scripts", "planner_servers.py"),
                             "--num", "1", "--scene", "grasp", "--no-mapping"],
                            stdout=out, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"planner server exited; see {logfile}")
        if "[servers] all" in open(logfile).read():
            return proc
        time.sleep(1)
    raise SystemExit(f"planner server did not start; see {logfile}")


def stop_server(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(40)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


SERVER = start_server()
APP = AppLauncher(ARGS).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cell_api import ResetOptions  # noqa: E402
from grasp import scene as G  # noqa: E402
from lab.tasks.base import CellEnv, CellEnvCfg  # noqa: E402
from legs import leg_ops, tool_pose  # noqa: E402
from rig import ROBOTS  # noqa: E402

LIFT = G.SCENE.pick["lift_m"]
HOLD_STEPS = 30


def put(cell, name, pose):
    """A payload to a pose (env-local), at rest."""
    ids = torch.tensor([0], device=cell.device)
    p = torch.tensor([list(np.asarray(pose[:3]) + cell.origins[0]) + list(pose[3:])],
                     dtype=torch.float32, device=cell.device)
    cell.payload[name].write_root_pose_to_sim(p, env_ids=ids)
    cell.payload[name].write_root_velocity_to_sim(torch.zeros((1, 6), device=cell.device), env_ids=ids)


def clear_table(cell):
    """The cylinders out of the cell: this is the empty-table check."""
    cell.place_bodies([0], [{name: None for name, *_ in G.CYLINDERS}])


def main():
    cfg = CellEnvCfg()
    cfg.sim.device = ARGS.device
    cfg.cell.scene = "grasp"
    cfg.cell.mapping = False
    cfg.cell.markers = "off"
    cfg.cell.verbose = False
    env = CellEnv(cfg)
    cell = env.cell
    cube = G.CUBE[0]
    closed = ROBOTS[cfg.cell.robot]["gripper"]["closed"]
    rng = np.random.default_rng(ARGS.seed)

    refused, lifted, dropped, angles, pushes = [], 0, [], [], []
    for t in range(ARGS.trials):
        x = rng.uniform(*G.CUBE_XY[0])
        y = rng.uniform(*G.CUBE_XY[1])
        yaw = rng.uniform(-math.pi / 4, math.pi / 4)
        tool_yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
        cell.reset([0], ResetOptions(block_on=0, clear_map=False, scan=False))
        clear_table(cell)
        put(cell, cube, [x, y, G.TABLE_TOP + G.CUBE_SIZE / 2,
                         math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        cell.idle([0], 30)
        start = cell.observe([0]).objects[cube][0, :3].copy()

        ops = leg_ops(tool_pose(x, y, G.GRASP_Z, tool_yaw), close=True, open_first=False,
                      descend=G.DESCEND, lift=LIFT)
        r = cell.run([0], [ops])[0]
        if not r.ok and "no plan" in (r.why or ""):
            refused.append((round(x, 3), round(y, 3), round(math.degrees(tool_yaw))))
            log(f"trial {t:2d}: cube [{x:.3f} {y:+.3f}] yaw {math.degrees(yaw):+5.1f}: refused ({r.why})")
            continue
        cell.idle([0], HOLD_STEPS)
        o = cell.observe([0])
        end = o.objects[cube][0, :3]
        rise = float(end[2] - start[2])
        push = float(np.linalg.norm(end[:2] - start[:2]))
        grip = float(o.gripper[0])
        ok = r.ok and rise >= LIFT / 2
        lifted += ok
        if ok:
            angles.append(grip)
            pushes.append(push)
        else:
            dropped.append((round(x, 3), round(y, 3), round(math.degrees(tool_yaw))))
        log(f"trial {t:2d}: cube [{x:.3f} {y:+.3f}] yaw {math.degrees(yaw):+5.1f}: "
            f"{'LIFTED' if ok else 'NOT lifted'}  rise {1000 * rise:+6.1f} mm, "
            f"moved {1000 * push:4.1f} mm sideways, gripper at {grip:+.3f} rad "
            f"(commanded {closed:+.3f}){'' if r.ok else f', {r.why}'}")

    tried = ARGS.trials - len(refused)
    log(f"{lifted}/{tried} planned grips lifted the cube; {len(refused)} refused by the planner")
    if angles:
        log(f"gripper when holding: {min(angles):+.3f}..{max(angles):+.3f} rad "
            f"(commanded {closed:+.3f}); cube moved sideways {1000 * min(pushes):.1f}.."
            f"{1000 * max(pushes):.1f} mm while being gripped")
    if refused:
        log(f"refused (x, y, tool yaw deg): {refused}")
    if dropped:
        log(f"not lifted (x, y, tool yaw deg): {dropped}")
    passed = tried > 0 and lifted == tried
    log("RESULT " + ("PASS" if passed else "FAIL"))
    env.close()
    return passed


if __name__ == "__main__":
    ok = False
    try:
        ok = main()
    except BaseException:
        import traceback
        traceback.print_exc()
    finally:
        stop_server(SERVER)
        sys.stdout.flush()
        os._exit(0 if ok else 1)
