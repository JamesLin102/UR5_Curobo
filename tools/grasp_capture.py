"""Capture what the wrist camera sees on the grasp scan, with the truth, for perception work.

    python tools/grasp_capture.py --out /tmp/grasp_capture --layouts 20
    python tools/grasp_capture.py --out DIR --bank eval --first 40 --layouts 10

No planner: each layout from the bank is staged (the cube and cylinders put
where the bank says), then the arm stops at each scan view in turn -- the
same joint blends the scan makes -- and the view is saved as perception gets
it (LabCell.camera_view: depth, colour, intrinsics, the camera pose by
forward kinematics). One .npz per layout, with the truth: the cube's pose as
the simulator has it after settling, and the cylinders.

tools/check_grasp_perception.py scores grasp.perception against these.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--out", required=True)
ap.add_argument("--bank", default="eval")
ap.add_argument("--first", type=int, default=0, help="bank row to start at")
ap.add_argument("--layouts", type=int, default=20)
ap.add_argument("--one-grip", action="store_true",
                help="only layouts where one grip works: the crowded ones training uses")
AppLauncher.add_app_launcher_args(ap)
ARGS = ap.parse_args()
ARGS.headless, ARGS.enable_cameras = True, True
if ARGS.device == "cuda:0" and "--device" not in sys.argv:
    ARGS.device = "cpu"      # the faster at these sizes; --device cuda:0 to override
APP = AppLauncher(ARGS).app

import numpy as np  # noqa: E402

from cell_api import Idle, ResetOptions, Scan  # noqa: E402
from grasp import scene as G  # noqa: E402
from grasp import task as T  # noqa: E402
from lab.tasks.base import CellEnv, CellEnvCfg  # noqa: E402


def main():
    cfg = CellEnvCfg()
    cfg.sim.device = ARGS.device
    cfg.cell.scene = "grasp"
    cfg.cell.planner_mode = "none"
    cfg.cell.mapping = True
    cfg.cell.markers = "off"
    cfg.cell.verbose = False
    env = CellEnv(cfg)
    cell = env.cell
    bank = T.Bank(T.bank_path(ARGS.bank))
    os.makedirs(ARGS.out, exist_ok=True)
    rows = np.arange(len(bank))
    if ARGS.one_grip:
        rows = rows[bank.usable(T.TaskCfg()).sum(axis=1) == 1]
    rows = rows[ARGS.first:ARGS.first + ARGS.layouts]
    for row in rows:
        cube, cyl = T.layout_from_bank(bank, row)
        cell.reset([0], ResetOptions(
            payload_poses={G.CUBE[0]: [cube[0], cube[1], G.TABLE_TOP + G.CUBE_SIZE / 2]
                           + T.cube_quat(cube[2])},
            body_poses=T.body_poses(cyl), scan=False, clear_map=False))
        views = []
        for q in G.SCAN_POSES[:-1]:
            cell.run([0], [[Scan(poses=[q]), Idle(10)]])
            views.append(cell.camera_view("wrist", [0])[0])
        truth = cell.observe([0]).objects[G.CUBE[0]][0]
        np.savez_compressed(
            os.path.join(ARGS.out, f"layout_{ARGS.bank}_{row:04d}.npz"),
            depth=np.stack([v["depth"] for v in views]),
            rgb=np.stack([v["rgb"] for v in views]),
            K=np.stack([v["K"] for v in views]),
            pose=np.stack([v["pose"] for v in views]),
            cube_pose=truth, cylinders=np.asarray(cyl, dtype=np.float64).reshape(-1, 2),
            row=row)
        print(f"[capture] layout {row}: {len(cyl)} cylinders, {len(views)} views", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
