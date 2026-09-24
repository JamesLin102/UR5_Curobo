"""Look at the grasp scene: build it on Isaac Lab, run its scan, keep the wrist camera's views.

    python tools/grasp_view.py                  # a window; the scan repeats until it is closed
    python tools/grasp_view.py --headless       # one scan, the images, and out
    python tools/grasp_view.py --out DIR        # where the images go (default: $TMPDIR/grasp_views)

No planner: the arm is blended between HOME and the scan poses in joint space,
exactly as the cell's Scan op does, so what is seen is what a scan will do.
The window also draws, as an overlay the cameras do not see, NO_GO (the box
any cylinder could occupy, which the arm must stay out of while scanning), and
on the table the regions the cube's centre (yellow) and the cylinders' centres
(blue) will be drawn from.

At each view the wrist camera's colour and depth images are written to --out,
with what the camera saw of the cube and the cylinders.
"""

import argparse
import math
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "grasp_views"),
                help="where the wrist camera's images go")
ap.add_argument("--blend", type=int, default=90, help="sim steps from one view to the next")
ap.add_argument("--hold", type=int, default=60, help="sim steps to hold each view")
AppLauncher.add_app_launcher_args(ap)
ARGS = ap.parse_args()
ARGS.enable_cameras = True
if ARGS.device == "cuda:0":
    ARGS.device = "cpu"       # see docs/isaaclab.md: CPU PhysX is the faster one here
APP = AppLauncher(ARGS).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cell_api import ResetOptions  # noqa: E402
from grasp import scene as G  # noqa: E402
from lab.tasks.base import CellEnv, CellEnvCfg  # noqa: E402

VIEW_NAMES = [v[0] for v in G.VIEWS]


def say(msg):
    print(f"[view] {msg}", flush=True)


def box_lines(dims, pose):
    lo = np.asarray(pose[:3]) - np.asarray(dims) / 2
    hi = np.asarray(pose[:3]) + np.asarray(dims) / 2
    c = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if bin(i ^ j).count("1") == 1]
    return [c[i] for i, _ in edges], [c[j] for _, j in edges]


def rect_lines(xy, z):
    (x0, x1), (y0, y1) = xy
    pts = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    return pts, pts[1:] + pts[:1]


def _draw():
    try:
        from isaacsim.util.debug_draw import _debug_draw
    except ImportError:
        return None
    return _debug_draw.acquire_debug_draw_interface()


def clear_overlay():
    """The overlay shows up in the camera's COLOUR image, so it goes before a capture."""
    draw = _draw()
    if draw is not None:
        draw.clear_points()
        draw.clear_lines()


def draw_regions():
    """NO_GO and the sampling regions, as a viewport overlay (not scene geometry)."""
    draw = _draw()
    if draw is None:
        return
    starts, ends, colours = [], [], []
    for (s, e), rgba in ((box_lines(*G.NO_GO), (0.90, 0.25, 0.20, 1.0)),
                         (rect_lines(G.CUBE_XY, G.TABLE_TOP + 0.002), (0.95, 0.80, 0.05, 1.0)),
                         (rect_lines(G.CYL_XY, G.TABLE_TOP + 0.002), (0.20, 0.35, 0.70, 1.0))):
        starts += s
        ends += e
        colours += [rgba] * len(s)
    draw.draw_lines(starts, ends, colours, [2.0] * len(starts))


def place_cube(cell):
    """The scene's cube pose, yaw included (reset puts the payload down square)."""
    name, _, pose, *_ = G.CUBE
    obj = cell.payload[name]
    p = torch.tensor([list(np.asarray(pose[:3]) + cell.origins[0]) + list(pose[3:])],
                     dtype=torch.float32, device=cell.device)
    obj.write_root_pose_to_sim(p, env_ids=torch.tensor([0], device=cell.device))
    obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=cell.device),
                                   env_ids=torch.tensor([0], device=cell.device))


def blend_to(view, q_to, steps):
    q_from = np.asarray(view.observe().q, dtype=np.float64)
    q_to = np.asarray(q_to, dtype=np.float64)
    for s in range(steps):
        t = (s + 1) / steps
        view.command_arm((1 - t) * q_from + t * q_to)
        view.idle(1)


def save_views(cell, name, out):
    from PIL import Image

    cam = cell.cams["wrist"]
    depth = cam.data.output["distance_to_image_plane"][0, :, :, 0].cpu().numpy()
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    rgb = cam.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
    Image.fromarray(rgb).save(os.path.join(out, f"{name}_rgb.png"))
    valid = depth > 0
    lo, hi = (np.percentile(depth[valid], [1, 99]) if valid.any() else (0.0, 1.0))
    grey = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (255 * (1 - grey)).astype(np.uint8)
    img[~valid] = 0
    Image.fromarray(img).save(os.path.join(out, f"{name}_depth.png"))
    # What of the cube the view holds: its yellow pixels, shaded ones included.
    yellow = (rgb[:, :, 0] > 120) & (rgb[:, :, 1] > 90) & (rgb[:, :, 2] < 0.5 * rgb[:, :, 1])
    say(f"  {name}: depth {depth[valid].min():.2f}..{depth[valid].max():.2f} m, "
        f"{int(yellow.sum())} cube-coloured pixels, images in {out}")


def main():
    cfg = CellEnvCfg()
    cfg.sim.device = ARGS.device
    cfg.cell.scene = "grasp"
    cfg.cell.planner_mode = "none"
    cfg.cell.mapping = True           # builds the wrist camera; nothing is mapped
    cfg.cell.markers = "off"          # overlays show up in the colour images
    env = CellEnv(cfg)
    cell = env.cell
    view = cell.view(0)
    os.makedirs(ARGS.out, exist_ok=True)

    cell.reset([0], ResetOptions(block_on=0, clear_map=False, scan=False))
    place_cube(cell)
    view.idle(30)
    o = view.observe()
    cube = o.objects[G.CUBE[0]]
    say(f"cube at [{cube[0]:+.3f} {cube[1]:+.3f} {cube[2]:+.3f}], "
        f"yaw {math.degrees(2 * math.atan2(cube[6], cube[3])):+.1f} deg; "
        f"{len(G.CYLINDERS)} cylinders r {1000 * G.CYL_RADIUS:.0f} mm h {G.CYL_HEIGHT:.2f} m")
    say(f"grasp height {1000 * G.GRASP_Z:.1f} mm, pre-grasp {1000 * (G.GRASP_Z + G.DESCEND):.0f} mm")

    rounds = 0
    while APP.is_running():
        draw_regions()
        rounds += 1
        for i, q in enumerate(G.SCAN_POSES[:-1]):
            name = VIEW_NAMES[i] if i < len(VIEW_NAMES) else f"scan{i}"
            blend_to(view, q, ARGS.blend)
            view.idle(ARGS.hold)
            if rounds == 1:
                clear_overlay()
                view.idle(2)
                save_views(cell, name, ARGS.out)
                draw_regions()
        blend_to(view, G.HOME, ARGS.blend)
        view.idle(ARGS.hold)
        if ARGS.headless:
            break
        if rounds == 1:
            say("scan shown; it repeats until the window is closed")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    APP.close()
