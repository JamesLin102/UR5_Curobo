"""Look at captured grasp scans in the browser: points, perception, truth. No simulator.

    python tools/grasp_capture.py --out /tmp/grasp_capture --layouts 20   # first, once
    python tools/view_captures.py /tmp/grasp_capture                      # then open the URL

Each layout tools/grasp_capture.py saved is one entry in the panel's dropdown:
its scan views as coloured point clouds with their frustums (cameras), what
grasp.perception makes of them against the truth (perception), and the scene
with the cylinders and the cube where they really stood (reference). The map
layer stays empty: nothing was fused. Runs until Ctrl-C.
"""

import argparse
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import scenes  # noqa: E402
from cloud_viewer import CloudViewer  # noqa: E402
from grasp import perception as P  # noqa: E402
from grasp import scene as G  # noqa: E402
from grasp import task as T  # noqa: E402
from grasp import viz  # noqa: E402


def load(path):
    d = np.load(path)
    views = [dict(depth=d["depth"][i], rgb=d["rgb"][i], K=d["K"][i], pose=d["pose"][i])
             for i in range(len(d["depth"]))]
    return views, d["cube_pose"], d["cylinders"]


def show(viewer, spec, path):
    views, cube_pose, cyl = load(path)
    n = viewer.show_views(views, labels=[f"view{i}" for i in range(len(views))])
    est = P.perceive(views)
    cube = (float(cube_pose[0]), float(cube_pose[1]), _yaw(cube_pose))
    lines = viz.show_estimate(viewer, est, cube, cyl)
    viewer.show_scene(spec, bodies=T.body_poses(cyl.tolist()),
                      payload={G.CUBE[0]: list(cube_pose)})
    viewer.show_voxels(None, 0)
    viewer.set_status(f"**{os.path.basename(path)}**  \n{len(views)} views, {n} points  \n"
                      + "  \n".join(lines))


def _yaw(pose):
    w, x, y, z = pose[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", help="a tools/grasp_capture.py --out directory")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--stride", type=int, default=4, help="every Nth pixel each way")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "layout_*.npz")))
    if not files:
        sys.exit(f"no layout_*.npz in {args.dir}; run tools/grasp_capture.py --out {args.dir}")
    spec = scenes.load("grasp")
    viewer = CloudViewer(port=args.port, stride=args.stride, title="grasp captures")
    pick = viewer.server.gui.add_dropdown("layout", [os.path.basename(f) for f in files])
    pick.on_update(lambda _: show(viewer, spec, os.path.join(args.dir, pick.value)))
    show(viewer, spec, files[0])
    print(f"[view] {len(files)} layouts from {args.dir}; open {viewer.url}  (Ctrl-C to stop)",
          flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
