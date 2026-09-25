"""The grasp example's perception layer for cloud_viewer: estimate against truth.

Simulator-free, like perception.py, so it draws a real arm's estimates too
(with truth=None). Everything env-local.

    yellow  the estimated cube, and the cylinders perception found
    green   the true cube (simulation only)
    blue    the true cylinders (simulation only)
    purple  the volume perception keeps points from (NO_GO): anything a
            cylinder could be, so the only place it looks
"""

import math

import numpy as np

from cloud_viewer import BLUE, GREEN, PURPLE, YELLOW

from . import scene as G
from .task import cube_quat


def _cube_pose(x, y, yaw):
    return [x, y, G.TABLE_TOP + G.CUBE_SIZE / 2] + list(cube_quat(yaw))


def show_estimate(viewer, est, cube_truth=None, cylinders_truth=None):
    """est a task.Estimate (None: none yet); cube_truth (x, y, yaw); cylinders_truth [(x, y)]."""
    viewer.clear("perception")
    dims, pose = G.NO_GO
    viewer.box("perception", "no_go", dims, pose, PURPLE)
    lines = []
    if est is None:
        lines.append("perception: not yet (scanning)")
    elif est.visibility <= 0.0:
        lines.append("**perception: no cube seen**")
    else:
        x, y, yaw = (float(v) for v in est.cube)
        viewer.box("perception", "cube_est", [G.CUBE_SIZE] * 3, _cube_pose(x, y, yaw), YELLOW)
        lines.append(f"cube estimate ({x:+.3f}, {y:+.3f}) m, yaw {math.degrees(yaw):+.1f}°, "
                     f"seen {100 * est.visibility:.0f}%")
        if cube_truth is not None:
            tx, ty, tyaw = (float(v) for v in cube_truth)
            viewer.box("perception", "cube_true", [G.CUBE_SIZE + 0.004] * 3,
                       _cube_pose(tx, ty, tyaw), GREEN)
            dyaw = (yaw - tyaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
            lines.append(f"error {1000 * math.hypot(x - tx, y - ty):.1f} mm, "
                         f"{math.degrees(dyaw):+.2f}°")
    found = [] if est is None else [c for c in est.cylinders if c[2] > 0.5]
    for i, (cx, cy, _) in enumerate(found):
        viewer.cylinder("perception", f"cylinder_est/{i}", G.CYL_RADIUS, G.CYL_HEIGHT,
                        [cx, cy, G.TABLE_TOP + G.CYL_HEIGHT / 2], YELLOW)
    if cylinders_truth is not None:
        truth = np.asarray(cylinders_truth, dtype=np.float64).reshape(-1, 2)
        lines.append(f"cylinders found {len(found)} of {len(truth)}")
        for i, (cx, cy) in enumerate(truth):
            viewer.cylinder("perception", f"cylinder_true/{i}", G.CYL_RADIUS + 0.002,
                            G.CYL_HEIGHT, [cx, cy, G.TABLE_TOP + G.CYL_HEIGHT / 2], BLUE)
    else:
        lines.append(f"cylinders found {len(found)}")
    return lines
