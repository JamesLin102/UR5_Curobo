"""Where the cube and the cylinders are, from the wrist camera's scan views.

Simulator-free -- numpy and scipy only -- because it has to run on the real
arm unchanged (docs/grasp_rl_plan.md §4). It takes what the real arm has:

    a view  dict(depth (H, W) metres, 0 = no reading,
                 rgb (H, W, 3) uint8,
                 K (3, 3) intrinsics,
                 pose 4x4 OPTICAL camera pose in the robot's base frame,
                      from forward kinematics and the hand-eye calibration)

and returns a grasp.task.Estimate, the same thing training gets from the
error model (task.synth_estimate), so a policy cannot tell which it is fed.

Per view:
  1. every depth pixel back-projected into the base frame;
  2. kept only inside the volume any cylinder could occupy (the scene's NO_GO
     box): the scan poses were solved to keep the whole arm out of it, so no
     point of the robot's own body can get in -- no self-mask needed;
  3. cylinders: points above the cube's height, clustered in plan; each
     cluster that rises well above the cube is one (its top may be out of
     frame), its axis fitted to the side points with the radius known (or
     the top disc's centre, when only the top is seen);
  4. the cube: pixels of its colour whose points lie on its top face; the
     smallest rectangle round them gives the yaw (mod 90 deg) and the centre,
     and how much of the 45 mm square they cover is the visibility.
Then the views are fused: the cube from the view that saw most of it, the
cylinders merged across views.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

from . import scene as G
from .task import Estimate


@dataclass
class PerceptionCfg:
    depth_min: float = 0.28          # the D435's near limit (scene camera "near")
    depth_max: float = 2.0
    top_band: float = 0.006          # a point within this of the cube's top is on it
    min_cube_px: int = 30            # fewer top-face pixels than this: not seen
    cyl_min_z: float = 0.07          # cylinder points are above the cube (0.047)
    # A cluster is a cylinder if it rises this high: clearly taller than the
    # cube, not necessarily to its top. A cylinder at the workspace's edge
    # (y ~ 0.31) has its top out of every view's frame -- HOME's footprint at
    # 0.30 m reaches only y ~ +-0.18 -- and was seen to 0.18-0.27 m only.
    cyl_min_top: float = 0.12
    cyl_min_pts: int = 150
    grid: float = 0.005              # plan-view cell for clustering, m
    merge: float = 0.03              # detections of one cylinder across views
    # The cube's colour: yellow, lit or shaded. Red and green both well above
    # blue BY A DIFFERENCE, not a ratio: the top face, lit from above, comes
    # out a washed yellow (247, 245, 154) whose blue is 0.6 of its green, the
    # shaded sides a deep one (218, 215, 96). Grey (the table) and blue (the
    # cylinders) have no such excess.
    yellow_excess: int = 35
    yellow_min: int = 60


@dataclass
class ViewResult:
    cube: Optional[tuple]            # (x, y, yaw, visibility, residual) or None
    cylinders: List[tuple]           # (x, y, points)


def backproject(depth, K, pose):
    """(N, 3) base-frame points and their (N,) flat pixel indices, for readings > 0."""
    v, u = np.nonzero(depth > 0)
    z = depth[v, u].astype(np.float64)
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], axis=1) @ pose[:3, :3].T + pose[:3, 3]
    return pts, v * depth.shape[1] + u, z


def _in_box(pts):
    dims, pose = G.NO_GO
    lo = np.asarray(pose[:3]) - np.asarray(dims) / 2
    hi = np.asarray(pose[:3]) + np.asarray(dims) / 2
    return np.all((pts >= lo) & (pts <= hi), axis=1)


def _fit_circle_fixed_r(xy, r, c0, iters=20):
    """Centre of a circle of known radius r through points on it (Gauss-Newton)."""
    c = np.asarray(c0, dtype=np.float64)
    for _ in range(iters):
        d = xy - c
        n = np.linalg.norm(d, axis=1)
        n = np.where(n < 1e-9, 1e-9, n)
        res = n - r
        J = -d / n[:, None]
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        c = c + step
        if np.linalg.norm(step) < 1e-6:
            break
    return c


def _cylinders(pts, cfg):
    top = G.TABLE_TOP + G.CYL_HEIGHT
    sel = pts[(pts[:, 2] > G.TABLE_TOP + cfg.cyl_min_z) & (pts[:, 2] < top + 0.01)]
    if len(sel) < cfg.cyl_min_pts:
        return []
    ij = np.floor(sel[:, :2] / cfg.grid).astype(int)
    lo = ij.min(axis=0)
    ij -= lo
    occ = np.zeros(ij.max(axis=0) + 1, dtype=bool)
    occ[ij[:, 0], ij[:, 1]] = True
    labels, n = ndimage.label(occ, structure=np.ones((3, 3)))
    lab = labels[ij[:, 0], ij[:, 1]]
    out = []
    for k in range(1, n + 1):
        p = sel[lab == k]
        if len(p) < cfg.cyl_min_pts or p[:, 2].max() < G.TABLE_TOP + cfg.cyl_min_top:
            continue
        side = p[p[:, 2] < top - 0.01]
        disc = p[p[:, 2] >= top - 0.01]
        if len(side) >= 50:
            c = _fit_circle_fixed_r(side[:, :2], G.CYL_RADIUS, p[:, :2].mean(axis=0))
        else:
            c = disc[:, :2].mean(axis=0)
        out.append((float(c[0]), float(c[1]), int(len(p))))
    return out


def _min_area_rect(xy):
    """(centre, angle, (w, h)) of the smallest rectangle round points (rotating calipers)."""
    hull = xy[ConvexHull(xy).vertices]
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        a = math.atan2(e[1], e[0])
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, s], [-s, c]])
        q = hull @ R.T
        mn, mx = q.min(axis=0), q.max(axis=0)
        area = float(np.prod(mx - mn))
        if best is None or area < best[0]:
            centre = ((mn + mx) / 2) @ R
            best = (area, centre, a, mx - mn)
    return best[1], best[2], best[3]


def _cube(pts, pix, rgb, cfg):
    if rgb is None:
        return None
    col = rgb.reshape(-1, 3)[pix].astype(np.float64)
    r, g, b = col[:, 0], col[:, 1], col[:, 2]
    rg = np.minimum(r, g)
    yellow = (rg - b > cfg.yellow_excess) & (rg > cfg.yellow_min) & (np.abs(r - g) < 60)
    top = G.TABLE_TOP + G.CUBE_SIZE
    on_top = yellow & (np.abs(pts[:, 2] - top) < cfg.top_band)
    face = pts[on_top, :2]
    if len(face) < cfg.min_cube_px:
        return None
    try:
        centre, ang, wh = _min_area_rect(face)
    except Exception:
        return None
    yaw = (ang + math.pi / 4) % (math.pi / 2) - math.pi / 4
    # How much of the top face was seen: covered 2 mm cells over the face's cells.
    cell = 0.002
    cells = {(int(x // cell), int(y // cell)) for x, y in face}
    visibility = min(1.0, len(cells) * cell * cell / (G.CUBE_SIZE ** 2))
    residual = float(abs(G.CUBE_SIZE - max(wh)) + abs(G.CUBE_SIZE - min(wh))) / 2
    return (float(centre[0]), float(centre[1]), float(yaw), float(visibility), residual)


def perceive_view(view, cfg: PerceptionCfg = PerceptionCfg()) -> ViewResult:
    pts, pix, z = backproject(view["depth"], view["K"], view["pose"])
    keep = (z >= cfg.depth_min) & (z <= cfg.depth_max) & _in_box(pts)
    pts, pix = pts[keep], pix[keep]
    return ViewResult(cube=_cube(pts, pix, view.get("rgb"), cfg),
                      cylinders=_cylinders(pts, cfg))


def fuse(results: List[ViewResult], cfg: PerceptionCfg = PerceptionCfg()) -> Optional[Estimate]:
    """One estimate from every view; None if no view saw the cube."""
    cubes = [r.cube for r in results if r.cube is not None]
    if not cubes:
        return None
    best = max(cubes, key=lambda c: c[3])
    # Average the views that saw nearly all of it; yaw on the 4-fold circle.
    good = [c for c in cubes if c[3] >= best[3] - 0.05]
    w = np.array([c[3] for c in good])
    x = float(np.average([c[0] for c in good], weights=w))
    y = float(np.average([c[1] for c in good], weights=w))
    a4 = math.atan2(np.average([math.sin(4 * c[2]) for c in good], weights=w),
                    np.average([math.cos(4 * c[2]) for c in good], weights=w))
    cube = np.array([x, y, a4 / 4])
    dets = [(cx, cy, n) for r in results for cx, cy, n in r.cylinders]
    merged = []
    for cx, cy, n in sorted(dets, key=lambda d: -d[2]):
        for m in merged:
            if math.hypot(cx - m[0] / m[2], cy - m[1] / m[2]) < cfg.merge:
                m[0] += cx * n
                m[1] += cy * n
                m[2] += n
                break
        else:
            merged.append([cx * n, cy * n, n])
    cyl = np.zeros((G.MAX_CYLINDERS, 3))
    for i, m in enumerate(sorted(merged, key=lambda m: -m[2])[:G.MAX_CYLINDERS]):
        cyl[i] = (m[0] / m[2], m[1] / m[2], 1.0)
    return Estimate(cube=cube, visibility=float(best[3]), residual=float(best[4]), cylinders=cyl)


def perceive(views, cfg: PerceptionCfg = PerceptionCfg()) -> Optional[Estimate]:
    return fuse([perceive_view(v, cfg) for v in views], cfg)
