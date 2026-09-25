"""Depth images to points. numpy only, so perception and the viewer share it, on the
simulator and on the real arm alike.

A view is what a camera gives, the same dict on every backend and on hardware:

    dict(depth (H, W) metres, 0 = no reading,
         rgb   (H, W, 3) uint8, or None for a depth-only camera,
         K     (3, 3) intrinsics,
         pose  4x4 OPTICAL camera pose (+Z along the view, +X right, +Y down)
               in the frame the points should come out in)

Pixel (u, v) is sampled at its centre, (u + 0.5, v + 0.5), which is where the
renderer and the D435 both put the ray.
"""

import numpy as np


def backproject(depth, K, pose):
    """(N, 3) points and their (N,) flat pixel indices and (N,) depths, for readings > 0."""
    v, u = np.nonzero(depth > 0)
    z = depth[v, u].astype(np.float64)
    x = (u + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (v + 0.5 - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], axis=1) @ pose[:3, :3].T + pose[:3, 3]
    return pts, v * depth.shape[1] + u, z


def view_cloud(view, stride=4, depth_range=(0.0, np.inf)):
    """(N, 3) points of one view and their (N, 3) uint8 colours, or None without rgb.

    Every `stride`-th pixel in each direction: 4 keeps a 640x480 frame to about
    19 k points, plenty to see by and cheap to send to a browser.
    """
    depth = np.asarray(view["depth"])
    sub = np.zeros_like(depth)
    sub[::stride, ::stride] = depth[::stride, ::stride]
    pts, pix, z = backproject(sub, np.asarray(view["K"]), np.asarray(view["pose"]))
    keep = (z >= depth_range[0]) & (z <= depth_range[1])
    pts, pix = pts[keep], pix[keep]
    rgb = view.get("rgb")
    colors = None if rgb is None else np.asarray(rgb).reshape(-1, 3)[pix].astype(np.uint8)
    return pts, colors
