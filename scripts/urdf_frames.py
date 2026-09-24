"""Frame arithmetic read straight off a URDF. numpy and the standard library only.

Both simulator backends need it (sim/sim_env.py on Isaac Sim, lab/ on Isaac Lab),
and neither may import the other, so it lives here. Nothing in this file needs
a running simulator.

Conventions: quaternions are (w, x, y, z); a rotation matrix's COLUMNS are the
frame's x, y, z axes; 4x4 transforms map child coordinates into the parent's.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np


# Optical (+Z view, +X right, +Y down) -> ROS body (+X view, +Y left, +Z up),
# as columns: body X = optical Z, body Y = -optical X, body Z = -optical Y.
# R_body = R_optical @ this.
OPTICAL_TO_ROS_BODY = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])


def gripper_pin_anchors(urdf_path):
    """Pin anchors, per side, in the inner knuckle's and finger tip's frames.

    The Robotiq 2F-85 is two mirrored parallelograms. With pivots

        A = knuckle joint          C = finger_tip joint  (in base coords at 0)
        B = inner_knuckle joint    D = the missing pin

    a parallelogram gives D = B + (C - A) exactly -- see sim_usd.close_gripper_linkage.

    Returns {side: ((x, y, z) on inner_knuckle, (x, y, z) on finger_tip)}.
    """
    root = ET.parse(urdf_path).getroot()
    origin = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = (o.get("xyz") if o is not None else None) or "0 0 0"
        origin[j.get("name")] = np.array([float(v) for v in xyz.split()])

    out = {}
    for side in ("left", "right"):
        p = f"robotiq_85_{side}_"
        try:
            A = origin[p + "knuckle_joint"]
            B = origin[p + "inner_knuckle_joint"]
            # C is the finger tip's pivot in BASE coordinates, so walk the
            # chain: the finger is fixed to the knuckle, the tip to the finger.
            C = A + origin[p + "finger_joint"] + origin[p + "finger_tip_joint"]
        except KeyError as missing:
            raise RuntimeError(
                f"{urdf_path} has no {missing}; the 4-bar cannot be closed")
        D = B + (C - A)
        out[side] = (D - B, D - C)
    return out


_URDF_TREES = {}


def urdf_tree(urdf):
    """child link -> (parent link, 4x4 transform), for the whole URDF."""
    if urdf not in _URDF_TREES:

        def rpy(r, p, y):
            cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                                      math.sin(p), math.cos(y), math.sin(y))
            return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                             [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                             [-sp, cp * sr, cp * cr]])

        tree = _URDF_TREES[urdf] = {}
        for j in ET.parse(urdf).getroot().findall("joint"):
            o = j.find("origin")
            T = np.eye(4)
            if o is not None:
                T[:3, 3] = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
                T[:3, :3] = rpy(*[float(v) for v in (o.get("rpy") or "0 0 0").split()])
            tree[j.find("child").get("link")] = (j.find("parent").get("link"), T)
    return _URDF_TREES[urdf]


def transform_to_ancestor(urdf, link, is_body):
    """(body, T): the nearest link at or above `link` that `is_body` accepts.

    T maps `link` coordinates into `body` coordinates. This is how a frame that
    fixed-joint merging folded away (grasp_frame, camera_link) is recovered:
    whichever body survived the merge, the fixed chain between the two is
    still written in the URDF.
    """
    tree = urdf_tree(urdf)
    T, node = np.eye(4), link
    while not is_body(node):
        if node not in tree:
            raise RuntimeError(f"{link} is not in {urdf}, or has no ancestor body")
        parent, M = tree[node]
        T = M @ T
        node = parent
    return node, T


def quat_to_matrix(q_wxyz):
    """Rotation matrix whose COLUMNS are the frame's x, y, z axes in world."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(m):
    """(w, x, y, z) from a rotation matrix, branching on the largest term.

    The trace branch alone loses precision, and divides by zero outright, for
    rotations near 180 degrees -- which is exactly what a straight-down camera
    is, so the branches matter here rather than being defensive boilerplate.
    """
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        sq = math.sqrt(t + 1.0) * 2
        return np.array([0.25 * sq, (m[2][1] - m[1][2]) / sq,
                         (m[0][2] - m[2][0]) / sq, (m[1][0] - m[0][1]) / sq])
    i = int(np.argmax([m[0][0], m[1][1], m[2][2]]))
    if i == 0:
        sq = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return np.array([(m[2][1] - m[1][2]) / sq, 0.25 * sq,
                         (m[0][1] + m[1][0]) / sq, (m[0][2] + m[2][0]) / sq])
    if i == 1:
        sq = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return np.array([(m[0][2] - m[2][0]) / sq, (m[0][1] + m[1][0]) / sq,
                         0.25 * sq, (m[1][2] + m[2][1]) / sq])
    sq = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return np.array([(m[1][0] - m[0][1]) / sq, (m[0][2] + m[2][0]) / sq,
                     (m[1][2] + m[2][1]) / sq, 0.25 * sq])


def split_transform(T):
    """(pos (3,), quat wxyz (4,)) of a 4x4 transform."""
    return np.asarray(T[:3, 3], dtype=float), matrix_to_quat(T[:3, :3])
