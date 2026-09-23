"""USD edits both simulator backends make to the imported robot.

sim_env.py (Isaac Sim) and lab/ (Isaac Lab) import the same URDF through two
different front ends, and then have to fix the same things on the stage: close
the gripper's loop, recover frames the importer merged away, put the pad
material on. Those fixes live here, once.

pxr and the Isaac modules can only be imported once a Kit app is running, so
every function imports what it needs itself. Importing this module needs
nothing.
"""

import numpy as np

from urdf_frames import gripper_pin_anchors, transform_to_ancestor, urdf_tree

# Payload and finger pads share this material. High friction on both is what
# makes a position-driven gripper hold rather than extrude what it squeezes.
GRIP_MATERIAL_PATH = "/World/physics/grip"
GRIP_MATERIAL = dict(static_friction=1.2, dynamic_friction=1.1, restitution=0.0)


# --- the Robotiq 2F-85's loop closure (gripper "linkage": "robotiq_2f85") ----
#
# The numbers for the arm and gripper -- drive gains, which joints the pin
# owns, pad geometry -- live in rig.ROBOTS with the measurements behind them.
# What stays here is code that only makes sense for this mechanism.
#
# The 2F-85 is two mirrored 4-bar linkages. A 4-bar needs a loop closure, and
# URDF is a tree, so in the model the inner knuckle hangs off the base as its
# own branch with nothing tying it to the finger tip. Under load the branches
# stall at different angles -- measured 0.22 rad apart within one side while
# gripping -- and the linkage visibly comes apart.
#
# USD is not a tree, so the pin can be added back here.
#
# Where the pin goes is READ OFF THE URDF rather than measured off the meshes.
# The mechanism is a parallelogram: the coupler's mimic multipliers are
# knuckle +1, finger_tip -1, so the finger tip's absolute orientation stays
# fixed, which is a parallelogram's defining property. With pivots
#
#     A = knuckle joint          C = finger_tip joint  (in base coords at 0)
#     B = inner_knuckle joint    D = the missing pin
#
# a parallelogram gives D = B + (C - A) exactly. No mesh fitting, no closest-
# surface-point search, and it stays right if the URDF is regenerated.
GRIPPER_PIN_AXIS = "Y"      # every 2F-85 joint turns about this link's Y


def close_gripper_linkage(stage, prim_path, urdf):
    """Pin each inner knuckle to its finger tip, closing the 4-bar in PhysX.

    Without this the knuckle is driven open-loop and fights whatever it
    touches. With it, the knuckle's angle comes from the mechanism, which is
    where it comes from on the real gripper.

    Idempotent: Define on an existing path re-authors the same joint.
    """
    from pxr import Gf, UsdPhysics

    anchors = gripper_pin_anchors(urdf)
    made = []
    for side, (on_knuckle, on_tip) in anchors.items():
        k = stage.GetPrimAtPath(f"{prim_path}/robotiq_85_{side}_inner_knuckle_link")
        f = stage.GetPrimAtPath(f"{prim_path}/robotiq_85_{side}_finger_tip_link")
        if not (k.IsValid() and f.IsValid()):
            print(f"[sim] cannot pin {side} linkage: link prim missing")
            continue
        path = f"{prim_path}/joints/{side}_linkage_pin"
        j = UsdPhysics.RevoluteJoint.Define(stage, path)
        j.CreateBody0Rel().SetTargets([k.GetPath()])
        j.CreateBody1Rel().SetTargets([f.GetPath()])
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(*on_knuckle.tolist()))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(*on_tip.tolist()))
        # Planar mechanism: the pin turns about the same axis the other
        # gripper joints do. Leaving the limits off keeps it a free hinge.
        j.CreateAxisAttr().Set(GRIPPER_PIN_AXIS)
        j.CreateExcludeFromArticulationAttr().Set(True)
        made.append(side)
    if made:
        one = anchors[made[0]]
        print(f"[sim] 4-bar closed: pinned {', '.join(made)} inner knuckle "
              f"to finger tip about {GRIPPER_PIN_AXIS}, "
              f"anchor {np.round(one[0], 5).tolist()} / "
              f"{np.round(one[1], 5).tolist()}")
    return made


# Gripper loop closures by rig.ROBOTS[...]["gripper"]["linkage"]. A gripper
# with a different mechanism adds its handler here -- same signature, returns
# what it made -- and neither backend changes.
LINKAGES = {
    "robotiq_2f85": close_gripper_linkage,
}


def apply_linkage(stage, prim_path, gripper, urdf, robot_key=""):
    """Run the loop closure rig.ROBOTS asks for, if any. Returns what it made."""
    linkage = gripper.get("linkage")
    if linkage is None:
        return []
    if linkage not in LINKAGES:
        raise ValueError(f"{robot_key}: unknown gripper linkage {linkage!r}")
    return LINKAGES[linkage](stage, prim_path, urdf)


def bind_pad_material(stage, prim_path, pad_links, material_path=GRIP_MATERIAL_PATH):
    """Bind the grip material to each pad link's colliders. Returns the links bound.

    Bound on the link's `collisions` prim with an authored binding rather than
    through a helper that walks the subtree: the importer makes its meshes
    instanceable, and a walker that skips instance proxies binds nothing.
    """
    from pxr import UsdShade

    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    bound = []
    for link in pad_links:
        p = stage.GetPrimAtPath(f"{prim_path}/{link}/collisions")
        if p.IsValid():
            UsdShade.MaterialBindingAPI(p).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")
            bound.append(link)
    return bound


def frame_prim(stage, prim_path, link, urdf):
    """Prim path for `link`, recreating it if merge_fixed_joints ate it.

    Merging keeps a prim for every link that carries geometry and discards the
    pure frames. Walk up the URDF to the nearest link that does have a prim,
    carry the transform along, and hang an Xform there. An Xform under a rigid
    body is just a frame -- it adds nothing for PhysX to solve, which is the
    whole point of having merged in the first place.

    Some merged frames are NOT discarded: the importer keeps them as an Xform
    child of the body they were merged into, with the joint's transform
    already on it -- grasp_frame comes out as robotiq_85_base_link/grasp_frame
    carrying its 0.130 m translate. Such a prim is used as it is. Defining it
    again and adding the chain's transform stacked a second copy of that
    offset on top, and the tool read 0.13 m past where it was.

    Returns (path of `link`, path of the body it was merged into).
    """
    from pxr import Gf, UsdGeom

    if stage.GetPrimAtPath(f"{prim_path}/{link}").IsValid():
        return f"{prim_path}/{link}", f"{prim_path}/{link}"
    tree = urdf_tree(urdf)
    # Kept by the importer under some ancestor body? Then it is already right.
    node = link
    while node in tree:
        parent = tree[node][0]
        kept = f"{prim_path}/{parent}/{link}"
        if stage.GetPrimAtPath(kept).IsValid():
            return kept, f"{prim_path}/{parent}"
        node = parent
    T, node = np.eye(4), link
    while node in tree:
        parent, M = tree[node]
        T = M @ T
        if stage.GetPrimAtPath(f"{prim_path}/{parent}").IsValid():
            host = f"{prim_path}/{parent}"
            xf = UsdGeom.Xform.Define(stage, f"{host}/{link}")
            R, t = T[:3, :3], T[:3, 3]
            # USD multiplies row vectors, so its matrix is the transpose of
            # this one with the translation along the bottom row.
            xf.AddTransformOp().Set(Gf.Matrix4d(
                float(R[0][0]), float(R[1][0]), float(R[2][0]), 0.0,
                float(R[0][1]), float(R[1][1]), float(R[2][1]), 0.0,
                float(R[0][2]), float(R[1][2]), float(R[2][2]), 0.0,
                float(t[0]), float(t[1]), float(t[2]), 1.0))
            return f"{host}/{link}", host
        node = parent
    raise RuntimeError(f"{link} is not in {urdf}, or has no ancestor with a prim")


def body_ancestor(stage, prim_path, link, urdf):
    """(body, T): the rigid body `link` ended up in, and its offset in it.

    Unlike frame_prim this creates nothing. It is for a backend that reads
    poses from the physics engine rather than from USD transforms, and so
    needs the surviving BODY plus a fixed offset, not a prim to read.
    """
    from pxr import UsdPhysics

    def is_body(name):
        p = stage.GetPrimAtPath(f"{prim_path}/{name}")
        return p.IsValid() and p.HasAPI(UsdPhysics.RigidBodyAPI)

    return transform_to_ancestor(urdf, link, is_body)


def draw_targets(targets):
    """Mark each goal in the viewport without putting anything in the scene.

    A viewport OVERLAY, not scene geometry: as geometry the depth cameras fuse
    the markers into the map as obstacles sitting exactly on the goals, and
    then every plan fails (measured: 193 of 199). An overlay is drawn by a
    separate pass that render products do not sample.

    Points plus a small axis cross, so a goal reads as a location rather than a
    stray dot. Redrawn from scratch each time, because the overlay accumulates.
    """
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()
    draw.clear_lines()
    blue = (0.05, 0.43, 0.62, 1.0)
    draw.draw_points([tuple(t[:3]) for t in targets], [blue] * len(targets),
                     [14.0] * len(targets))
    arm = 0.03
    starts, ends = [], []
    for t in targets:
        x, y, z = t[:3]
        for dx, dy, dz in ((arm, 0, 0), (0, arm, 0), (0, 0, arm)):
            starts.append((x - dx, y - dy, z - dz))
            ends.append((x + dx, y + dy, z + dz))
    draw.draw_lines(starts, ends, [blue] * len(starts), [2.0] * len(starts))
