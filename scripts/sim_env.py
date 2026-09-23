"""The Isaac Sim side as a library: build the cell once, then drive it.

    from sim_env import launch, SimEnv, EnvCfg, ResetOptions
    launch(headless=False)          # must come before anything else from Isaac
    env = SimEnv(EnvCfg(scene="pick_place"))
    obs = env.reset(ResetOptions(block_on=0))
    env.move_to(target)             # planned by cuRobo, mapping along the way
    env.move_tool_z(-0.125)         # IK + interpolation for the last few cm
    env.grip(close=True)

This process runs Isaac Sim ONLY. It must never import cuRobo: Isaac Sim 5.1
requires Warp 1.8.2 and cuRobo 0.8 requires Warp >= 1.13, so they cannot live
in one interpreter. Planning and mapping happen in planner_server.py, reached
through planner_client.Planner.

Isaac Sim's own modules can only be imported once a SimulationApp exists, so
they are imported by launch(), not at the top of this file. Everything below
that touches them must run after launch().
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenes  # noqa: E402
from planner_client import Planner  # noqa: E402
from rig import DEFAULT_ROBOT, ROBOTS, SIM_DT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_APP = None
_HEADLESS = False


def launch(headless=False, width=1600, height=900):
    """Start Isaac Sim and import its modules. Idempotent; returns the app."""
    global _APP, _HEADLESS
    global omni, Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade
    global UrdfJointTargetType, World, PhysicsMaterial
    global DynamicCuboid, FixedCuboid, VisualCuboid, SingleArticulation
    global ArticulationAction, set_camera_view, Camera, _debug_draw
    if _APP is not None:
        return _APP
    from isaacsim import SimulationApp
    _APP = SimulationApp({"headless": headless, "width": width, "height": height})
    _HEADLESS = headless

    import omni.kit.commands
    from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade
    from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType
    from isaacsim.core.api import World
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.sensors.camera import Camera
    from isaacsim.util.debug_draw import _debug_draw
    return _APP


# The gripper joints' <limit velocity="..."> in the URDF. Used to budget how
# many sim steps a full open or close actually needs; commanding it faster
# just leaves the joint short of the target when the next plan starts.
GRIPPER_RAD_PER_S = 2.0

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

# The joints the PIN is responsible for rather than the drive. They are not
# commanded, and they must not be HELD either: the importer gives every joint
# a position drive at default_drive_strength, so left alone they are pinned to
# zero at 1e6 stiffness and spend the whole grasp fighting the loop closure.
# That fight is what makes the linkage visibly come apart.
#
# Measured in free air, closing onto nothing, as the spread across the six
# joints expressed as a fraction of a full close:
#
#     pin only, follower drives left alone : 0.805..0.922, spread 0.118
#     pin, follower drives zeroed          : 0.997..1.000, spread 0.003
#
# The second also reaches its commanded angle, which the first never does.
PINNED_FOLLOWER = "inner_knuckle_joint"

# Drive stiffness for the gripper joints alone.
#
# The importer gives EVERY joint default_drive_strength, which is 1e6 because
# the arm needs it. A 2F-85 link weighs 14 g, and a 1e6 drive on it simply
# overpowers the 4-bar's loop closure: the links stop agreeing with each other
# and the linkage visibly comes apart. Measured as the spread across the six
# joints while gripping a 45 mm block, which is the thing you can see:
#
#     1e6 -> 0.201     1e4 -> 0.191     1e2 -> 0.064
#     1e5 -> 0.197     1e3 -> 0.099
#
# Monotonic, and nothing else moved it: solver iterations (64, 255) changed it
# by 0.000, and driving two joints instead of four by 0.005.
PLANNED_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

# The importer's own stiffness, kept; the damping it does NOT give.
#
# A position drive with no damping is an undamped spring, and it rings around
# a moving target: during a 12 cm vertical move the joints that were told to
# hold still buzzed instead, reversing direction 37-47 times, and the two that
# were moving ran in surges rather than at speed. Measured over that move, as
# velocity sign flips across all six joints and the velocity ripple on
# wrist_1:
#
#     damping    0 -> 138 flips, ripple 0.48, lag  8 mrad
#               20 ->  19 flips, ripple 0.13, lag  9 mrad
#               50 ->   0 flips, ripple 0.19, lag 22 mrad
#              150 ->   0 flips, ripple 0.20, lag 66 mrad
#
# 20 is the knee: the joints that should not move stop moving (|v| max
# 0.00 rad/s, the remaining flips are noise below 0.005), and it costs almost
# nothing in tracking. 50 buys only sub-visible noise for four times the lag.
ARM_DRIVE_STIFFNESS = 625.0
ARM_DRIVE_DAMPING = 20.0

GRIPPER_DRIVE_STIFFNESS = 1.0e2
GRIPPER_DRIVE_DAMPING = 1.0e1


def _gripper_pin_anchors(urdf_path):
    """Pin anchors, per side, in the inner knuckle's and finger tip's frames.

    Returns {side: ((x, y, z) on inner_knuckle, (x, y, z) on finger_tip)}.
    """
    import xml.etree.ElementTree as ET

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


def _urdf_tree(urdf):
    """child link -> (parent link, 4x4 transform), for the whole URDF."""
    if urdf not in _URDF_TREES:
        import xml.etree.ElementTree as ET

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
    if stage.GetPrimAtPath(f"{prim_path}/{link}").IsValid():
        return f"{prim_path}/{link}", f"{prim_path}/{link}"
    tree = _urdf_tree(urdf)
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


def close_gripper_linkage(stage, prim_path, urdf):
    """Pin each inner knuckle to its finger tip, closing the 4-bar in PhysX.

    Without this the knuckle is driven open-loop and fights whatever it
    touches. With it, the knuckle's angle comes from the mechanism, which is
    where it comes from on the real gripper.
    """
    anchors = _gripper_pin_anchors(urdf)
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


def tune_arm_drives(stage, joints):
    """Give the arm's position drives some damping. They ship with none.

    Isaac Sim 5.x ignores the importer's default_drive_strength and
    default_position_drive_damping entirely -- every joint comes out at
    stiffness 625 and damping 0 whatever is asked for. An undamped position
    drive rings around a moving target, which is what makes a slow vertical
    move judder.
    """
    k, d = ARM_DRIVE_STIFFNESS, ARM_DRIVE_DAMPING
    n = 0
    for prim in stage.Traverse():
        if prim.GetName() not in joints:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            drive.CreateStiffnessAttr().Set(k)
            drive.CreateDampingAttr().Set(d)
            n += 1
    print(f"[sim] arm drives: {n} at stiffness {k:g}, damping {d:g}")


def tune_gripper_drives(stage, prim_path, joints):
    """Soften the gripper's drives, and release the ones the pin owns.

    Two separate things, both about the same 1e6 default: the followers must
    not be held at all, and the rest must not be held hard enough to tear the
    linkage apart.
    """
    freed, softened = [], []
    for prim in stage.Traverse():
        name = prim.GetName()
        if name not in joints:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            continue
        if PINNED_FOLLOWER in name:
            drive.CreateStiffnessAttr().Set(0.0)
            drive.CreateDampingAttr().Set(0.0)
            freed.append(name)
        else:
            drive.CreateStiffnessAttr().Set(GRIPPER_DRIVE_STIFFNESS)
            drive.CreateDampingAttr().Set(GRIPPER_DRIVE_DAMPING)
            softened.append(name)
    print(f"[sim] gripper drives: {len(softened)} at "
          f"{GRIPPER_DRIVE_STIFFNESS:g}/{GRIPPER_DRIVE_DAMPING:g}, "
          f"{len(freed)} released to the 4-bar")
    return freed


def build_stage(world, scene, robot_key, urdf):
    """Import the arm from URDF, add obstacles, and hide one from the planner."""
    status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    # MERGE the fixed joints. This URDF has 26 links that are pure coordinate
    # frames -- tool0, flange, grasp_frame, camera_link and the D435i's six
    # optical frames among them -- with no mass and no geometry. Imported as
    # separate bodies the importer gives each "a small isotropic inertia", and
    # a chain of twenty near-massless bodies hanging off the wrist wrecks the
    # articulation solver: holding HOME, wrist_1 drifted up to 770 mrad while
    # every other joint stayed inside 5 mrad. Merging drops that to 4.4 mrad.
    # Measured both ways; the 4-bar pin was ruled out first (663 mrad with it
    # removed).
    #
    # Fixed joints are not degrees of freedom, so nothing is lost mechanically
    # -- but the merged frames have no prims any more, which is what
    # frame_prim() below exists to paper over.
    cfg.merge_fixed_joints = True
    cfg.fix_base = True
    cfg.make_default_prim = False
    cfg.create_physics_scene = False
    cfg.distance_scale = 1.0
    cfg.default_drive_type = UrdfJointTargetType.JOINT_DRIVE_POSITION
    # These two are DEAD in Isaac Sim 5.x. Whatever they are set to, every
    # joint comes out at stiffness 625 and damping 0 -- verified by reading
    # the DriveAPI back after import at 1e6 and at 1e5, identical both times.
    # tune_arm_drives() below is what actually sets the gains.
    cfg.default_drive_strength = 1e6
    cfg.default_position_drive_damping = 1e5
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=urdf, import_config=cfg
    )
    print(f"[sim] {robot_key} imported at {prim_path}")

    world.scene.add_default_ground_plane()
    for name, dims, pose, colour in scene.obstacles:
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/obstacles/{name}",
                name=name,
                position=np.array(pose[:3]),
                scale=np.array(dims),
                color=np.array(colour),
            )
        )
    # Target markers are drawn as a VIEWPORT OVERLAY, not as scene geometry.
    #
    # They only show where the tool is being sent, and they must not reach the
    # depth image: as geometry the cameras fuse them into the map as obstacles
    # sitting exactly on the goals, and then every plan fails because the goal
    # is inside an obstacle. Measured with no other obstacle in the scene at
    # all: 193 of 199 plans failed, and the only voxels above the table were
    # these two markers.
    #
    # USD's `purpose = guide` also keeps them out of the depth image, and was
    # the first fix here, but guides are hidden in the viewport too -- which
    # left a correct map and nothing for a person to look at. An overlay is
    # drawn by a separate pass that render products do not sample, so it solves
    # both halves: visible to you, invisible to the cameras. Goals that stay
    # reachable once the map is live are what prove the second half.
    draw_targets(scene.targets)

    # The bodies the planner is never told about. VisualCuboids, not physics
    # bodies, so dragging one in the viewport does not fight PhysX -- it still
    # renders into depth, which is all the cameras need. That does mean the arm
    # passes through rather than hitting it.
    #
    # These deliberately keep their default render purpose, unlike the target
    # markers above: being seen is the entire point of them.
    bodies = {}
    for name, dims, pose, colour in scene.unmapped:
        bodies[name] = VisualCuboid(
            prim_path=f"/World/{name}",
            name=name,
            position=np.array(pose[:3]),
            scale=np.array(dims),
            color=np.array(colour),
        )

    close_gripper_linkage(world.stage, prim_path, urdf)
    tune_arm_drives(world.stage, set(PLANNED_JOINTS))
    tune_gripper_drives(world.stage, prim_path,
                        set(ROBOTS[robot_key]["gripper_joints"]))

    # Payloads. Real rigid bodies, unlike everything above: they fall, they can
    # be squeezed, and they come away when the gripper closes. High friction on
    # both the block and the finger pads is what makes a position-driven
    # gripper hold rather than extrude what it is squeezing.
    payload = {}
    if scene.payload:
        grip_mat = PhysicsMaterial(prim_path="/World/physics/grip",
                                   static_friction=1.2, dynamic_friction=1.1,
                                   restitution=0.0)
        for name, dims, pose, colour, mass in scene.payload:
            cube = DynamicCuboid(
                prim_path=f"/World/{name}",
                name=name,
                position=np.array(pose[:3]),
                scale=np.array(dims),
                color=np.array(colour),
                mass=mass,
            )
            cube.apply_physics_material(grip_mat)
            payload[name] = cube
        # The finger tip carries the rubber pad; this model has no separate
        # pad link, so the friction goes on the tip itself.
        for link in ("robotiq_85_left_finger_tip_link",
                     "robotiq_85_right_finger_tip_link"):
            p = world.stage.GetPrimAtPath(f"{prim_path}/{link}/collisions")
            if p.IsValid():
                UsdShade.MaterialBindingAPI(p).Bind(
                    UsdShade.Material(world.stage.GetPrimAtPath("/World/physics/grip")),
                    UsdShade.Tokens.weakerThanDescendants, "physics")

    light = UsdLux.DistantLight.Define(world.stage, "/World/DistantLight")
    light.CreateIntensityAttr(2500)
    light.CreateAngleAttr(1.0)
    return prim_path, bodies, payload


def draw_targets(targets):
    """Mark each goal in the viewport without putting anything in the scene.

    Points plus a small axis cross, so a goal reads as a location rather than a
    stray dot. Redrawn from scratch each time, because the overlay accumulates.
    """
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


def attach_wrist_camera(world, prim_path, spec, urdf):
    """Put a D435i-like camera on the URDF's camera_link.

    camera_link is cuRobo's OPTICAL frame: +Z forward, +X right, +Y down.
    isaacsim's Camera wrapper does NOT use the raw USD -Z convention -- it takes
    orientation in ROS body axes, where the view direction is +X, +Y is left and
    +Z is up. The two differ by a 120 degree rotation, not a flip, so the
    180-about-X "correction" this used to apply was a no-op on the view axis and
    left the camera staring sideways down its own wrist.

    Mapping body axes onto the optical frame:
        body +X (view) = optical +Z
        body +Y (left) = -optical +X
        body +Z (up)   = -optical +Y
    which is the quaternion below. Verified at runtime by the check further
    down, which prints the view direction in camera_link axes.
    """
    link, body = frame_prim(world.stage, prim_path, spec["link"], urdf)

    cam = Camera(
        prim_path=f"{link}/d435i",
        resolution=(spec["width"], spec["height"]),
        translation=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.5, 0.5, -0.5, 0.5]),  # optical -> ROS body, wxyz
    )
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()

    aperture = 20.955  # USD default horizontal aperture, mm
    focal = aperture / (2.0 * math.tan(math.radians(spec["horizontal_fov_deg"]) / 2.0))
    cam.set_focal_length(focal / 10.0)  # Camera API works in cm
    cam.set_clipping_range(spec["near"], spec["far"])
    # Verify, don't assume: a USD camera looks along its own -Z, but the
    # isaacsim Camera wrapper may already account for that. Compare the prim's
    # actual view direction against camera_link's optical +Z and say so.
    cache = UsdGeom.XformCache()
    m_link = cache.GetLocalToWorldTransform(world.stage.GetPrimAtPath(link))
    m_cam = cache.GetLocalToWorldTransform(world.stage.GetPrimAtPath(f"{link}/d435i"))
    def axis(m, i):
        r = m.ExtractRotationMatrix()
        return np.array([r[i][0], r[i][1], r[i][2]])
    view = -axis(m_cam, 2)          # USD camera looks down its own -Z
    optical = axis(m_link, 2)       # camera_link's +Z is the optical axis
    dot = float(np.dot(view, optical))
    basis = np.stack([axis(m_link, 0), axis(m_link, 1), axis(m_link, 2)])
    local = basis @ view            # view direction expressed in camera_link
    print(f"[sim]   view in camera_link axes: {local.round(3)}  "
          f"(want [0 0 1])")
    # Image-horizontal should run along the camera body's long edge, which is
    # +Y of the D435i's own link. Check the roll, not just the view direction.
    m_body = cache.GetLocalToWorldTransform(world.stage.GetPrimAtPath(body))
    img_right = axis(m_link, 0)     # optical +X after the URDF yaw
    bar = axis(m_body, 1)           # long edge of the D435i body
    # Signed, not |dot|: +1 and -1 both mean "aligned with the long edge" but
    # they differ by a 180 degree roll, i.e. an upside-down image. An earlier
    # version compared magnitudes and happily passed the flipped one.
    print(f"[sim]   image-right vs body long edge: dot = "
          f"{float(np.dot(img_right, bar)):+.3f}  (want -1.000)")
    print(f"[sim] wrist camera on {spec['link']}: "
          f"{spec['width']}x{spec['height']}, {spec['horizontal_fov_deg']:.0f} deg HFOV")
    print(f"[sim] view-vs-optical dot = {dot:+.3f}  "
          f"({'AGREE' if dot > 0.9 else 'REVERSED' if dot < -0.9 else 'PERPENDICULAR'})")
    print(f"[sim]   usd camera view dir (world): {view.round(3)}")
    print(f"[sim]   camera_link +Z    (world): {optical.round(3)}")
    return cam


# Optical (+Z view, +X right, +Y down) -> ROS body (+X view, +Y left, +Z up),
# as columns: body X = optical Z, body Y = -optical X, body Z = -optical Y.
# R_body = R_optical @ this.
OPTICAL_TO_ROS_BODY = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])


def _quat_to_matrix(q_wxyz):
    """Rotation matrix whose COLUMNS are the frame's x, y, z axes in world."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat(m):
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


def attach_overhead_camera(world, spec):
    """Fixed camera on a gantry above the cell, looking straight down.

    The scene stores the OPTICAL pose (+Z along the view), because that is what
    cuRobo's mapper kernels consume and what the planner server hands to
    CameraObservation verbatim. Isaac Sim wants ROS BODY axes, so the body
    quaternion is DERIVED here instead of being written down a second time:
    moving the camera in the scene moves it here too, with no second set of
    magic numbers to keep in sync.

    This camera exists because a wrist camera cannot see above its own
    altitude, which is what forced the demo obstacle to be shaped to suit the
    sensor rather than the other way round.
    """
    pose = spec["pose"]
    r_opt = _quat_to_matrix(pose[3:])
    q_body = _matrix_to_quat(r_opt @ OPTICAL_TO_ROS_BODY)

    path = "/World/overhead_cam"
    cam = Camera(
        prim_path=path,
        resolution=(spec["width"], spec["height"]),
        position=np.array(pose[:3]),
        orientation=q_body,
    )
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()

    aperture = 20.955  # USD default horizontal aperture, mm
    focal = aperture / (2.0 * math.tan(math.radians(spec["horizontal_fov_deg"]) / 2.0))
    cam.set_focal_length(focal / 10.0)  # Camera API works in cm
    cam.set_clipping_range(spec["near"], spec["far"])

    # Verify, don't assume. The wrist camera's "obvious" 180-about-X correction
    # turned out to be a no-op on the view axis, so every camera here states
    # what it expects and checks it against the prim's real transform.
    cache = UsdGeom.XformCache()
    m_cam = cache.GetLocalToWorldTransform(world.stage.GetPrimAtPath(path))
    r = m_cam.ExtractRotationMatrix()
    view = -np.array([r[2][0], r[2][1], r[2][2]])  # USD camera looks down -Z
    want = r_opt[:, 2]                             # optical +Z from the scene
    dot = float(np.dot(view, want))
    print(f"[sim] overhead camera at {np.array(pose[:3]).round(3)}: "
          f"{spec['width']}x{spec['height']}, {spec['horizontal_fov_deg']:.0f} deg HFOV")
    print(f"[sim]   view dir (world): {view.round(3)}  want {want.round(3)}  "
          f"dot = {dot:+.3f}  ({'AGREE' if dot > 0.99 else 'WRONG'})")
    if dot < 0.99:
        raise RuntimeError(
            f"overhead camera is not pointing where the scene says: view {view} "
            f"vs optical +Z {want}. Fix the conversion, do not adjust the check."
        )
    return cam


def grab_depth(cam):
    """Perpendicular depth in metres, or None until the annotator has data."""
    frame = cam.get_current_frame()
    depth = frame.get("distance_to_image_plane")
    if depth is None:
        return None
    depth = np.asarray(depth, dtype=np.float32)
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


# --- the interface ------------------------------------------------------------


@dataclass
class EnvCfg:
    scene: str = scenes.DEFAULT
    robot: str = DEFAULT_ROBOT
    mapping: bool = True        # False: no cameras, the planner sees the static scene
    overhead: bool = True       # the fixed camera, on top of the wrist one
    map_every: int = 6          # fuse a frame every N sim steps while moving
    depth_lag: int = 2          # sim steps the depth annotator trails the physics by
    verbose: bool = True


@dataclass
class ResetOptions:
    block_on: int = 0                       # which target the payload starts under
    slab_pose: Optional[List[float]] = None  # [x, y, z] for the unmapped body; None = scene's
    clear_map: bool = True                  # empty the server's map first
    scan: bool = True                       # sweep scene.scan_poses to seed the map
    settle_steps: int = 60


@dataclass
class MoveResult:
    ok: bool
    reason: Optional[str] = None     # why not, in the server's words
    solve_ms: float = 0.0
    waypoints: int = 0
    clearance: Optional[str] = None  # closest approach to unmapped bodies, as reported
    sim_steps: int = 0


@dataclass
class GripResult:
    settled: bool                    # False: still creeping when the budget ran out
    steps: int
    target: float                    # commanded leader angle, rad
    error: float                     # worst driven joint's distance from it
    spread: Dict[str, float] = field(default_factory=dict)  # every joint, as fraction closed


@dataclass
class Observation:
    q: np.ndarray                    # arm joints, in the planner's order
    gripper: float                   # leader joint, rad (0 open)
    tool_pose: np.ndarray            # tool frame [x, y, z, qw, qx, qy, qz], from the stage
    objects: Dict[str, np.ndarray]   # payload poses, same layout -- ground truth
    holding: bool
    sim_time: float
    depth: Optional[Dict[str, np.ndarray]] = None


class SimEnv:
    """One cell: the arm, its cameras, the scene. Primitives, not a task.

    Every method steps the simulation itself and returns when it is done, so a
    caller (the demo loop, or a learning loop deciding where to go next) only
    ever sees the world between motions.
    """

    def __init__(self, cfg: EnvCfg):
        if _APP is None:
            raise RuntimeError("call sim_env.launch() before building a SimEnv")
        self.cfg = cfg
        self.scene = scenes.load(cfg.scene)
        self.spec = ROBOTS[cfg.robot]
        self.urdf = f"{ROOT}/{self.spec['urdf']}"
        self.log(f"robot: {cfg.robot}  urdf: {os.path.basename(self.urdf)}")
        try:
            self.planner = Planner(cfg.scene, log=self.log)
        except Exception:
            _APP.close()
            raise

        self.world = World(physics_dt=SIM_DT, rendering_dt=SIM_DT,
                           stage_units_in_meters=1.0)
        prim_path, self.bodies, self.payload = build_stage(
            self.world, self.scene, cfg.robot, self.urdf)
        self.prim_path = prim_path
        if not _HEADLESS:
            set_camera_view(eye=[2.0, 1.6, 1.4], target=[0.35, 0.0, 0.35])

        self.robot = SingleArticulation(prim_path=prim_path, name=cfg.robot)
        self.world.scene.add(self.robot)
        self.cams = {}
        if cfg.mapping:
            self.cams["wrist"] = attach_wrist_camera(
                self.world, prim_path, self.scene.cameras["wrist"], self.urdf)
            if cfg.overhead:
                self.cams["overhead"] = attach_overhead_camera(
                    self.world, self.scene.cameras["overhead"])
        # A frame to read the tool pose from. merge_fixed_joints leaves the
        # tool frame without a prim of its own; frame_prim hangs one back on.
        self._tool_prim, _ = frame_prim(self.world.stage, prim_path,
                                        self.spec["tool_frame"], self.urdf)
        self.world.reset()
        self.robot.initialize()
        # Rendering is only needed for the cameras, or for a person to watch.
        self._render = bool(self.cams) or not _HEADLESS

        px = PhysxSchema.PhysxArticulationAPI.Get(self.world.stage, prim_path)
        if px:
            px.CreateSolverPositionIterationCountAttr(64)
            px.CreateSolverVelocityIterationCountAttr(16)

        probe = self.planner.plan(self.scene.home, self.scene.targets[0])
        curobo_names = probe["joint_names"]
        sim_names = list(self.robot.dof_names)

        # The simulator has more joints than the planner: the gripper is
        # articulated in the URDF but locked out of cuRobo's cspace, so it is
        # driven here on its own channel. Address the two sets by INDEX rather
        # than reindexing whole arrays -- an arm command must never disturb the
        # fingers, and a joint the planner has never heard of must not be looked
        # up in its name list.
        missing = [j for j in curobo_names if j not in sim_names]
        if missing:
            raise RuntimeError(f"planner joints absent from the simulator: {missing}")
        self.arm_idx = [sim_names.index(j) for j in curobo_names]

        coupling_all = self.spec["gripper_joints"]
        grip_name = next(iter(coupling_all))
        if grip_name not in sim_names:
            raise RuntimeError(f"gripper joint {grip_name} absent from the simulator")
        self.grip_idx = sim_names.index(grip_name)
        # The whole linkage, driven explicitly. The URDF carries no <mimic> tags:
        # PhysX refuses to build the constraint ("needs a finite limit set to be
        # used by the mimic joint feature", although every one of them has
        # lower="0" upper="0.8757") and that failure takes the articulation with
        # it -- the fingers come apart on screen. Verified by stripping the tags:
        # the PhysX error count goes 1 -> 0 and the articulation builds. The
        # coupling therefore lives in rig.ROBOTS and is applied here.
        #
        # Worth remembering that cuRobo showed a perfectly correct gripper
        # throughout, because it resolves this itself and never goes near PhysX.
        # Checking the planner's collision spheres is not checking the simulator.
        # The inner knuckles are NOT driven: close_gripper_linkage pins them to
        # their finger, so the mechanism sets their angle the way it does on the
        # real gripper. Driving them as well would fight that pin.
        coupling = {j: m for j, m in coupling_all.items()
                    if j in sim_names and PINNED_FOLLOWER not in j}
        self.grip_idx_all = np.array([sim_names.index(j) for j in coupling])
        # Every gripper joint, followers included, for reporting. Signed by its
        # multiplier so all six read as "fraction closed" and can be compared.
        self.all_grip_names = [j for j in coupling_all if j in sim_names]
        self.all_grip_idx = np.array([sim_names.index(j) for j in self.all_grip_names])
        self.all_grip_sign = np.array([coupling_all[j] for j in self.all_grip_names],
                                      dtype=np.float32)
        self.grip_mult = np.array(list(coupling.values()), dtype=np.float32)

        self.grip_open = self.spec["gripper_open"]
        self.grip_closed = self.spec["gripper_closed"]
        self.log(f"joints: {len(sim_names)} in sim, {len(curobo_names)} planned")
        self.log(f"gripper: {len(coupling)} load-bearing joints driven, "
                 f"{self.grip_open} open .. {self.grip_closed} closed; "
                 f"inner knuckles set by the pinned 4-bar")

        self.steps = 0
        self.q_history = []
        self._goal = None          # last commanded tool pose; move_tool_z is relative to it
        self._closed = False
        self._home()
        for _ in range(60):
            self._step()

        self.intrinsics = {}
        if self.cams:
            for _ in range(10):  # let the annotators produce their first frame
                self._step()
            for name, c in self.cams.items():
                k = c.get_intrinsics_matrix()
                self.intrinsics[name] = k
                self.log(f"{name} intrinsics from Isaac Sim: fx={k[0,0]:.1f} "
                         f"fy={k[1,1]:.1f} cx={k[0,2]:.1f} cy={k[1,2]:.1f}")
        # Where each payload spawns: reset() puts it back at this height.
        self._payload_home = {b[0]: np.array(b[2][:3], dtype=np.float32)
                              for b in self.scene.payload}

    # --- plumbing -----------------------------------------------------------

    def log(self, msg):
        if self.cfg.verbose:
            print(f"[sim] {msg}", flush=True)

    @property
    def running(self):
        return _APP.is_running()

    @property
    def gripper_closed(self):
        """Whether the last grip() closed -- on the block or on nothing."""
        return self._closed

    def close(self):
        _APP.close()

    def _step(self):
        self.world.step(render=self._render)
        self.steps += 1

    def command_arm(self, q_curobo):
        """Send one planner-ordered joint vector, touching nothing else."""
        self.robot.apply_action(ArticulationAction(
            joint_positions=np.asarray(q_curobo, dtype=np.float32),
            joint_indices=np.asarray(self.arm_idx)))

    def command_gripper(self, angle):
        """One commanded angle -> every joint of the linkage.

        Open-loop on purpose. Slaving the followers to the leader's MEASURED
        angle was tried and measured worse: under load the left/right spread
        went from 0.146 to 0.297 rad, because a follower commanded to the
        leader's position still cannot get there -- its own contact is what
        stops it, not its command.

        The linkage visibly separates while gripping, by up to 0.22 rad within
        one side. See HANDOVER section 6: a 4-bar needs a loop closure URDF
        cannot express, and six independent position drives is not it.
        """
        self.robot.apply_action(ArticulationAction(
            joint_positions=(self.grip_mult * angle).astype(np.float32),
            joint_indices=self.grip_idx_all))

    def q_now(self):
        q_sim = self.robot.get_joint_positions()
        return [float(q_sim[i]) for i in self.arm_idx]

    def _home(self):
        """Teleport the arm to HOME with the gripper open, and hold it there."""
        full = np.array(self.robot.get_joint_positions(), dtype=np.float32)
        for k, i in enumerate(self.arm_idx):
            full[i] = self.scene.home[k]
        full[self.grip_idx] = self.grip_open
        self.robot.set_joint_positions(full)
        self.robot.set_joint_velocities(np.zeros_like(full))
        self.command_arm(self.scene.home)
        self.command_gripper(self.grip_open)
        self._closed = False
        self._goal = None

    def fuse(self, require_still=False):
        """Fuse one frame, pairing the depth with the pose it was rendered at.

        Isaac Sim's depth annotator trails the physics by a step or two, so
        pairing a frame with the CURRENT joint state misaligns the self-mask
        and the arm smears its own image into the map as phantom obstacles.
        Gating on "arm is stationary" avoids that but starves the map (measured:
        420 voxels instead of 6500). Compensating the lag instead keeps the
        coverage and the alignment.

        The lag applies to the FIXED camera too. It never moves, so its own
        pose needs no correction, but the arm inside its view does -- and that
        is what the self-mask is aligned against.

        Returns how many frames were sent. The send is fire-and-forget, so this
        says nothing about what the mapper made of them.
        """
        if not self.cams:
            return 0
        if require_still:
            v = self.robot.get_joint_velocities()
            if v is not None and float(np.abs(np.asarray(v)).max()) > 0.05:
                return 0
        lag = self.cfg.depth_lag
        q_lagged = self.q_history[-1 - lag] if len(self.q_history) > lag \
            else self.q_now()
        sent = 0
        for name, c in self.cams.items():
            depth = grab_depth(c)
            if depth is None:
                continue
            self.planner.map_frame(q_lagged, depth, self.intrinsics[name], name)
            sent += 1
        return sent

    # --- primitives ---------------------------------------------------------

    def reset(self, options: Optional[ResetOptions] = None) -> Observation:
        """Arm to HOME, gripper open, payload onto a target, map rebuilt.

        The payload goes under target `block_on`, at the height the scene
        spawns it at -- which, for pick_place, is sitting on that pedestal.
        """
        o = options or ResetOptions()
        self._home()
        for name, cube in self.payload.items():
            home = self._payload_home[name]
            t = self.scene.targets[o.block_on]
            cube.set_world_pose(position=np.array([t[0], t[1], home[2]]),
                                orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))
        if o.slab_pose is not None:
            if not self.bodies:
                raise ValueError("slab_pose given, but the scene has no unmapped body")
            prim = next(iter(self.bodies.values()))
            prim.set_world_pose(position=np.array(o.slab_pose[:3]))
        if o.clear_map:
            self.planner.reset_map()
        self.q_history.clear()
        for _ in range(o.settle_steps):
            self._step()
        if o.scan and self.cams:
            self.scan()
        return self.observe()

    def scan(self):
        """Sweep scene.scan_poses so the map has content before the first plan."""
        self.log("scanning the cell before planning...")
        fused = 0
        for pose in self.scene.scan_poses:
            for step in range(70):
                if not self.running:
                    break
                blend = min(1.0, (step + 1) / 50.0)
                cmd = [(1 - blend) * a + blend * b for a, b in zip(self.q_now(), pose)]
                self.command_arm(cmd)
                self._step()
                self.q_history.append(self.q_now())
                if step % self.cfg.map_every == 0:
                    fused += self.fuse()
        fused += self.fuse()
        self.log(f"scan done: {fused} frames sent to the mapper")
        return fused

    def move_to(self, pose) -> MoveResult:
        """Plan from where the arm is to a tool pose, and execute it.

        Depth keeps being fused along the way. Without that the unmapped body
        decays out of the map between plans and the next one drives through it.
        """
        res = self.planner.plan(self.q_now(), list(pose))
        if not res.get("ok"):
            return MoveResult(ok=False, reason=res.get("goal", res.get("status")))
        traj = res["traj"]
        start = self.steps
        for k, wp in enumerate(traj):
            if not self.running:
                return MoveResult(ok=False, reason="simulation closed")
            self.command_arm(wp)
            self._step()
            self.q_history.append(self.q_now())
            if self.cams and k % self.cfg.map_every == 0:
                self.fuse()
        self._goal = list(pose)
        return MoveResult(ok=True, solve_ms=res["solve_ms"], waypoints=len(traj),
                          clearance=res.get("clearance"),
                          sim_steps=self.steps - start)

    def move_tool_z(self, dz, n_steps=70) -> MoveResult:
        """Move the tool dz straight up or down from the last goal, by IK.

        plan_pose is no use here: the block is in the map, and the planner will
        not route a tool into an obstacle. Asking for IK at the end pose keeps
        both ENDPOINTS exact and only the path between them unchecked -- which
        over 12 cm of vertical move is acceptable, and over a long one would
        not be.

        Relative to the last commanded goal, not the measured tool pose, so
        tracking error does not accumulate across a descend and a lift.
        """
        if self._goal is None:
            return MoveResult(ok=False, reason="no goal yet: move_to first")
        goal = list(self._goal)
        goal[2] += dz
        q_goal = self.planner.ik(self.q_now(), goal)
        if q_goal is None:
            self.log(f"  no IK for z{dz:+.3f} m; skipping")
            return MoveResult(ok=False, reason="no IK")
        start = self.q_now()
        # A 12 cm vertical move is a small joint move. If it is not, the IK
        # came back on a different branch from the one the arm is standing in,
        # and blending straight to it sweeps the arm through that difference
        # instead of going down.
        step = [abs(b - a) for a, b in zip(start, q_goal)]
        self.log(f"  z{dz:+.3f}: joint travel "
                 f"{' '.join(f'{v:.3f}' for v in step)} rad"
                 f"   worst {max(step):.3f} over {n_steps} steps "
                 f"({max(step) / (n_steps * SIM_DT):.2f} rad/s)")
        t0 = self.steps
        for i in range(n_steps):
            if not self.running:
                return MoveResult(ok=False, reason="simulation closed")
            blend = (i + 1) / n_steps
            self.command_arm([a + (b - a) * blend for a, b in zip(start, q_goal)])
            self._step()
        self._goal = goal
        return MoveResult(ok=True, sim_steps=self.steps - t0)

    def grip(self, close: bool) -> GripResult:
        """Stroke the gripper open or closed, then wait for it to stop."""
        stroke = int(abs(self.grip_closed - self.grip_open) / GRIPPER_RAD_PER_S / SIM_DT) + 6
        for i in range(stroke):
            f = (i + 1) / stroke if close else 1 - (i + 1) / stroke
            self.command_gripper(self.grip_open + (self.grip_closed - self.grip_open) * f)
            self._step()
        # Do not move until the fingers have actually stopped.
        end = self.grip_closed if close else self.grip_open
        waited, quiet, err = self._settle_gripper(end)
        self._closed = close
        # Every joint, not just the worst: a linkage that has come apart and a
        # linkage that has simply stalled on the object look identical in a
        # single number, and they need opposite fixes.
        q_all = self.robot.get_joint_positions()[self.all_grip_idx]
        spread = {n.split("robotiq_85_")[-1][:-6]: float(v * m)
                  for n, v, m in zip(self.all_grip_names, q_all, self.all_grip_sign)}
        return GripResult(settled=quiet, steps=waited, target=end, error=err,
                          spread=spread)

    def _settle_gripper(self, target, max_steps=240, quiet_steps=12, tol=2.0e-4):
        """Step until the driven gripper joints stop moving.

        The stroke is commanded open-loop over a fixed number of steps, which
        is right in free air and wrong on an object: the fingers stall against
        it and the joints lag their command -- measured at 0.365 rad while
        squeezing. Lifting on the ramp's last step therefore lifts before the
        grip has closed, and the block gets pushed across its pedestal instead
        of picked up.

        Waiting on the MEASURED joints is not the same as commanding from
        them. Slaving the followers to the leader's measured angle was tried
        and is worse (the left/right spread went 0.146 -> 0.297 rad); this
        only waits.

        Returns (steps waited, whether it went quiet, worst joint error).
        """
        want = self.grip_mult * target
        prev = self.robot.get_joint_positions()[self.grip_idx_all]
        quiet = 0
        for i in range(max_steps):
            if not self.running:
                return i, False, float(np.abs(prev - want).max())
            self._step()
            now = self.robot.get_joint_positions()[self.grip_idx_all]
            quiet = quiet + 1 if float(np.abs(now - prev).max()) < tol else 0
            prev = now
            if quiet >= quiet_steps:
                return i + 1, True, float(np.abs(now - want).max())
        return max_steps, False, float(np.abs(prev - want).max())

    def idle(self, n):
        """Step n times with every command held."""
        for _ in range(n):
            if not self.running:
                return
            self._step()

    def observe(self, images=False) -> Observation:
        """The world as it is now. Object poses are the simulator's, not perceived."""
        cache = UsdGeom.XformCache()
        m = cache.GetLocalToWorldTransform(self.world.stage.GetPrimAtPath(self._tool_prim))
        r = m.ExtractRotationMatrix()
        # USD matrices are row-vector: rows are the frame's axes in world.
        R = np.array([[r[j][i] for j in range(3)] for i in range(3)])
        t = m.ExtractTranslation()
        tool = np.array([t[0], t[1], t[2], *_matrix_to_quat(R)], dtype=np.float32)
        objects = {}
        for name, cube in self.payload.items():
            p, q = cube.get_world_pose()
            objects[name] = np.array([*p, *q], dtype=np.float32)
        # Holding: the gripper was last closed, and something is between the
        # fingers -- within 5 cm of the tool frame. A heuristic, not a contact
        # query: good enough to tell "carrying it" from "closed on air".
        holding = self._closed and any(
            float(np.linalg.norm(o[:3] - tool[:3])) < 0.05 for o in objects.values())
        depth = None
        if images and self.cams:
            depth = {n: grab_depth(c) for n, c in self.cams.items()}
        return Observation(
            q=np.array(self.q_now(), dtype=np.float32),
            gripper=float(self.robot.get_joint_positions()[self.grip_idx]),
            tool_pose=tool, objects=objects, holding=holding,
            sim_time=self.steps * SIM_DT, depth=depth)
