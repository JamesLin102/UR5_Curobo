"""UR5e + 2F-85 + wrist D435i: cuRobo 0.8 planning against a live volumetric map.

This process runs Isaac Sim ONLY. It must never import cuRobo: Isaac Sim 5.1
requires Warp 1.8.2 and cuRobo 0.8 requires Warp >= 1.13, so they cannot live
in one interpreter. Planning and mapping happen in planner_server.py, reached
over a local socket.

The scene's `unmapped` bodies are never described to the planner. The arm can
only discover them through the cameras, so avoiding them is proof the map is
actually feeding the planner. Scenes live in scripts/scenes/; both processes
must be started with the same --scene.

Start planner_server.py first, then:
    python scripts/isaacsim_ur5e_demo.py --robot ur5_robotiq
"""

import argparse
import math
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenes  # noqa: E402
from proto import recv_msg, send_msg  # noqa: E402
from rig import DEFAULT_ROBOT, HOST, PORT, ROBOTS, SIM_DT  # noqa: E402

# planner_server.py flushes every line; without the same here, this side's
# output sits in the stdout buffer whenever it is redirected to a file, and the
# demo looks hung next to a server that is visibly working.
sys.stdout.reconfigure(line_buffering=True)

_ap = argparse.ArgumentParser()
_ap.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
_ap.add_argument("--scene", default=scenes.DEFAULT, choices=scenes.available(),
                 help="scene module under scripts/scenes/; must match the server")
_ap.add_argument("--no-mapping", action="store_true")
_ap.add_argument("--no-overhead", action="store_true",
                 help="wrist camera only, for A/B against the fixed camera")
_ap.add_argument("--map-every", type=int, default=6, help="fuse a frame every N sim steps")
_ap.add_argument("--move-body", "--move-cube", dest="move_body",
                 action="store_true",
                 help="drive the unmapped body along its scene motion instead "
                      "of leaving it parked")
_ap.add_argument("--static", action="store_true",
                 help="hold the arm at HOME; no planning, just look")
_ap.add_argument("--depth-lag", type=int, default=2,
                 help="sim steps the depth annotator trails the physics by")
ARGS = _ap.parse_args()
SCENE = scenes.load(ARGS.scene)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": False, "width": 1600, "height": 900})

import numpy as np  # noqa: E402
import omni.kit.commands  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.materials import PhysicsMaterial  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, VisualCuboid  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from isaacsim.util.debug_draw import _debug_draw  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = f"{ROOT}/{ROBOTS[ARGS.robot]['urdf']}"

# The gripper joints' <limit velocity="..."> in the URDF. Used to budget how
# many sim steps a full open or close actually needs; commanding it faster
# just leaves the joint short of the target when the next plan starts.
GRIPPER_RAD_PER_S = 2.0


class Planner:
    """Framed socket client for planner_server.py."""

    def __init__(self, timeout_s=600):
        deadline = time.time() + timeout_s
        while True:
            try:
                self.sock = socket.create_connection((HOST, PORT), timeout=300)
                break
            except OSError:
                if time.time() > deadline:
                    raise RuntimeError(
                        f"planner server not reachable on {HOST}:{PORT} - "
                        "start scripts/planner_server.py first"
                    )
                time.sleep(2.0)
        print(f"[demo] connected to planner on {HOST}:{PORT}")
        self._check_scene()

    def _check_scene(self):
        """Refuse to run against a planner that loaded a different scene.

        The two processes never exchange geometry, so a mismatch would show up
        only as inexplicably wrong plans. Checked here, on connect, rather than
        on the first plan: this is before the stage is built, so the failure is
        immediate and costs nothing.
        """
        send_msg(self.sock, {"op": "scene", "scene": ARGS.scene})
        header, _ = recv_msg(self.sock)
        theirs = header.get("scene")
        if theirs != ARGS.scene:
            print(f"[demo] SCENE MISMATCH: the planner is running {theirs!r}, "
                  f"this process has {ARGS.scene!r}.", flush=True)
            print("[demo] Both sides build their world from the scene, so they "
                  "must match. Restart one of them.", flush=True)
            simulation_app.close()
            sys.exit(1)

    def plan(self, q, target):
        send_msg(self.sock, {"op": "plan", "q": list(map(float, q)), "target": target})
        header, payload = recv_msg(self.sock)
        if header.get("ok"):
            header["traj"] = np.frombuffer(payload, dtype=np.float32).reshape(
                header["n"], header["dof"]
            )
        return header

    def ik(self, q, target):
        """Joint angles for one tool pose, or None. See planner_server.handle_ik."""
        send_msg(self.sock, {"op": "ik", "q": list(map(float, q)), "target": target})
        header, payload = recv_msg(self.sock)
        if not header.get("ok"):
            return None
        return np.frombuffer(payload, dtype=np.float32).tolist()

    def map_frame(self, q, depth, K, cam_name):
        send_msg(
            self.sock,
            {
                "op": "map",
                "cam": cam_name,
                "q": list(map(float, q)),
                "h": int(depth.shape[0]),
                "w": int(depth.shape[1]),
                "K": [[float(v) for v in row] for row in K],
            },
            np.ascontiguousarray(depth, dtype=np.float32).tobytes(),
        )
        # Fire-and-forget: the sim must not stall on a round-trip, so there is
        # no reply to return. The server's own frame count is only visible in
        # its log.


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


_URDF_TREE = None


def _urdf_tree():
    """child link -> (parent link, 4x4 transform), for the whole URDF."""
    global _URDF_TREE
    if _URDF_TREE is None:
        import xml.etree.ElementTree as ET

        def rpy(r, p, y):
            cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                                      math.sin(p), math.cos(y), math.sin(y))
            return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                             [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                             [-sp, cp * sr, cp * cr]])

        _URDF_TREE = {}
        for j in ET.parse(URDF).getroot().findall("joint"):
            o = j.find("origin")
            T = np.eye(4)
            if o is not None:
                T[:3, 3] = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
                T[:3, :3] = rpy(*[float(v) for v in (o.get("rpy") or "0 0 0").split()])
            _URDF_TREE[j.find("child").get("link")] = (j.find("parent").get("link"), T)
    return _URDF_TREE


def frame_prim(stage, prim_path, link):
    """Prim path for `link`, recreating it if merge_fixed_joints ate it.

    Merging keeps a prim for every link that carries geometry and discards the
    pure frames. Walk up the URDF to the nearest link that does have a prim,
    carry the transform along, and hang an Xform there. An Xform under a rigid
    body is just a frame -- it adds nothing for PhysX to solve, which is the
    whole point of having merged in the first place.

    Returns (path of `link`, path of the body it was merged into).
    """
    if stage.GetPrimAtPath(f"{prim_path}/{link}").IsValid():
        return f"{prim_path}/{link}", f"{prim_path}/{link}"
    tree = _urdf_tree()
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
    raise RuntimeError(f"{link} is not in {URDF}, or has no ancestor with a prim")


def close_gripper_linkage(stage, prim_path):
    """Pin each inner knuckle to its finger tip, closing the 4-bar in PhysX.

    Without this the knuckle is driven open-loop and fights whatever it
    touches. With it, the knuckle's angle comes from the mechanism, which is
    where it comes from on the real gripper.
    """
    anchors = _gripper_pin_anchors(URDF)
    made = []
    for side, (on_knuckle, on_tip) in anchors.items():
        k = stage.GetPrimAtPath(f"{prim_path}/robotiq_85_{side}_inner_knuckle_link")
        f = stage.GetPrimAtPath(f"{prim_path}/robotiq_85_{side}_finger_tip_link")
        if not (k.IsValid() and f.IsValid()):
            print(f"[demo] cannot pin {side} linkage: link prim missing")
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
        print(f"[demo] 4-bar closed: pinned {', '.join(made)} inner knuckle "
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
    print(f"[demo] arm drives: {n} at stiffness {k:g}, damping {d:g}")


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
    print(f"[demo] gripper drives: {len(softened)} at "
          f"{GRIPPER_DRIVE_STIFFNESS:g}/{GRIPPER_DRIVE_DAMPING:g}, "
          f"{len(freed)} released to the 4-bar")
    return freed


def build_stage(world):
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
        "URDFParseAndImportFile", urdf_path=URDF, import_config=cfg
    )
    print(f"[demo] {ARGS.robot} imported at {prim_path}")

    world.scene.add_default_ground_plane()
    for name, dims, pose, colour in SCENE.obstacles:
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
    # both halves: visible to you, invisible to the cameras. `baseline` still
    # reporting "0 tall []" is what proves the second half.
    draw_targets(SCENE.targets)

    # The bodies the planner is never told about. VisualCuboids, not physics
    # bodies, so dragging one in the viewport does not fight PhysX -- it still
    # renders into depth, which is all the cameras need. That does mean the arm
    # passes through rather than hitting it.
    #
    # These deliberately keep their default render purpose, unlike the target
    # markers above: being seen is the entire point of them.
    bodies = {}
    for name, dims, pose, colour in SCENE.unmapped:
        bodies[name] = VisualCuboid(
            prim_path=f"/World/{name}",
            name=name,
            position=np.array(pose[:3]),
            scale=np.array(dims),
            color=np.array(colour),
        )

    close_gripper_linkage(world.stage, prim_path)
    tune_arm_drives(world.stage, set(PLANNED_JOINTS))
    tune_gripper_drives(world.stage, prim_path,
                        set(ROBOTS[ARGS.robot].get("gripper_joints") or {}))

    # Payloads. Real rigid bodies, unlike everything above: they fall, they can
    # be squeezed, and they come away when the gripper closes. High friction on
    # both the block and the finger pads is what makes a position-driven
    # gripper hold rather than extrude what it is squeezing.
    payload = {}
    if SCENE.payload:
        grip_mat = PhysicsMaterial(prim_path="/World/physics/grip",
                                   static_friction=1.2, dynamic_friction=1.1,
                                   restitution=0.0)
        for name, dims, pose, colour, mass in SCENE.payload:
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


_DRAW = _debug_draw.acquire_debug_draw_interface()


def draw_targets(targets):
    """Mark each goal in the viewport without putting anything in the scene.

    Points plus a small axis cross, so a goal reads as a location rather than a
    stray dot. Redrawn from scratch each time, because the overlay accumulates.
    """
    _DRAW.clear_points()
    _DRAW.clear_lines()
    blue = (0.05, 0.43, 0.62, 1.0)
    _DRAW.draw_points([tuple(t[:3]) for t in targets], [blue] * len(targets),
                      [14.0] * len(targets))
    arm = 0.03
    starts, ends = [], []
    for t in targets:
        x, y, z = t[:3]
        for dx, dy, dz in ((arm, 0, 0), (0, arm, 0), (0, 0, arm)):
            starts.append((x - dx, y - dy, z - dz))
            ends.append((x + dx, y + dy, z + dz))
    _DRAW.draw_lines(starts, ends, [blue] * len(starts), [2.0] * len(starts))


def attach_wrist_camera(world, prim_path):
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
    spec = SCENE.cameras["wrist"]
    link, body = frame_prim(world.stage, prim_path, spec["link"])

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
    print(f"[demo]   view in camera_link axes: {local.round(3)}  "
          f"(want [0 0 1])")
    # Image-horizontal should run along the camera body's long edge, which is
    # +Y of the D435i's own link. Check the roll, not just the view direction.
    m_body = cache.GetLocalToWorldTransform(world.stage.GetPrimAtPath(body))
    img_right = axis(m_link, 0)     # optical +X after the URDF yaw
    bar = axis(m_body, 1)           # long edge of the D435i body
    # Signed, not |dot|: +1 and -1 both mean "aligned with the long edge" but
    # they differ by a 180 degree roll, i.e. an upside-down image. An earlier
    # version compared magnitudes and happily passed the flipped one.
    print(f"[demo]   image-right vs body long edge: dot = "
          f"{float(np.dot(img_right, bar)):+.3f}  (want -1.000)")
    print(f"[demo] wrist camera on {spec['link']}: "
          f"{spec['width']}x{spec['height']}, {spec['horizontal_fov_deg']:.0f} deg HFOV")
    print(f"[demo] view-vs-optical dot = {dot:+.3f}  "
          f"({'AGREE' if dot > 0.9 else 'REVERSED' if dot < -0.9 else 'PERPENDICULAR'})")
    print(f"[demo]   usd camera view dir (world): {view.round(3)}")
    print(f"[demo]   camera_link +Z    (world): {optical.round(3)}")
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


def attach_overhead_camera(world):
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
    spec = SCENE.cameras["overhead"]
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
    print(f"[demo] overhead camera at {np.array(pose[:3]).round(3)}: "
          f"{spec['width']}x{spec['height']}, {spec['horizontal_fov_deg']:.0f} deg HFOV")
    print(f"[demo]   view dir (world): {view.round(3)}  want {want.round(3)}  "
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


def move_bodies(bodies, t_s):
    """Slide the scene's unmapped body along y, inside the cameras' scan band.

    The planner is never told about this. The only way the arm can know where
    the body is, is the cameras -- so a change in the planned route when it
    arrives is the whole proof.

    Returns the body's current y, or None if the scene has no unmapped body.
    """
    if not bodies:
        return None
    name, prim = next(iter(bodies.items()))
    home_pose = SCENE.body(name)[2]
    sweep = SCENE.motions.get(name)
    if not ARGS.move_body or not sweep:
        return home_pose[1]             # parked: the A/B test wants it still
    a, b = sweep["y_from"], sweep["y_to"]
    phase = (t_s % sweep["period_s"]) / sweep["period_s"]
    # triangle wave: out and back, with a pause at each end
    u = min(1.0, max(0.0, abs(1.0 - 2.0 * phase) * 1.4 - 0.2))
    y = a + (b - a) * u
    prim.set_world_pose(position=np.array([sweep["x"], y, home_pose[2]]))
    return y


def main():
    print(f"[demo] robot: {ARGS.robot}  urdf: {os.path.basename(URDF)}")
    planner = Planner()

    world = World(physics_dt=SIM_DT, rendering_dt=SIM_DT, stage_units_in_meters=1.0)
    prim_path, bodies, payload = build_stage(world)
    set_camera_view(eye=[2.0, 1.6, 1.4], target=[0.35, 0.0, 0.35])

    robot = SingleArticulation(prim_path=prim_path, name="ur5e")
    world.scene.add(robot)
    cams = {}
    if not ARGS.no_mapping:
        cams["wrist"] = attach_wrist_camera(world, prim_path)
        if not ARGS.no_overhead:
            cams["overhead"] = attach_overhead_camera(world)
    world.reset()
    robot.initialize()

    px = PhysxSchema.PhysxArticulationAPI.Get(world.stage, prim_path)
    if px:
        px.CreateSolverPositionIterationCountAttr(64)
        px.CreateSolverVelocityIterationCountAttr(16)

    probe = planner.plan(SCENE.home, SCENE.targets[0])
    curobo_names = probe["joint_names"]
    sim_names = list(robot.dof_names)

    # The simulator has more joints than the planner: the gripper is
    # articulated in the URDF but locked out of cuRobo's cspace, so it is
    # driven here on its own channel. Address the two sets by INDEX rather
    # than reindexing whole arrays -- an arm command must never disturb the
    # fingers, and a joint the planner has never heard of must not be looked
    # up in its name list.
    missing = [j for j in curobo_names if j not in sim_names]
    if missing:
        raise RuntimeError(f"planner joints absent from the simulator: {missing}")
    arm_idx = [sim_names.index(j) for j in curobo_names]

    GRIPPER_COUPLING = ROBOTS[ARGS.robot].get("gripper_joints") or {}
    grip_name = next(iter(GRIPPER_COUPLING), None)
    grip_idx = sim_names.index(grip_name) if grip_name in sim_names else None
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
    coupling = {j: m for j, m in GRIPPER_COUPLING.items()
                if j in sim_names and PINNED_FOLLOWER not in j}
    grip_idx_all = np.array([sim_names.index(j) for j in coupling])
    # Every gripper joint, followers included, for reporting. Signed by its
    # multiplier so all six read as "fraction closed" and can be compared.
    all_grip_names = [j for j in GRIPPER_COUPLING if j in sim_names]
    all_grip_idx = np.array([sim_names.index(j) for j in all_grip_names])
    all_grip_sign = np.array([GRIPPER_COUPLING[j] for j in all_grip_names],
                             dtype=np.float32)
    grip_mult = np.array(list(coupling.values()), dtype=np.float32)

    grip_open = ROBOTS[ARGS.robot].get("gripper_open", 0.0)
    grip_closed = ROBOTS[ARGS.robot].get("gripper_closed", 0.0)
    print(f"[demo] joints: {len(sim_names)} in sim, {len(curobo_names)} planned")
    if grip_idx is None:
        print("[demo] no gripper joint to drive")
    else:
        print(f"[demo] gripper: {len(coupling)} load-bearing joints driven, "
              f"{grip_open} open .. {grip_closed} closed; "
              f"inner knuckles set by the pinned 4-bar")

    def command_arm(q_curobo):
        """Send one planner-ordered joint vector, touching nothing else."""
        robot.apply_action(ArticulationAction(
            joint_positions=np.asarray(q_curobo, dtype=np.float32),
            joint_indices=np.asarray(arm_idx)))

    def command_gripper(angle):
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
        if not len(grip_idx_all):
            return
        robot.apply_action(ArticulationAction(
            joint_positions=(grip_mult * angle).astype(np.float32),
            joint_indices=grip_idx_all))

    full_home = np.array(robot.get_joint_positions(), dtype=np.float32)
    for k, i in enumerate(arm_idx):
        full_home[i] = SCENE.home[k]
    if grip_idx is not None:
        full_home[grip_idx] = grip_open
    robot.set_joint_positions(full_home)
    command_arm(SCENE.home)
    command_gripper(grip_open)
    for _ in range(60):
        world.step(render=True)

    intrinsics = {}
    if cams:
        for _ in range(10):  # let the annotators produce their first frame
            world.step(render=True)
        for name, c in cams.items():
            k = c.get_intrinsics_matrix()
            intrinsics[name] = k
            print(f"[demo] {name} intrinsics from Isaac Sim: fx={k[0,0]:.1f} "
                  f"fy={k[1,1]:.1f} cx={k[0,2]:.1f} cy={k[1,2]:.1f}")

    def q_now():
        q_sim = robot.get_joint_positions()
        return [float(q_sim[i]) for i in arm_idx]

    q_history = []

    def fuse(require_still=False):
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
        if not cams:
            return 0
        if require_still:
            v = robot.get_joint_velocities()
            if v is not None and float(np.abs(np.asarray(v)).max()) > 0.05:
                return 0
        q_lagged = q_history[-1 - ARGS.depth_lag] if len(q_history) > ARGS.depth_lag \
            else q_now()
        sent = 0
        for name, c in cams.items():
            depth = grab_depth(c)
            if depth is None:
                continue
            planner.map_frame(q_lagged, depth, intrinsics[name], name)
            sent += 1
        return sent

    # --- scan sweep: build a map before trusting it to plan ---------------
    if cams:
        print("[demo] scanning the cell before planning...")
        fused = 0
        for pose in SCENE.scan_poses:
            for step in range(70):
                if not simulation_app.is_running():
                    break
                blend = min(1.0, (step + 1) / 50.0)
                cmd = [(1 - blend) * a + blend * b for a, b in zip(q_now(), pose)]
                command_arm(cmd)
                world.step(render=True)
                q_history.append(q_now())
                if step % ARGS.map_every == 0:
                    fused += fuse()
        fused += fuse()
        print(f"[demo] scan done: {fused} frames sent to the mapper")

    print("[demo] -------------------------------------------------------------")
    for name in bodies:
        print(f"[demo]  Drag /World/{name} into the arm's path in the viewport.")
    print("[demo]  The planner is never told where it is.")
    if cams:
        print(f"[demo]  The cameras ({', '.join(cams)}) have to find it, and the")
        print("[demo]  arm should route around it.")
    else:
        print("[demo]  Mapping is off, so nothing can find it: the arm will")
        print("[demo]  drive straight through. This is the A/B baseline.")
    print("[demo] -------------------------------------------------------------")

    if ARGS.static:
        print("[demo] static mode: arm held at HOME, no planning. "
              "Inspect the d435i view, then Ctrl-C or close the window.")
        while simulation_app.is_running():
            command_arm(SCENE.home)
            world.step(render=True)
        simulation_app.close()
        return

    def move_tool_z(pose, dz, n_steps):
        """Put the tool at `pose` shifted by dz in z, by IK then interpolation.

        plan_pose is no use here: the block is in the map, and the planner will
        not route a tool into an obstacle. Asking for IK at the end pose keeps
        both ENDPOINTS exact and only the path between them unchecked -- which
        over 12 cm of vertical move is acceptable, and over a long one would
        not be.

        Returns False if IK found nothing, rather than moving somewhere wrong.
        """
        goal = list(pose)
        goal[2] += dz
        q_goal = planner.ik(q_now(), goal)
        if q_goal is None:
            print(f"[demo]   no IK for z{dz:+.3f} m; skipping")
            return False
        start = q_now()
        # A 12 cm vertical move is a small joint move. If it is not, the IK
        # came back on a different branch from the one the arm is standing in,
        # and blending straight to it sweeps the arm through that difference
        # instead of going down.
        step = [abs(b - a) for a, b in zip(start, q_goal)]
        print(f"[demo]   z{dz:+.3f}: joint travel "
              f"{' '.join(f'{v:.3f}' for v in step)} rad"
              f"   worst {max(step):.3f} over {n_steps} steps "
              f"({max(step) / (n_steps * SIM_DT):.2f} rad/s)")
        for i in range(n_steps):
            if not simulation_app.is_running():
                return False
            blend = (i + 1) / n_steps
            command_arm([a + (b - a) * blend for a, b in zip(start, q_goal)])
            world.step(render=True)
        return True

    def settle_gripper(target, max_steps=240, quiet_steps=12, tol=2.0e-4):
        """Step until the driven gripper joints stop moving.

        The stroke above is commanded open-loop over a fixed number of steps,
        which is right in free air and wrong on an object: the fingers stall
        against it and the joints lag their command -- measured at 0.365 rad
        while squeezing. Lifting on the ramp's last step therefore lifts
        before the grip has closed, and the block gets pushed across its
        pedestal instead of picked up.

        Waiting on the MEASURED joints is not the same as commanding from
        them. Slaving the followers to the leader's measured angle was tried
        and is worse (the left/right spread went 0.146 -> 0.297 rad); this
        only waits.

        Returns (steps waited, whether it went quiet, worst joint error).
        """
        if not len(grip_idx_all):
            return 0, True, 0.0
        want = grip_mult * target
        prev = robot.get_joint_positions()[grip_idx_all]
        quiet = 0
        for i in range(max_steps):
            if not simulation_app.is_running():
                return i, False, float(np.abs(prev - want).max())
            world.step(render=True)
            now = robot.get_joint_positions()[grip_idx_all]
            quiet = quiet + 1 if float(np.abs(now - prev).max()) < tol else 0
            prev = now
            if quiet >= quiet_steps:
                return i + 1, True, float(np.abs(now - want).max())
        return max_steps, False, float(np.abs(prev - want).max())

    def hold_still(n):
        for _ in range(n):
            if not simulation_app.is_running():
                return
            world.step(render=True)

    def pick_place_cycle(plan_no, src, dst):
        """Lift the block off `src` and set it down on `dst`.

        Called with src and dst swapped each time, so the block shuttles back
        and forth rather than being picked once and abandoned.
        """
        for idx, grab in ((src, True), (dst, False)):
            what = f"pick@{idx}" if grab else f"place@{idx}"
            res = planner.plan(q_now(), SCENE.targets[idx])
            if not res.get("ok"):
                print(f"[demo] plan #{plan_no} ({what}) blocked - waiting")
                hold_still(30)
                return False
            print(f"[demo] plan #{plan_no} -> {what}: solve "
                  f"{res['solve_ms']:.0f} ms, {len(res['traj'])} waypoints")
            for k, wp in enumerate(res["traj"]):
                if not simulation_app.is_running():
                    return False
                command_arm(wp)
                world.step(render=True)
                q_history.append(q_now())
                # Keep the map fed during the long legs, the same way the
                # avoidance loop does. Without this the slab decays out of the
                # map between grasps and the next plan drives through it.
                if cams and k % ARGS.map_every == 0:
                    fuse()
            hold_still(20)

            if not move_tool_z(SCENE.targets[idx], -SCENE.pick["descend_m"], 70):
                return False
            hold_still(15)
            stroke = int(abs(grip_closed - grip_open) / GRIPPER_RAD_PER_S / SIM_DT) + 6
            for i in range(stroke):
                command_gripper(grip_open + (grip_closed - grip_open) *
                                ((i + 1) / stroke if grab else 1 - (i + 1) / stroke))
                world.step(render=True)
            # Do not move until the fingers have actually stopped.
            end = grip_closed if grab else grip_open
            waited, quiet, err = settle_gripper(end)
            # Every joint, not just the worst: a linkage that has come apart
            # and a linkage that has simply stalled on the object look
            # identical in a single number, and they need opposite fixes.
            q_all = robot.get_joint_positions()[all_grip_idx]
            spread = " ".join(f"{n.split('robotiq_85_')[-1][:-6]}={v * m:+.3f}"
                              for n, v, m in zip(all_grip_names, q_all, all_grip_sign))
            print(f"[demo]   {'close' if grab else 'open'}: settled after "
                  f"{waited} steps{'' if quiet else ' (TIMED OUT, still moving)'}"
                  f", want {end:+.3f} rad")
            print(f"[demo]     {spread}")
            lifted = list(SCENE.targets[idx])
            lifted[2] += SCENE.pick["lift_m"] - SCENE.pick["descend_m"]
            move_tool_z(lifted, 0.0, 70)
            hold_still(20)

            for name, cube in payload.items():
                pos = cube.get_world_pose()[0]
                print(f"[demo]   after {what}: {name} at "
                      f"[{pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}]")
        return True

    target_idx, plan_no, steps = 0, 0, 0
    if payload:
        print(f"[demo] pick-and-place: {list(payload)} shuttling between "
              f"{len(SCENE.targets)} pedestals, descend "
              f"{SCENE.pick['descend_m']:.3f} m / lift {SCENE.pick['lift_m']:.3f} m "
              f"by IK, the rest planned")
        src, dst = 0, 1
        while simulation_app.is_running():
            plan_no += 1
            if pick_place_cycle(plan_no, src, dst):
                src, dst = dst, src      # next time, bring it back

        simulation_app.close()
        return

    while simulation_app.is_running():
        result = planner.plan(q_now(), SCENE.targets[target_idx])
        plan_no += 1
        if not result.get("ok"):
            # Keep the body moving while we retry. Without this the sim freezes
            # the one thing that could clear the route, and a blocked plan stays
            # blocked forever.
            y = None
            for _ in range(30):
                if not simulation_app.is_running():
                    break
                steps += 1
                y = move_bodies(bodies, steps * SIM_DT)
                world.step(render=True)
                if cams and steps % ARGS.map_every == 0:
                    fuse()
            where = f" | body y={y:+.2f}" if y is not None else ""
            print(f"[demo] plan #{plan_no} blocked - waiting{where}")
            continue

        traj = result["traj"]
        y = move_bodies(bodies, steps * SIM_DT)
        where = f" | body y={y:+.2f}" if y is not None else ""
        print(f"[demo] plan #{plan_no} -> target {target_idx}: "
              f"solve {result['solve_ms']:.0f} ms, {len(traj)} waypoints{where}")

        for wp_row in traj:
            if not simulation_app.is_running():
                break
            command_arm(wp_row)
            move_bodies(bodies, steps * SIM_DT)
            world.step(render=True)
            q_history.append(q_now())
            steps += 1
            if cams and steps % ARGS.map_every == 0:
                fuse()

        # Close on arrival and open again before leaving. The planner is not
        # told about this: the config locks the gripper OPEN, which is its
        # widest, so a route that cleared with it open stays clear while it
        # closes. Closing mid-travel would not be safe on that argument.
        # The gripper's URDF velocity limit is 2.0 rad/s, so a full 0.8 rad
        # stroke needs 0.4 s -- 24 steps at 60 Hz. Ten-step ramps left it
        # stranded at +0.334 rad when the next plan started. Budget the travel
        # from the limit rather than guessing: close, hold, open, settle.
        stroke_steps = int(abs(grip_closed - grip_open) / GRIPPER_RAD_PER_S / SIM_DT) + 4
        hold_steps = 10
        settle = 90 if grip_idx is not None else 45
        for i in range(settle):
            if not simulation_app.is_running():
                break
            steps += 1
            if grip_idx is not None:
                if i < stroke_steps:
                    phase = i / stroke_steps
                elif i < stroke_steps + hold_steps:
                    phase = 1.0
                else:
                    phase = max(0.0, 1.0 - (i - stroke_steps - hold_steps) / stroke_steps)
                command_gripper(grip_open + (grip_closed - grip_open) * phase)
            move_bodies(bodies, steps * SIM_DT)
            world.step(render=True)
            if cams and i >= 15 and i % ARGS.map_every == 0:
                fuse()

        err = np.rad2deg(np.abs(np.asarray(q_now()) - traj[-1]))
        grip = "" if grip_idx is None else \
            f", gripper back to {robot.get_joint_positions()[grip_idx]:+.3f} rad"
        print(f"[demo]   tracking error: max {err.max():.2f} deg{grip}")
        target_idx = 1 - target_idx

    simulation_app.close()


if __name__ == "__main__":
    main()
