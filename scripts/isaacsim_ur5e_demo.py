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
    python scripts/isaacsim_ur5e_demo.py --robot ur5e_2f85
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
from pxr import PhysxSchema, UsdGeom, UsdLux  # noqa: E402

from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, VisualCuboid  # noqa: E402
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


def build_stage(world):
    """Import the arm from URDF, add obstacles, and hide one from the planner."""
    status, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False
    cfg.fix_base = True
    cfg.make_default_prim = False
    cfg.create_physics_scene = False
    cfg.distance_scale = 1.0
    cfg.default_drive_type = UrdfJointTargetType.JOINT_DRIVE_POSITION
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

    light = UsdLux.DistantLight.Define(world.stage, "/World/DistantLight")
    light.CreateIntensityAttr(2500)
    light.CreateAngleAttr(1.0)
    return prim_path, bodies


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
    link = f"{prim_path}/{spec['link']}"
    if not world.stage.GetPrimAtPath(link).IsValid():
        raise RuntimeError(f"{link} missing - rebuild the URDF with tools/build_ur5e_2f85_urdf.py")

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
    # camera_mount's +Y. Check the roll, not just the view direction.
    m_mount = cache.GetLocalToWorldTransform(
        world.stage.GetPrimAtPath(f"{prim_path}/camera_mount"))
    img_right = axis(m_link, 0)     # optical +X after the URDF yaw
    bar = axis(m_mount, 1)          # long edge of the D435i body
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
    prim_path, bodies = build_stage(world)
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
    coupling = {j: m for j, m in GRIPPER_COUPLING.items() if j in sim_names}
    grip_idx_all = np.array([sim_names.index(j) for j in coupling])
    grip_mult = np.array(list(coupling.values()), dtype=np.float32)

    grip_open = ROBOTS[ARGS.robot].get("gripper_open", 0.0)
    grip_closed = ROBOTS[ARGS.robot].get("gripper_closed", 0.0)
    print(f"[demo] joints: {len(sim_names)} in sim, {len(curobo_names)} planned")
    if grip_idx is None:
        print("[demo] no gripper joint to drive")
    else:
        print(f"[demo] gripper: {len(coupling)} joints driven explicitly, "
              f"{grip_open} open .. {grip_closed} closed "
              f"(no <mimic>; PhysX rejects it)")

    def command_arm(q_curobo):
        """Send one planner-ordered joint vector, touching nothing else."""
        robot.apply_action(ArticulationAction(
            joint_positions=np.asarray(q_curobo, dtype=np.float32),
            joint_indices=np.asarray(arm_idx)))

    def command_gripper(angle):
        """One commanded angle -> every joint of the linkage."""
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

    target_idx, plan_no, steps = 0, 0, 0
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
