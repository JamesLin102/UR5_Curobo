"""UR5e + 2F-85 + wrist D435i: cuRobo 0.8 planning against a live volumetric map.

This process runs Isaac Sim ONLY. It must never import cuRobo: Isaac Sim 5.1
requires Warp 1.8.2 and cuRobo 0.8 requires Warp >= 1.13, so they cannot live
in one interpreter. Planning and mapping happen in planner_server.py, reached
over a local socket.

The scene contains one obstacle the planner is never told about
(scene_def.DRAG_CUBE). The arm can only discover it through the wrist camera,
so avoiding it is proof the map is actually feeding the planner.

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
from proto import recv_msg, send_msg  # noqa: E402
from scene_def import (  # noqa: E402
    CAMERA, CUBE_SWEEP, DEFAULT_ROBOT, DRAG_CUBE, HOME, HOST, OBSTACLES, PORT,
    ROBOTS, SCAN_POSES, SIM_DT, TARGETS,
)

_ap = argparse.ArgumentParser()
_ap.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
_ap.add_argument("--no-mapping", action="store_true")
_ap.add_argument("--map-every", type=int, default=6, help="fuse a frame every N sim steps")
_ap.add_argument("--move-cube", action="store_true",
                 help="sweep the cube along y instead of leaving it parked")
_ap.add_argument("--static", action="store_true",
                 help="hold the arm at HOME; no planning, just look")
_ap.add_argument("--depth-lag", type=int, default=2,
                 help="sim steps the depth annotator trails the physics by")
ARGS = _ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": False, "width": 1600, "height": 900})

import numpy as np  # noqa: E402
import omni.kit.commands  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux  # noqa: E402

from isaacsim.asset.importer.urdf._urdf import UrdfJointTargetType  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, VisualCuboid  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = f"{ROOT}/{ROBOTS[ARGS.robot]['urdf']}"


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

    def plan(self, q, target):
        send_msg(self.sock, {"op": "plan", "q": list(map(float, q)), "target": target})
        header, payload = recv_msg(self.sock)
        if header.get("ok"):
            header["traj"] = np.frombuffer(payload, dtype=np.float32).reshape(
                header["n"], header["dof"]
            )
        return header

    def map_frame(self, q, depth, K):
        send_msg(
            self.sock,
            {
                "op": "map",
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
    for name, dims, pose, colour in OBSTACLES:
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/obstacles/{name}",
                name=name,
                position=np.array(pose[:3]),
                scale=np.array(dims),
                color=np.array(colour),
            )
        )
    for i, t in enumerate(TARGETS):
        VisualCuboid(
            prim_path=f"/World/targets/target_{i}",
            name=f"target_{i}",
            position=np.array(t[:3]),
            scale=np.array([0.04, 0.04, 0.04]),
            color=np.array([0.05, 0.43, 0.62]),
        )

    # The draggable cube. A VisualCuboid, not a physics body, so dragging it in
    # the viewport does not fight PhysX -- it still renders into depth, which is
    # all the camera needs.
    name, dims, pose, colour = DRAG_CUBE
    cube = VisualCuboid(
        prim_path=f"/World/{name}",
        name=name,
        position=np.array(pose[:3]),
        scale=np.array(dims),
        color=np.array(colour),
    )

    light = UsdLux.DistantLight.Define(world.stage, "/World/DistantLight")
    light.CreateIntensityAttr(2500)
    light.CreateAngleAttr(1.0)
    return prim_path, cube


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
    link = f"{prim_path}/{CAMERA['link']}"
    if not world.stage.GetPrimAtPath(link).IsValid():
        raise RuntimeError(f"{link} missing - rebuild the URDF with tools/build_ur5e_2f85_urdf.py")

    cam = Camera(
        prim_path=f"{link}/d435i",
        resolution=(CAMERA["width"], CAMERA["height"]),
        translation=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.5, 0.5, -0.5, 0.5]),  # optical -> ROS body, wxyz
    )
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()

    aperture = 20.955  # USD default horizontal aperture, mm
    focal = aperture / (2.0 * math.tan(math.radians(CAMERA["horizontal_fov_deg"]) / 2.0))
    cam.set_focal_length(focal / 10.0)  # Camera API works in cm
    cam.set_clipping_range(CAMERA["near"], CAMERA["far"])
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
    print(f"[demo] wrist camera on {CAMERA['link']}: "
          f"{CAMERA['width']}x{CAMERA['height']}, {CAMERA['horizontal_fov_deg']:.0f} deg HFOV")
    print(f"[demo] view-vs-optical dot = {dot:+.3f}  "
          f"({'AGREE' if dot > 0.9 else 'REVERSED' if dot < -0.9 else 'PERPENDICULAR'})")
    print(f"[demo]   usd camera view dir (world): {view.round(3)}")
    print(f"[demo]   camera_link +Z    (world): {optical.round(3)}")
    return cam


def grab_depth(cam):
    """Perpendicular depth in metres, or None until the annotator has data."""
    frame = cam.get_current_frame()
    depth = frame.get("distance_to_image_plane")
    if depth is None:
        return None
    depth = np.asarray(depth, dtype=np.float32)
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def move_cube(cube, t_s):
    """Slide the cube back and forth along y, inside the camera's scan band.

    The planner is never told about this. The only way the arm can know where
    the cube is, is the wrist camera -- so a change in the planned route when
    the cube arrives is the whole proof.
    """
    if not ARGS.move_cube:
        return DRAG_CUBE[2][1]          # parked: the A/B test wants it still
    a, b = CUBE_SWEEP["y_from"], CUBE_SWEEP["y_to"]
    phase = (t_s % CUBE_SWEEP["period_s"]) / CUBE_SWEEP["period_s"]
    # triangle wave: out and back, with a pause at each end
    u = min(1.0, max(0.0, abs(1.0 - 2.0 * phase) * 1.4 - 0.2))
    y = a + (b - a) * u
    pos = np.array([CUBE_SWEEP["x"], y, DRAG_CUBE[2][2]])
    cube.set_world_pose(position=pos)
    return y


def main():
    print(f"[demo] robot: {ARGS.robot}  urdf: {os.path.basename(URDF)}")
    planner = Planner()

    world = World(physics_dt=SIM_DT, rendering_dt=SIM_DT, stage_units_in_meters=1.0)
    prim_path, cube = build_stage(world)
    set_camera_view(eye=[2.0, 1.6, 1.4], target=[0.35, 0.0, 0.35])

    robot = SingleArticulation(prim_path=prim_path, name="ur5e")
    world.scene.add(robot)
    cam = None if ARGS.no_mapping else attach_wrist_camera(world, prim_path)
    world.reset()
    robot.initialize()

    px = PhysxSchema.PhysxArticulationAPI.Get(world.stage, prim_path)
    if px:
        px.CreateSolverPositionIterationCountAttr(64)
        px.CreateSolverVelocityIterationCountAttr(16)

    probe = planner.plan(HOME, TARGETS[0])
    curobo_names = probe["joint_names"]
    sim_names = list(robot.dof_names)
    curobo_to_sim = [curobo_names.index(j) for j in sim_names]
    sim_to_curobo = [sim_names.index(j) for j in curobo_names]
    print(f"[demo] joints: sim {sim_names}")

    home_sim = np.array(HOME)[curobo_to_sim]
    robot.set_joint_positions(home_sim)
    robot.apply_action(ArticulationAction(joint_positions=home_sim))
    for _ in range(60):
        world.step(render=True)

    K = None
    if cam is not None:
        for _ in range(10):  # let the annotator produce its first frame
            world.step(render=True)
        K = cam.get_intrinsics_matrix()
        print(f"[demo] intrinsics from Isaac Sim: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
              f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    def q_now():
        q_sim = robot.get_joint_positions()
        return [float(q_sim[sim_names.index(j)]) for j in curobo_names]

    q_history = []

    def fuse(require_still=False):
        """Fuse one frame, pairing the depth with the pose it was rendered at.

        Isaac Sim's depth annotator trails the physics by a step or two, so
        pairing a frame with the CURRENT joint state misaligns the self-mask
        and the arm smears its own image into the map as phantom obstacles.
        Gating on "arm is stationary" avoids that but starves the map (measured:
        420 voxels instead of 6500). Compensating the lag instead keeps the
        coverage and the alignment.

        Returns True if a frame was sent. The send is fire-and-forget, so this
        says nothing about what the mapper made of it.
        """
        if cam is None:
            return False
        if require_still:
            v = robot.get_joint_velocities()
            if v is not None and float(np.abs(np.asarray(v)).max()) > 0.05:
                return False
        depth = grab_depth(cam)
        if depth is None:
            return False
        q_lagged = q_history[-1 - ARGS.depth_lag] if len(q_history) > ARGS.depth_lag \
            else q_now()
        planner.map_frame(q_lagged, depth, K)
        return True

    # --- scan sweep: build a map before trusting it to plan ---------------
    if cam is not None:
        print("[demo] scanning the cell before planning...")
        fused = 0
        for pose in SCAN_POSES:
            goal_sim = np.array(pose)[curobo_to_sim]
            for step in range(70):
                if not simulation_app.is_running():
                    break
                blend = min(1.0, (step + 1) / 50.0)
                cmd = (1 - blend) * np.array(robot.get_joint_positions()) + blend * goal_sim
                robot.apply_action(ArticulationAction(joint_positions=cmd))
                world.step(render=True)
                q_history.append(q_now())
                if step % ARGS.map_every == 0:
                    fused += fuse()
        fused += fuse()
        print(f"[demo] scan done: {fused} frames sent to the mapper")

    print("[demo] -------------------------------------------------------------")
    print(f"[demo]  Drag /World/{DRAG_CUBE[0]} into the arm's path in the viewport.")
    print("[demo]  The planner is never told where it is - the wrist camera")
    print("[demo]  has to find it, and the arm should route around it.")
    print("[demo] -------------------------------------------------------------")

    if ARGS.static:
        print("[demo] static mode: arm held at HOME, no planning. "
              "Inspect the d435i view, then Ctrl-C or close the window.")
        while simulation_app.is_running():
            robot.apply_action(ArticulationAction(joint_positions=home_sim))
            world.step(render=True)
        simulation_app.close()
        return

    target_idx, plan_no, steps = 0, 0, 0
    while simulation_app.is_running():
        result = planner.plan(q_now(), TARGETS[target_idx])
        plan_no += 1
        if not result.get("ok"):
            # Keep the cube moving while we retry. Without this the sim freezes
            # the one thing that could clear the route, and a blocked plan stays
            # blocked forever.
            y = None
            for _ in range(30):
                if not simulation_app.is_running():
                    break
                steps += 1
                y = move_cube(cube, steps * SIM_DT)
                world.step(render=True)
                if cam is not None and steps % ARGS.map_every == 0:
                    fuse()
            print(f"[demo] plan #{plan_no} blocked - waiting "
                  f"| cube y={y:+.2f}" if y is not None else
                  f"[demo] plan #{plan_no} blocked - waiting")
            continue

        traj = result["traj"]
        print(f"[demo] plan #{plan_no} -> target {target_idx}: "
              f"solve {result['solve_ms']:.0f} ms, {len(traj)} waypoints "
              f"| cube y={move_cube(cube, steps * SIM_DT):+.2f}")

        for wp_row in traj:
            if not simulation_app.is_running():
                break
            robot.apply_action(ArticulationAction(joint_positions=wp_row[curobo_to_sim]))
            move_cube(cube, steps * SIM_DT)
            world.step(render=True)
            q_history.append(q_now())
            steps += 1
            if cam is not None and steps % ARGS.map_every == 0:
                fuse()

        for i in range(45):
            if not simulation_app.is_running():
                break
            steps += 1
            move_cube(cube, steps * SIM_DT)
            world.step(render=True)
            if cam is not None and i >= 15 and i % ARGS.map_every == 0:
                fuse()

        want = traj[-1][curobo_to_sim]
        err = np.rad2deg(np.abs(np.asarray(robot.get_joint_positions()) - want))
        print(f"[demo]   tracking error: max {err.max():.2f} deg")
        target_idx = 1 - target_idx

    simulation_app.close()


if __name__ == "__main__":
    main()
