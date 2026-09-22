"""cuRobo 0.8 planning + volumetric mapping service.

Runs in its own process because cuRobo 0.8 needs Warp >= 1.13 while Isaac Sim
5.1 ships and requires Warp 1.8.2 -- the two cannot share one interpreter.
Isaac Sim talks to this over a local socket.

Two operations, framed by scripts/proto.py:

    {"op": "map",  "q": [6 floats], "h": H, "w": W, "K": 3x3} + float32 depth
        -> fuse one wrist-camera frame into the TSDF. The camera pose is NOT
           sent: this process computes it from q by forward kinematics, since
           camera_link is a frame in the robot's own URDF. One less thing to
           keep in sync.

    {"op": "plan", "q": [6 floats], "target": [x,y,z,qw,qx,qy,qz]}
        -> plan against the static scene PLUS whatever the map has learned.

Run:
    python scripts/planner_server.py --robot ur5e_2f85
"""

import argparse
import copy
import os
import socket
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenes  # noqa: E402
from proto import recv_msg, send_msg  # noqa: E402
from rig import DEFAULT_ROBOT, HOST, PORT, ROBOTS, SIM_DT  # noqa: E402

from curobo.types import CameraObservation, ContentPath, GoalToolPose, JointState, Pose  # noqa: E402
from curobo.kinematics import Kinematics, KinematicsCfg  # noqa: E402
from curobo.scene import Cuboid, Scene  # noqa: E402
from curobo.perception import FilterDepth, Mapper, MapperCfg, RobotSegmenter  # noqa: E402
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: E402
from curobo._src.robot.loader.util import load_robot_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def static_scene(scene):
    """The world the planner IS told about: exact cuboids, nothing perceived.

    scene.unmapped is deliberately absent -- those bodies exist only in the
    simulator, and the whole point is that they reach the planner through the
    cameras or not at all.
    """
    return Scene(cuboid=[Cuboid(name=n, dims=d, pose=p)
                         for n, d, p, _ in scene.obstacles])


class Mapping:
    """Wrist-camera RGB-D -> TSDF -> ESDF, fed back into the planner's scene."""

    def __init__(self, robot_dict, kin, scene):
        self.kin = kin
        self.scene = scene
        cameras, mapper_cfg = scene.cameras, scene.mapper
        # Buffers are sized for the largest camera; each frame is integrated on
        # its own, so num_cameras stays 1 (verified: two poses fused through
        # separate integrate() calls land in one map exactly where predicted).
        h = max(c["height"] for c in cameras.values())
        w = max(c["width"] for c in cameras.values())
        self.mapper = Mapper(
            MapperCfg(
                voxel_size=mapper_cfg["voxel_size"],
                esdf_voxel_size=mapper_cfg["esdf_voxel_size"],
                extent_meters_xyz=tuple(mapper_cfg["extent"]),
                extent_esdf_meters_xyz=tuple(mapper_cfg["extent"]),
                grid_center=torch.tensor(mapper_cfg["grid_center"], dtype=torch.float32),
                truncation_distance=mapper_cfg["voxel_size"] * 4.0,
                depth_minimum_distance=mapper_cfg["depth_min"],
                depth_maximum_distance=mapper_cfg["depth_max"],
                decay_factor=mapper_cfg["decay_factor"],
                frustum_decay_factor=mapper_cfg["frustum_decay_factor"],
                minimum_tsdf_weight=mapper_cfg["minimum_tsdf_weight"],
                num_cameras=1,
                image_height=h,
                image_width=w,
            )
        )
        # Every camera sees the robot -- the wrist one stares at its own
        # gripper, the overhead one looks down on the whole arm -- so each needs
        # self-masking, or the arm maps itself and then refuses to move.
        #
        # ONE SEGMENTER PER CAMERA, deliberately. RobotSegmenter builds its
        # projection rays on the first frame it ever sees and then freezes them
        # (get_pointcloud_from_depth only calls update_camera_projection while
        # _projection_rays is None), so a shared instance would silently mask
        # the second camera using the first one's intrinsics. Per-camera rigs
        # keep each camera's resolution and FOV independent.
        #
        # Built directly rather than via RobotSegmenter.from_robot_file: that
        # helper cannot pass ops_dtype, and the default (bfloat16) is rejected
        # by the segmenter's own tensor check, which only accepts float16 or
        # float32. Upstream bug in 0.8 main; float32 sidesteps it.
        # 0.05 is too tight for a wrist camera: the gripper sits ~0.15 m from
        # the lens, so a few of its pixels survive the mask, get fused, and then
        # the robot's own start state reads as in-collision -- after which every
        # plan fails. 0.12 clears the arm's immediate surroundings.
        # A camera that derives its pose by forward kinematics needs its frame
        # to actually be one. Without this the failure is a KeyError on
        # tool_poses[...] inside the first integrate, several seconds into a
        # run, naming a dict rather than the problem. The common way to hit it
        # is a mapping scene with --robot ur5e: camera_link only exists in the
        # 2F-85 URDF.
        missing = {
            name: spec["link"] for name, spec in cameras.items()
            if "link" in spec and spec["link"] not in kin.tool_frames
        }
        if missing:
            named = ", ".join(f"{n} wants {l!r}" for n, l in missing.items())
            raise SystemExit(
                f"[planner] this robot has no frame for: {named}.\n"
                f"[planner] Its tool_frames are {list(kin.tool_frames)}.\n"
                f"[planner] A camera's \"link\" must be one of those -- add it to "
                f"tool_frames in the robot config, or run a robot that has it."
            )

        self.rigs = {}
        for cam_name, spec in cameras.items():
            self.rigs[cam_name] = {
                "segmenter": RobotSegmenter(
                    kin, distance_threshold=mapper_cfg["self_mask_margin"],
                    use_cuda_graph=False, ops_dtype=torch.float32,
                ),
                "filter": FilterDepth(
                    image_shape=(spec["height"], spec["width"]),
                    depth_minimum_distance=mapper_cfg["depth_min"],
                    depth_maximum_distance=mapper_cfg["depth_max"],
                ),
                # Static cameras carry their pose here; the rest get it by FK
                # from the joint state that arrives with each frame.
                "link": spec.get("link"),
                "pose": (None if "pose" not in spec
                         else Pose.from_list(list(spec["pose"]))),
                "frames": 0,
            }
        # sphere index -> link name, so a self-hit can say WHICH link.
        self.sphere_link = []
        for name in robot_dict["robot_cfg"]["kinematics"]["collision_link_names"]:
            n = len(robot_dict["robot_cfg"]["kinematics"]["collision_spheres"][name])
            self.sphere_link.extend([name] * n)

        self.frames = 0
        self.voxel_grid = None
        self.last_esdf_ms = 0.0
        self.occupied = 0
        self.self_hits = 0
        # Whatever volumes the scene asked to be watched, pre-resolved to
        # corner tensors. "The map has tall stuff in it" is not the same as
        # "the map has THE OBSTACLE", and conflating the two cost a lot of
        # debugging -- so the scene names the volume it cares about and this
        # process stays ignorant of what is in it.
        self.watch = [
            (w.name,
             torch.tensor(w.bounds()[0], device="cuda"),
             torch.tensor(w.bounds()[1], device="cuda"))
            for w in scene.watch
        ]
        self.watch_report = ""
        self.tall = 0
        self.tall_where = ""
        self.last_q = None

    def integrate(self, depth: torch.Tensor, K: torch.Tensor, q: torch.Tensor,
                  cam_name: str):
        """Fuse one frame from one camera.

        depth is (H, W) metres, K is 3x3, q is (1, dof). Each camera is
        integrated on its own rather than batched with the others: a fixed
        camera never moves, so pairing it with the wrist camera would drag it
        into depth-lag bookkeeping it does not need.
        """
        rig = self.rigs[cam_name]
        js = JointState.from_position(q, joint_names=self.kin.joint_names)
        # Static cameras carry their pose; the rest are frames in the URDF, so
        # the pose comes from forward kinematics on the joint state we were
        # sent. Either way the robot's own spheres come from q, because every
        # camera here has the arm somewhere in its view.
        cam_pose = (rig["pose"] if rig["pose"] is not None
                    else self.kin.compute_kinematics(js).tool_poses[rig["link"]])

        # Everything downstream wants a leading camera/batch dimension.
        depth_b = depth.unsqueeze(0)  # (1, H, W)
        k_b = K.unsqueeze(0)  # (1, 3, 3)

        # depth_to_meter defaults to 0.001 because RealSense hardware reports
        # millimetres. Isaac Sim's distance_to_image_plane is already in metres,
        # so without this the whole point cloud collapses to ~1 mm from the lens,
        # lands inside the robot's own spheres, and 100% of the image is masked
        # away as "robot" -- which is why the map stayed empty.
        _, masked = rig["segmenter"].get_robot_mask_from_active_js(
            CameraObservation(
                name=cam_name, depth_image=depth_b, intrinsics=k_b,
                pose=cam_pose, depth_to_meter=1.0,
            ),
            js,
        )
        filtered, _ = rig["filter"](masked)

        # RGB is unused here, but Mapper.integrate dereferences it
        # unconditionally even though the colour grid is nominally optional.
        rgb = torch.full(
            (1, depth.shape[0], depth.shape[1], 3), 128, dtype=torch.uint8, device=depth.device
        )
        self.mapper.integrate(
            CameraObservation(
                name=cam_name,
                depth_image=filtered,
                rgb_image=rgb,
                intrinsics=k_b,
                pose=cam_pose,
                depth_to_meter=1.0,
            )
        )
        self.last_q = q
        rig["frames"] += 1
        self.frames += 1

    def _count_self_hits(self, voxels) -> int:
        """Occupied voxels sitting inside the robot at its last known pose."""
        if voxels.centers is None or len(voxels.centers) == 0 or self.last_q is None:
            return 0
        js = JointState.from_position(self.last_q, joint_names=self.kin.joint_names)
        sph = self.kin.compute_kinematics(js).robot_spheres.reshape(-1, 4)
        sph = sph[sph[:, 3] > 0]
        d = torch.cdist(voxels.centers.float(), sph[:, :3].float())
        return int((d < sph[:, 3].unsqueeze(0)).any(dim=1).sum().item())

    def self_hit_report(self, q: torch.Tensor) -> str:
        """How much of the map is sitting inside the robot's own body?

        If self-masking is leaking, the arm maps itself, the map then says the
        arm is inside an obstacle, and every plan fails. This counts occupied
        voxels that fall within the robot's collision spheres at pose q.
        """
        voxels = self.mapper.extract_occupied_voxels()
        centers = voxels.centers
        if centers is None or len(centers) == 0:
            return "map empty"
        js = JointState.from_position(q, joint_names=self.kin.joint_names)
        spheres = self.kin.compute_kinematics(js).robot_spheres.reshape(-1, 4)
        spheres = spheres[spheres[:, 3] > 0]
        d = torch.cdist(centers.float(), spheres[:, :3].float())
        hit = d < spheres[:, 3].unsqueeze(0)
        inside = hit.any(dim=1).sum().item()
        if not inside:
            return f"{len(centers)} occupied voxels, none inside the robot"
        # Which links? sphere_index -> link, using the config's declared order.
        per_link = {}
        for si in hit.any(dim=0).nonzero().flatten().tolist():
            per_link[self.sphere_link[si]] = per_link.get(self.sphere_link[si], 0) + 1
        top = ", ".join(f"{k}:{v}" for k, v in
                        sorted(per_link.items(), key=lambda kv: -kv[1])[:5])
        return f"{len(centers)} voxels, {inside} inside the robot [{top}]"

    def refresh_esdf(self, planner):
        """Recompute the distance field and hand it to the planner.

        An ESDF built from an EMPTY map reads as zero distance everywhere,
        which the planner interprets as "solid obstacle everywhere" and then
        nothing is reachable. So the grid only goes in once the map actually
        holds something; until then the planner sees the static scene alone.
        """
        t0 = time.perf_counter()
        self.voxel_grid = self.mapper.compute_esdf()
        torch.cuda.synchronize()
        self.last_esdf_ms = (time.perf_counter() - t0) * 1e3

        voxels = self.mapper.extract_occupied_voxels()
        self.occupied = 0 if voxels.centers is None else len(voxels.centers)
        self.self_hits = self._count_self_hits(voxels)

        # Per watched volume: how many occupied voxels are inside it, and how
        # high they reach. The z extent is the number that separates "the
        # cameras can see this obstacle" from "the cameras can see the BOTTOM
        # of it" -- a wrist camera never sees above its own altitude, so its
        # counts stop around 0.36 m however tall the obstacle really is.
        parts = []
        for name, lo_t, hi_t in self.watch:
            n, zs = 0, ""
            if voxels.centers is not None and len(voxels.centers):
                c = voxels.centers.float()
                inside = ((c >= lo_t) & (c <= hi_t)).all(dim=1)
                n = int(inside.sum().item())
                if n:
                    z = c[inside][:, 2]
                    zs = f" z{z.min():.2f}..{z.max():.2f}"
            parts.append(f"{n} in {name}{zs}")
        self.watch_report = (", ".join(parts) + ", ") if parts else ""

        # Broader and weaker: anything standing well above the table. Catches
        # geometry a watched volume no longer covers -- e.g. after it moves.
        self.tall = 0
        self.tall_where = ""
        if voxels.centers is not None and len(voxels.centers):
            m = voxels.centers[:, 2] > 0.15
            self.tall = int(m.sum().item())
            if self.tall:
                t = voxels.centers[m]
                lo = t.min(0).values.cpu().numpy()
                hi = t.max(0).values.cpu().numpy()
                self.tall_where = (f"x{lo[0]:+.2f}..{hi[0]:+.2f} "
                                   f"y{lo[1]:+.2f}..{hi[1]:+.2f} "
                                   f"z{lo[2]:+.2f}..{hi[2]:+.2f}")

        world = static_scene(self.scene)
        if self.occupied > 0:
            world.voxel = [self.voxel_grid]
        planner.update_world(world)


def build(robot_key, scene, use_cuda_graph=True):
    spec = ROBOTS[robot_key]
    content = ContentPath(
        robot_config_absolute_path=f"{ROOT}/{spec['config']}",
        robot_urdf_absolute_path=f"{ROOT}/{spec['urdf']}",
        robot_asset_absolute_path=f"{ROOT}/assets/robot/ur_description",
    )
    robot_dict = load_robot_yaml(content)

    # Two views of the same URDF. `kin` keeps every tool frame, including
    # camera_link, so the mapper can get the camera pose by forward kinematics.
    # The planner gets a copy with only the real goal frame: any frame left in
    # tool_frames becomes something plan_pose demands a target for.
    kin = Kinematics(KinematicsCfg.from_content_path(content))
    planner_dict = copy.deepcopy(robot_dict)
    planner_dict["robot_cfg"]["kinematics"]["tool_frames"] = [spec["tool_frame"]]

    planner = MotionPlanner(
        MotionPlannerCfg.create(
            robot=planner_dict,
            scene_model=static_scene(scene),
            use_cuda_graph=use_cuda_graph,
            interpolation_dt=SIM_DT,
            # Reserve room for the ESDF grid up front: the planner is built
            # before the map exists, and update_world cannot grow the cache.
            collision_cache={
                "obb": 32,
                "voxel": {
                    "layers": 1,
                    "dims": list(scene.mapper["extent"]),
                    "voxel_size": scene.mapper["esdf_voxel_size"],
                },
            },
        )
    )
    planner.warmup()
    return kin, planner, robot_dict, spec["tool_frame"]


def handle_plan(planner, kin, tool_frame, header):
    q = torch.tensor([header["q"]], device="cuda", dtype=torch.float32)
    start = JointState.from_position(q, joint_names=kin.joint_names)
    goal = GoalToolPose.from_poses({tool_frame: Pose.from_list(header["target"])})
    res = planner.plan_pose(goal, start)
    ok = res is not None and bool(res.success[0])
    out = {"ok": ok, "joint_names": list(kin.joint_names), "tool_frame": tool_frame}
    if not ok:
        out["status"] = str(getattr(res, "status", "unknown"))
        return out, b""
    # The trajectory is WIDER than the cspace: cuRobo appends the joints it
    # was told to lock, so a 6-DOF plan on a robot with a locked gripper comes
    # back 7 columns wide, with the gripper last. Select by the trajectory's
    # own joint_names rather than trusting its width -- sending the extra
    # column would hand the client a vector one longer than the joint list it
    # was given, and nothing would say so.
    itraj = res.interpolated_trajectory
    traj = itraj.position[0, 0]
    names = list(itraj.joint_names) if itraj.joint_names is not None \
        else list(kin.joint_names)
    keep = [names.index(j) for j in kin.joint_names]
    if keep != list(range(traj.shape[1])):
        traj = traj[:, keep]

    n = int(res.interpolated_last_tstep[0]) if res.interpolated_last_tstep is not None \
        else traj.shape[0]
    n = max(2, min(n, traj.shape[0]))
    out["solve_ms"] = float(res.solve_time * 1e3)
    out["n"] = n
    out["dof"] = traj.shape[1]
    return out, traj[:n].contiguous().cpu().numpy().astype(np.float32).tobytes()


def main():
    import warp as wp

    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
    ap.add_argument("--scene", default=scenes.DEFAULT, choices=scenes.available(),
                    help="scene module under scripts/scenes/")
    ap.add_argument("--no-mapping", action="store_true")
    ap.add_argument("--no-cuda-graph", action="store_true",
                    help="build the planner without CUDA graphs")
    args = ap.parse_args()

    scene = scenes.load(args.scene)
    print(f"[planner] warp {wp.config.version}  robot {args.robot}  "
          f"scene {args.scene}", flush=True)
    print("[planner] building (first run compiles CUDA kernels)...", flush=True)
    kin, planner, robot_dict, tool_frame = build(
        args.robot, scene, use_cuda_graph=not args.no_cuda_graph)
    print(f"[planner] planner ready. tool={tool_frame}", flush=True)

    mapping = None
    if not args.no_mapping:
        mapping = Mapping(robot_dict, kin, scene)
        cams = ", ".join(
            f"{n} ({'fixed' if scene.cameras[n].get('pose') else scene.cameras[n]['link']})"
            for n in mapping.rigs)
        print(f"[planner] mapper ready. {scene.mapper['voxel_size'] * 100:.1f} cm "
              f"TSDF / {scene.mapper['esdf_voxel_size'] * 100:.0f} cm ESDF, "
              f"{mapping.mapper.memory_usage_mb():.0f} MB", flush=True)
        print(f"[planner] cameras: {cams}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((HOST, PORT))
    except OSError as exc:
        # Closing the Isaac Sim window does not stop this process, so the
        # usual way to hit this is a server left over from the last run. The
        # trap is that the new server dies here while the sim happily connects
        # to the OLD one and plans against whatever scene THAT loaded.
        raise SystemExit(
            f"[planner] cannot bind {HOST}:{PORT} ({exc}).\n"
            f"[planner] A previous planner_server is probably still running -- "
            f"closing the sim window does not stop it.\n"
            f"[planner]   ss -ltnp | grep {PORT}     # see what holds it\n"
            f"[planner]   fuser -k {PORT}/tcp        # stop it"
        ) from exc
    srv.listen(1)
    print(f"[planner] listening on {HOST}:{PORT}", flush=True)

    while True:
        conn, _ = srv.accept()
        print("[planner] client connected", flush=True)
        try:
            while True:
                header, payload = recv_msg(conn)
                op = header.get("op")

                if op == "scene":
                    # Handshake. The two processes never exchange geometry, so
                    # a scene disagreement would otherwise show up only as
                    # inexplicably wrong plans.
                    asked = header.get("scene")
                    if asked != args.scene:
                        print(f"[planner] client wants scene {asked!r}, this "
                              f"server is running {args.scene!r} - it will stop",
                              flush=True)
                    send_msg(conn, {"ok": asked == args.scene, "scene": args.scene})

                elif op == "plan":
                    out, blob = handle_plan(planner, kin, tool_frame, header)
                    if out["ok"]:
                        print(f"[planner] plan ok: {out['solve_ms']:.0f} ms, "
                              f"{out['n']} waypoints", flush=True)
                    else:
                        diag = ""
                        if mapping is not None:
                            q = torch.tensor([header["q"]], device="cuda",
                                             dtype=torch.float32)
                            diag = " | " + mapping.self_hit_report(q)
                        print(f"[planner] plan FAILED: {out['status']}{diag}", flush=True)
                    send_msg(conn, out, blob)

                elif op == "map":
                    if mapping is None:
                        send_msg(conn, {"ok": False, "reason": "mapping disabled"})
                        continue
                    h, w = header["h"], header["w"]
                    depth = torch.frombuffer(
                        bytearray(payload), dtype=torch.float32
                    ).reshape(h, w).cuda()
                    K = torch.tensor(header["K"], device="cuda", dtype=torch.float32)
                    q = torch.tensor([header["q"]], device="cuda", dtype=torch.float32)
                    cam_name = header.get("cam")
                    if cam_name not in mapping.rigs:
                        # Refuse rather than guess. Silently defaulting to the
                        # wrist camera would self-mask a fixed camera's frame
                        # with the wrong pose and quietly poison the map.
                        print(f"[planner] dropping frame from unknown camera "
                              f"{cam_name!r}", flush=True)
                        continue
                    mapping.integrate(depth, K, q, cam_name)
                    refreshed = (mapping.frames
                                 % scene.mapper["esdf_every_n_frames"] == 0)
                    if refreshed:
                        mapping.refresh_esdf(planner)
                        per_cam = " ".join(f"{n}:{r['frames']}"
                                           for n, r in mapping.rigs.items())
                        print(f"[planner] map: {mapping.frames} frames fused "
                              f"({per_cam}), "
                              f"ESDF {mapping.last_esdf_ms:.1f} ms, "
                              f"{mapping.occupied} voxels, "
                              f"{mapping.watch_report}"
                              f"{mapping.tall} tall [{mapping.tall_where}], "
                              f"{mapping.self_hits} on the robot"
                              + ("" if mapping.occupied else " (map empty - "
                                 "planner using static scene only)"), flush=True)
                    # No reply: "map" is fire-and-forget. A 1.2 MB frame every
                    # few sim steps with a synchronous round-trip each time
                    # throttled the simulation to a crawl.
                else:
                    send_msg(conn, {"ok": False, "reason": f"unknown op {op}"})
        except (ConnectionError, OSError) as exc:
            print(f"[planner] client gone ({exc})", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
