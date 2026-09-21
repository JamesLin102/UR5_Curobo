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
from proto import recv_msg, send_msg  # noqa: E402
from scene_def import (  # noqa: E402
    CAMERA, DEFAULT_ROBOT, DRAG_CUBE, HOST, MAPPER, OBSTACLES, PORT, ROBOTS,
    SIM_DT,
)

from curobo.types import CameraObservation, ContentPath, GoalToolPose, JointState, Pose  # noqa: E402
from curobo.kinematics import Kinematics, KinematicsCfg  # noqa: E402
from curobo.scene import Cuboid, Scene  # noqa: E402
from curobo.perception import FilterDepth, Mapper, MapperCfg, RobotSegmenter  # noqa: E402
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: E402
from curobo._src.robot.loader.util import load_robot_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def static_scene():
    return Scene(cuboid=[Cuboid(name=n, dims=d, pose=p) for n, d, p, _ in OBSTACLES])


class Mapping:
    """Wrist-camera RGB-D -> TSDF -> ESDF, fed back into the planner's scene."""

    def __init__(self, robot_dict, kin):
        self.kin = kin
        h, w = CAMERA["height"], CAMERA["width"]
        self.mapper = Mapper(
            MapperCfg(
                voxel_size=MAPPER["voxel_size"],
                esdf_voxel_size=MAPPER["esdf_voxel_size"],
                extent_meters_xyz=tuple(MAPPER["extent"]),
                extent_esdf_meters_xyz=tuple(MAPPER["extent"]),
                grid_center=torch.tensor(MAPPER["grid_center"], dtype=torch.float32),
                truncation_distance=MAPPER["voxel_size"] * 4.0,
                depth_minimum_distance=MAPPER["depth_min"],
                depth_maximum_distance=MAPPER["depth_max"],
                decay_factor=MAPPER["decay_factor"],
                frustum_decay_factor=MAPPER["frustum_decay_factor"],
                minimum_tsdf_weight=MAPPER["minimum_tsdf_weight"],
                num_cameras=1,
                image_height=h,
                image_width=w,
            )
        )
        # The gripper fills a good part of a wrist camera's view. Without this
        # the arm maps itself and then refuses to move.
        #
        # Built directly rather than via RobotSegmenter.from_robot_file: that
        # helper cannot pass ops_dtype, and the default (bfloat16) is rejected
        # by the segmenter's own tensor check, which only accepts float16 or
        # float32. Upstream bug in 0.8 main; float32 sidesteps it.
        # 0.05 is too tight for a wrist camera: the gripper sits ~0.15 m from
        # the lens, so a few of its pixels survive the mask, get fused, and then
        # the robot's own start state reads as in-collision -- after which every
        # plan fails. 0.12 clears the arm's immediate surroundings.
        self.segmenter = RobotSegmenter(
            kin, distance_threshold=MAPPER["self_mask_margin"],
            use_cuda_graph=False, ops_dtype=torch.float32,
        )
        self.depth_filter = FilterDepth(
            image_shape=(h, w),
            depth_minimum_distance=MAPPER["depth_min"],
            depth_maximum_distance=MAPPER["depth_max"],
        )
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
        # The cube's own bounding box, for the decisive count below. "The map
        # has tall stuff in it" is not the same as "the map has THE CUBE", and
        # conflating the two cost a lot of debugging.
        _, cdim, cpose, _ = DRAG_CUBE
        self.cube_lo = torch.tensor(
            [cpose[i] - cdim[i] / 2 for i in range(3)], device="cuda")
        self.cube_hi = torch.tensor(
            [cpose[i] + cdim[i] / 2 for i in range(3)], device="cuda")
        self.tall = 0
        self.tall_where = ""
        self.in_cube = 0
        self.last_q = None

    def integrate(self, depth: torch.Tensor, K: torch.Tensor, q: torch.Tensor):
        """Fuse one frame. depth is (H, W) metres, K is 3x3, q is (1, dof)."""
        js = JointState.from_position(q, joint_names=self.kin.joint_names)
        cam_pose = self.kin.compute_kinematics(js).tool_poses[CAMERA["link"]]

        # Everything downstream wants a leading camera/batch dimension.
        depth_b = depth.unsqueeze(0)  # (1, H, W)
        k_b = K.unsqueeze(0)  # (1, 3, 3)

        # depth_to_meter defaults to 0.001 because RealSense hardware reports
        # millimetres. Isaac Sim's distance_to_image_plane is already in metres,
        # so without this the whole point cloud collapses to ~1 mm from the lens,
        # lands inside the robot's own spheres, and 100% of the image is masked
        # away as "robot" -- which is why the map stayed empty.
        _, masked = self.segmenter.get_robot_mask_from_active_js(
            CameraObservation(
                name="wrist_d435i", depth_image=depth_b, intrinsics=k_b,
                pose=cam_pose, depth_to_meter=1.0,
            ),
            js,
        )
        filtered, _ = self.depth_filter(masked)

        # RGB is unused here, but Mapper.integrate dereferences it
        # unconditionally even though the colour grid is nominally optional.
        rgb = torch.full(
            (1, depth.shape[0], depth.shape[1], 3), 128, dtype=torch.uint8, device=depth.device
        )
        self.mapper.integrate(
            CameraObservation(
                name="wrist_d435i",
                depth_image=filtered,
                rgb_image=rgb,
                intrinsics=k_b,
                pose=cam_pose,
                depth_to_meter=1.0,
            )
        )
        self.last_q = q
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

        # The decisive number: occupied voxels inside the cube's own bounding
        # box. "The map has tall stuff" is not the same as "the map has THE
        # CUBE", and conflating the two cost a lot of debugging.
        self.in_cube = 0
        if voxels.centers is not None and len(voxels.centers):
            c = voxels.centers.float()
            self.in_cube = int(
                ((c >= self.cube_lo) & (c <= self.cube_hi)).all(dim=1).sum().item())

        # Broader and weaker: anything standing well above the table. The cell
        # is otherwise flat, so this catches tall geometry the cube's nominal
        # bounding box no longer covers -- e.g. after it has been dragged.
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

        scene = static_scene()
        if self.occupied > 0:
            scene.voxel = [self.voxel_grid]
        planner.update_world(scene)


def build(robot_key, use_cuda_graph=True):
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
            scene_model=static_scene(),
            use_cuda_graph=use_cuda_graph,
            interpolation_dt=SIM_DT,
            # Reserve room for the ESDF grid up front: the planner is built
            # before the map exists, and update_world cannot grow the cache.
            collision_cache={
                "obb": 32,
                "voxel": {
                    "layers": 1,
                    "dims": list(MAPPER["extent"]),
                    "voxel_size": MAPPER["esdf_voxel_size"],
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
    traj = res.interpolated_trajectory.position[0, 0]
    n = int(res.interpolated_last_tstep[0]) if res.interpolated_last_tstep is not None \
        else traj.shape[0]
    n = max(2, min(n, traj.shape[0]))
    out["solve_ms"] = float(res.solve_time * 1e3)
    out["n"] = n
    out["dof"] = traj.shape[1]
    return out, traj[:n].cpu().numpy().astype(np.float32).tobytes()


def main():
    import warp as wp

    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
    ap.add_argument("--no-mapping", action="store_true")
    ap.add_argument("--no-cuda-graph", action="store_true",
                    help="build the planner without CUDA graphs")
    args = ap.parse_args()

    print(f"[planner] warp {wp.config.version}  robot {args.robot}", flush=True)
    print("[planner] building (first run compiles CUDA kernels)...", flush=True)
    kin, planner, robot_dict, tool_frame = build(
        args.robot, use_cuda_graph=not args.no_cuda_graph)
    print(f"[planner] planner ready. tool={tool_frame}", flush=True)

    mapping = None
    if not args.no_mapping:
        mapping = Mapping(robot_dict, kin)
        print(f"[planner] mapper ready. {MAPPER['voxel_size'] * 100:.1f} cm TSDF / "
              f"{MAPPER['esdf_voxel_size'] * 100:.0f} cm ESDF, "
              f"{mapping.mapper.memory_usage_mb():.0f} MB", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"[planner] listening on {HOST}:{PORT}", flush=True)

    while True:
        conn, _ = srv.accept()
        print("[planner] client connected", flush=True)
        try:
            while True:
                header, payload = recv_msg(conn)
                op = header.get("op")

                if op == "plan":
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
                    mapping.integrate(depth, K, q)
                    refreshed = mapping.frames % MAPPER["esdf_every_n_frames"] == 0
                    if refreshed:
                        mapping.refresh_esdf(planner)
                        print(f"[planner] map: {mapping.frames} frames fused, "
                              f"ESDF {mapping.last_esdf_ms:.1f} ms, "
                              f"{mapping.occupied} voxels, {mapping.in_cube} INSIDE-CUBE, "
                              f"{mapping.tall} tall, "
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
