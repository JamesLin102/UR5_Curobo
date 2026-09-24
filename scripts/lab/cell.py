"""LabCell: N cells on Isaac Lab, driven by the same primitives as sim_env.SimEnv.

    cell = LabCell(env)                     # built by lab.tasks.base.CellEnv
    results = cell.run(env_ids, programs)   # cell_api ops, all envs at once
    obs = cell.observe(env_ids)             # CellObs: arrays with an env axis
    view = cell.view(0)                     # one env as a cell_api.CellLike

Primitives, not a task: what a leg IS lives in legs.py, what a step means
lives in each example's lab_env.py. Every pose in or out is in the scene's own
coordinates -- relative to the environment's origin -- so N cells look like N
copies of the one the scene describes, and the planner, which knows nothing
of environments, sees each of them exactly as it would on the Isaac Sim side.

The joint bookkeeping (planner order vs simulator order, the gripper's
coupling and its pinned followers) follows sim_env.SimEnv line for line; see
the comments there for why each piece is the way it is.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

import omni.kit.app
import isaaclab.sim as sim_utils
from isaaclab.utils.math import combine_frame_transforms

from cell_api import (HOLDING_RADIUS, MOVE_Z_STEPS, Grip, Idle, MoveTo, MoveZ, Observation,
                      ResetOptions, Scan)
from rig import ROBOTS, SIM_DT
from urdf_frames import quat_to_matrix

from .markers import GoalMarkers
from .programs import ProgramRunner


@dataclass
class CellObs:
    """observe() for k environments: row i belongs to env_ids[i]. Env-local poses."""
    q: np.ndarray                      # (k, dof) arm joints, planner order
    gripper: np.ndarray                # (k,) leader joint, rad
    tool_pose: np.ndarray              # (k, 7) [x, y, z, qw, qx, qy, qz]
    objects: Dict[str, np.ndarray]     # name -> (k, 7), ground truth
    holding: np.ndarray                # (k,) bool
    sim_time: float
    depth: Optional[Dict[str, np.ndarray]] = None   # name -> (k, H, W)


class LabCell:
    def __init__(self, env):
        self.sim, self.scene = env.sim, env.scene
        self.cfg, self.spec, self.pool = env.cfg.cell, env.scene_spec, env.pool
        h = env.handles
        self.handles = h
        self.robot, self.payload, self.cams = h.robot, h.payload, h.cameras
        self.num_envs = self.scene.num_envs
        self.device = self.sim.device
        self.dt = SIM_DT
        self.close = env.close
        self.origins = self.scene.env_origins.cpu().numpy().astype(np.float64)
        self.rig = ROBOTS[self.cfg.robot]
        self.log(f"robot: {self.cfg.robot}  urdf: {os.path.basename(h.urdf)}  "
                 f"envs: {self.num_envs}")

        self._map_joints()
        body, pos, quat = h.tool
        ids, _ = self.robot.find_bodies(f"^{body}$")
        if len(ids) != 1:
            raise RuntimeError(f"tool frame {self.rig['tool_frame']} should be on body "
                               f"{body}, which the articulation does not have")
        self.tool_body = ids[0]
        self.tool_pos = torch.tensor(pos, dtype=torch.float32, device=self.device)
        self.tool_quat = torch.tensor(quat, dtype=torch.float32, device=self.device)
        self.log(f"tool frame {self.rig['tool_frame']}: on body {body}, "
                 f"offset {np.round(pos, 4).tolist()}")

        self.bodies = {name: sim_utils.XformPrimView(path, device=self.device,
                                                     sync_usd_on_fabric_write=True)
                       for name, path in h.unmapped.items()}
        self._payload_home = {b[0]: np.array(b[2][:3], dtype=np.float64)
                              for b in self.spec.payload}

        # Rendering is only needed for the cameras, or for a person to watch.
        self._render = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self.steps = 0
        self.goal = [None] * self.num_envs       # last commanded tool pose, env-local
        self.closed = np.zeros(self.num_envs, dtype=bool)
        self._arm_cmd = np.zeros((self.num_envs, len(self.arm_idx)))
        self._grip_cmd = np.zeros((self.num_envs, len(self.grip_idx_all)))
        self._arm_dirty = np.zeros(self.num_envs, dtype=bool)
        self._grip_dirty = np.zeros(self.num_envs, dtype=bool)
        lag = self.cfg.depth_lag
        self._hist = np.zeros((self.num_envs, lag + 1, len(self.arm_idx)))
        self._hist_n = np.zeros(self.num_envs, dtype=int)
        self._hist_at = 0
        self._refresh()

        self.runner = ProgramRunner(self)
        self.markers = GoalMarkers(self, self.cfg.markers)

        self._cameras_checked_after_reset = False

        all_ids = list(range(self.num_envs))
        self._home(all_ids)
        self.idle(all_ids, 60)
        self.intrinsics = {}
        if self.cams:
            self.idle(all_ids, 10)   # let the annotators produce their first frame
            for name, cam in self.cams.items():
                K = cam.data.intrinsic_matrices.cpu().numpy().astype(np.float64)
                self.intrinsics[name] = K
                k = K[0]
                self.log(f"{name} intrinsics: fx={k[0, 0]:.1f} fy={k[1, 1]:.1f} "
                         f"cx={k[0, 2]:.1f} cy={k[1, 2]:.1f}")
            self.check_cameras()
        self.markers.draw()

    # --- joints ---------------------------------------------------------------

    def _map_joints(self):
        """Planner order onto simulator order, arm and gripper apart. As in SimEnv."""
        target = self.spec.targets[0] if self.spec.targets else None
        planned = self.pool.joint_names(self.spec.home, target)
        sim_names = list(self.robot.joint_names)
        missing = [j for j in planned if j not in sim_names]
        if missing:
            raise RuntimeError(f"planner joints absent from the simulator: {missing}")
        self.arm_names = planned
        self.arm_idx = [sim_names.index(j) for j in planned]

        grip = self.rig["gripper"]
        coupling_all = grip["joints"]
        follower = grip.get("pinned_follower")
        leader = next(iter(coupling_all))
        if leader not in sim_names:
            raise RuntimeError(f"gripper joint {leader} absent from the simulator")
        self.grip_idx = sim_names.index(leader)
        coupling = {j: m for j, m in coupling_all.items()
                    if j in sim_names and not (follower and follower in j)}
        self.grip_idx_all = [sim_names.index(j) for j in coupling]
        self.grip_mult = np.array(list(coupling.values()), dtype=np.float64)
        self.all_grip_names = [j for j in coupling_all if j in sim_names]
        self.all_grip_idx = [sim_names.index(j) for j in self.all_grip_names]
        self.all_grip_sign = np.array([coupling_all[j] for j in self.all_grip_names])
        prefix = os.path.commonprefix(self.all_grip_names)
        self.grip_short = [n[len(prefix):].removesuffix("_joint") for n in self.all_grip_names]
        self.grip_open, self.grip_closed = grip["open"], grip["closed"]
        self.grip_speed = grip["speed"]

        t = lambda idx: torch.tensor(idx, dtype=torch.long, device=self.device)  # noqa: E731
        self._arm_ids, self._grip_ids = t(self.arm_idx), t(self.grip_idx_all)
        self.log(f"joints: {len(sim_names)} in sim, {len(planned)} planned")
        self.log(f"gripper: {len(coupling)} load-bearing joints driven, "
                 f"{self.grip_open} open .. {self.grip_closed} closed; "
                 f"{len(self.all_grip_names) - len(coupling)} set by the linkage")

    # --- plumbing ---------------------------------------------------------------

    def log(self, msg, env=None):
        if self.cfg.verbose:
            tag = "" if env is None or self.num_envs == 1 else f"[env {env}] "
            print(f"[lab] {tag}{msg}", flush=True)

    @property
    def running(self):
        return omni.kit.app.get_app().is_running()

    def _ids(self, env_ids):
        return torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)

    def tick(self):
        """One physics step for every environment, then read back what it did."""
        self._flush()
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        if self._render:
            self.sim.render()
        self.scene.update(self.dt)
        self.update_cameras()
        self.steps += 1
        self._refresh()
        self._hist[:, self._hist_at] = self._q
        self._hist_at = (self._hist_at + 1) % self._hist.shape[1]
        self._hist_n = np.minimum(self._hist_n + 1, self._hist.shape[1])

    def update_cameras(self):
        """The cameras are not in scene.sensors (see scene_cfg.spawn_cell), so
        the scene does not update them: this does, after every step."""
        for cam in self.cams.values():
            cam.update(self.dt)

    def _refresh(self):
        """One device-to-host copy per tick; everything else reads these."""
        q = self.robot.data.joint_pos.cpu().numpy().astype(np.float64)
        self._q, self._q_all = q[:, self.arm_idx], q

    def _flush(self):
        """Hand this tick's commands to the articulation, one call per joint group."""
        for dirty, cmd, ids in ((self._arm_dirty, self._arm_cmd, self._arm_ids),
                                (self._grip_dirty, self._grip_cmd, self._grip_ids)):
            if dirty.any():
                envs = np.nonzero(dirty)[0]
                self.robot.set_joint_position_target(
                    torch.tensor(cmd[envs], dtype=torch.float32, device=self.device),
                    joint_ids=ids, env_ids=self._ids(envs))
                dirty[:] = False

    def command_arm(self, env, q):
        """One planner-ordered joint vector for one env, touching nothing else."""
        self._arm_cmd[env] = q
        self._arm_dirty[env] = True

    def command_gripper(self, env, angle):
        """One commanded angle -> every driven joint of the linkage. Open-loop."""
        self._grip_cmd[env] = self.grip_mult * angle
        self._grip_dirty[env] = True

    def q_now(self, env):
        return self._q[env].copy()

    def grip_driven_q(self, env):
        return self._q_all[env, self.grip_idx_all].copy()

    def grip_spread(self, env):
        """Every gripper joint as a fraction closed, followers included."""
        q = self._q_all[env, self.all_grip_idx]
        return {n: float(v * m) for n, v, m in zip(self.grip_short, q, self.all_grip_sign)}

    def _lagged_q(self, env):
        """The arm as it was depth_lag steps ago, which is what the depth shows."""
        lag = self.cfg.depth_lag
        if self._hist_n[env] <= lag:
            return self.q_now(env)
        return self._hist[env, (self._hist_at - 1 - lag) % self._hist.shape[1]].copy()

    def fuse(self, env_ids):
        """Send each camera's depth, paired with the pose it was rendered at.

        Fire-and-forget, as on the Isaac Sim side: returns how many frames
        were sent, which says nothing about what the mapper made of them.
        """
        sent = 0
        for name, cam in self.cams.items():
            depth = cam.data.output["distance_to_image_plane"]
            for e in env_ids:
                d = np.nan_to_num(depth[e, :, :, 0].cpu().numpy().astype(np.float32),
                                  nan=0.0, posinf=0.0, neginf=0.0)
                self.pool.map_frame(e, self._lagged_q(e), d, self.intrinsics[name][e], name)
                sent += 1
        return sent

    # --- primitives, for many environments --------------------------------------

    def run(self, env_ids, programs):
        return self.runner.run(env_ids, programs)

    def idle(self, env_ids, n):
        self.run(env_ids, [[Idle(n)] for _ in env_ids])

    def _home(self, env_ids):
        """Teleport the arm to HOME, gripper open, and hold it there."""
        ids = self._ids(env_ids)
        full = self.robot.data.joint_pos[ids].clone()
        full[:, self._arm_ids] = torch.tensor(self.spec.home, dtype=full.dtype,
                                              device=self.device)
        # Every gripper joint, not only the leader: a teleport that leaves the
        # pinned knuckles where they were starts the 4-bar out of closure.
        full[:, self.all_grip_idx] = torch.tensor(
            self.grip_open * self.all_grip_sign, dtype=full.dtype, device=self.device)
        self.robot.write_joint_state_to_sim(full, torch.zeros_like(full), env_ids=ids)
        for e in env_ids:
            self.command_arm(e, self.spec.home)
            self.command_gripper(e, self.grip_open)
            self.goal[e] = None
            self.closed[e] = False
        self._refresh()

    def reset(self, env_ids, options=None):
        """Arm to HOME, gripper open, payload onto a target, map rebuilt.

        `options` is one ResetOptions for all, or one per env.
        """
        env_ids = list(env_ids)
        opts = options if isinstance(options, (list, tuple)) else [options] * len(env_ids)
        opts = [o or ResetOptions() for o in opts]
        ids = self._ids(env_ids)
        self._home(env_ids)
        for name, obj in self.payload.items():
            pose = torch.zeros((len(env_ids), 7), device=self.device)
            for k, (e, o) in enumerate(zip(env_ids, opts)):
                t = self.spec.targets[o.block_on]
                pose[k, :3] = torch.tensor(
                    [t[0], t[1], self._payload_home[name][2]] + self.origins[e])
                pose[k, 3] = 1.0
            obj.write_root_pose_to_sim(pose, env_ids=ids)
            obj.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device),
                                           env_ids=ids)
        for k, (e, o) in enumerate(zip(env_ids, opts)):
            if o.slab_pose is None:
                continue
            if not self.bodies:
                raise ValueError("slab_pose given, but the scene has no unmapped body")
            view = next(iter(self.bodies.values()))
            view.set_world_poses(
                positions=torch.tensor([np.asarray(o.slab_pose[:3]) + self.origins[e]],
                                       dtype=torch.float32, device=self.device),
                indices=[e])
        clear = [e for e, o in zip(env_ids, opts) if o.clear_map]
        if clear:
            self.pool.reset_map(clear)
        self._hist_n[env_ids] = 0
        self.run(env_ids, [[Idle(o.settle_steps)] + ([Scan()] if o.scan and self.cams else [])
                           for o in opts])
        if self.cams and not self._cameras_checked_after_reset:
            # Once more after the first reset: a camera that renders right at
            # start-up and wrong after a reset is how the Fabric pose copy
            # described in scene_cfg.spawn_cell showed itself.
            self.check_cameras(env_ids)
            self._cameras_checked_after_reset = True
        return self.observe(env_ids)

    def observe(self, env_ids, images=False):
        """The world as it is now, env-local. Object poses are ground truth."""
        env_ids = list(env_ids)
        ids = self._ids(env_ids)
        origin = torch.tensor(self.origins[env_ids], dtype=torch.float32, device=self.device)
        body = self.robot.data.body_link_pose_w[ids, self.tool_body]
        n = len(env_ids)
        pos, quat = combine_frame_transforms(body[:, :3], body[:, 3:],
                                             self.tool_pos.expand(n, 3),
                                             self.tool_quat.expand(n, 4))
        tool = torch.cat([pos - origin, quat], dim=1).cpu().numpy()
        objects = {}
        for name, obj in self.payload.items():
            p = obj.data.root_pose_w[ids]
            objects[name] = torch.cat([p[:, :3] - origin, p[:, 3:]], dim=1).cpu().numpy()
        near = np.zeros(n, dtype=bool)
        for o in objects.values():
            near |= np.linalg.norm(o[:, :3] - tool[:, :3], axis=1) < HOLDING_RADIUS
        depth = None
        if images and self.cams:
            depth = {name: np.nan_to_num(cam.data.output["distance_to_image_plane"][ids, :, :, 0]
                                         .cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                     for name, cam in self.cams.items()}
        return CellObs(q=self._q[env_ids].astype(np.float32),
                       gripper=self._q_all[env_ids, self.grip_idx].astype(np.float32),
                       tool_pose=tool.astype(np.float32), objects=objects,
                       holding=self.closed[env_ids] & near,
                       sim_time=self.steps * SIM_DT, depth=depth)

    def camera_pose(self, name, env=0):
        """4x4 env-local OPTICAL pose of a camera, as the mapper will be told it.

        A link camera's pose is forward kinematics of the body it hangs off
        -- what planner_server derives from the joint state it is sent; a
        fixed camera's is the scene's. Not Camera.data.pos_w: under Fabric
        that is not updated for a prim that rides on a rigid body.
        """
        spec = self.spec.cameras[name]
        if "link" not in spec:
            W = np.eye(4)
            W[:3, :3], W[:3, 3] = quat_to_matrix(spec["pose"][3:]), spec["pose"][:3]
            return W
        body, T = self.handles.camera_mounts[name]
        b = self.robot.find_bodies(f"^{body}$")[0][0]
        bp = self.robot.data.body_link_pose_w[env, b].cpu().numpy().astype(np.float64)
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = quat_to_matrix(bp[3:]), bp[:3] - self.origins[env]
        return M @ T

    def _scene_distance(self, pts, env=0):
        """Distance from each env-local point to the nearest surface the scene knows."""
        d = np.abs(pts[:, 2])                                   # the ground, z = 0
        boxes = [(dims, pose) for _, dims, pose, _ in self.spec.obstacles]
        for name, dims, pose, _ in self.spec.unmapped:
            if self.spec.shape(name) == "cylinder":
                R = quat_to_matrix(pose[3:])
                local = (pts - np.asarray(pose[:3])) @ R
                q = np.stack([np.linalg.norm(local[:, :2], axis=1) - dims[0] / 2.0,
                              np.abs(local[:, 2]) - dims[2] / 2.0], axis=1)
                outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
                d = np.minimum(d, np.abs(outside + np.minimum(q.max(axis=1), 0.0)))
            else:
                boxes.append((dims, pose))
        for name, dims, *_ in self.spec.payload:
            p = self.payload[name].data.root_pose_w[env].cpu().numpy().astype(np.float64)
            boxes.append((dims, list(p[:3] - self.origins[env]) + list(p[3:])))
        for dims, pose in boxes:
            R = quat_to_matrix(pose[3:])
            q = np.abs((pts - np.asarray(pose[:3])) @ R) - np.asarray(dims) / 2.0
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            d = np.minimum(d, np.abs(outside + np.minimum(q.max(axis=1), 0.0)))
        return d

    def camera_alignment(self, name, env=0, pose=None):
        """(median, fraction within 10 mm, pixels): how well a depth image fits the scene.

        Back-projects the camera's current depth image through `pose` (default:
        the pose the mapper would pair it with, camera_pose()) and measures how
        far the points land from the surfaces the scene defines.
        """
        cam = self.cams[name]
        depth = np.nan_to_num(cam.data.output["distance_to_image_plane"][env, :, :, 0]
                              .cpu().numpy().astype(np.float64), posinf=0.0)
        K = self.intrinsics[name][env]
        v, u = np.nonzero(depth > 0)
        if not len(v):
            return float("inf"), 0.0, 0
        z = depth[v, u]
        rays = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1],
                         np.ones_like(z)], axis=1)
        W = self.camera_pose(name, env) if pose is None else pose
        dist = self._scene_distance((rays * z[:, None]) @ W[:3, :3].T + W[:3, 3], env)
        return float(np.median(dist)), float((dist < 0.01).mean()), len(dist)

    def check_cameras(self, env_ids=None):
        """Verify, don't assume: does each depth image line up with the scene?

        Right, the median distance from the scene's surfaces is a few mm (the
        arm's own pixels are the outliers); a camera rendering from anywhere
        else -- wrong mount, wrong convention, a pose that does not follow the
        robot, a neighbouring cell in view -- puts it decimetres off. That is
        the failure that would otherwise only show up as a map full of
        phantom obstacles. Every environment is checked.
        """
        env_ids = range(self.num_envs) if env_ids is None else env_ids
        for name in self.cams:
            rows = [(e, *self.camera_alignment(name, e)) for e in env_ids]
            e, median, near, n = max(rows, key=lambda r: r[1])
            ok = median < 0.005
            where = f" (worst of {len(rows)} envs: env {e})" if len(rows) > 1 else ""
            self.log(f"camera {name}: {n} depth pixels, median {median * 1000:.1f} mm "
                     f"from the scene's surfaces, {near * 100:.0f}% within 10 mm{where} "
                     f"({'AGREE' if ok else 'WRONG'})")
            if not ok:
                raise RuntimeError(
                    f"camera {name}'s depth in env {e} does not line up with the scene "
                    f"through the pose the mapper would pair it with (median "
                    f"{median * 1000:.0f} mm off). Fix the mount, do not adjust the check.")

    def view(self, env=0):
        return CellView(self, env)


class CellView:
    """One environment of a LabCell as a cell_api.CellLike: blocking, scene coordinates.

    Other environments hold still while this one moves, which is what makes
    the demo loop and the oracle checks runnable on either backend unchanged.
    """

    def __init__(self, cell: LabCell, env: int):
        self.cell, self.env = cell, env
        self.scene = cell.spec
        self.payload = cell.payload
        self.bodies = cell.bodies
        self.cams = cell.cams

    @property
    def running(self):
        return self.cell.running

    @property
    def gripper_closed(self):
        return bool(self.cell.closed[self.env])

    def _one(self, op):
        return self.cell.run([self.env], [[op]])[0].results[0][1]

    def reset(self, options=None):
        return self._obs(self.cell.reset([self.env], options or ResetOptions()))

    def move_to(self, pose):
        return self._one(MoveTo(list(pose)))

    def move_tool_z(self, dz, n_steps=MOVE_Z_STEPS):
        return self._one(MoveZ(dz, n_steps=n_steps))

    def grip(self, close):
        return self._one(Grip(close=close))

    def idle(self, n):
        self._one(Idle(n))

    def scan(self):
        return self._one(Scan())

    def command_arm(self, q):
        self.cell.command_arm(self.env, q)

    def observe(self, images=False):
        return self._obs(self.cell.observe([self.env], images=images))

    def close(self):
        self.cell.close()

    @staticmethod
    def _obs(o: CellObs):
        return Observation(
            q=o.q[0], gripper=float(o.gripper[0]), tool_pose=o.tool_pose[0],
            objects={n: v[0] for n, v in o.objects.items()}, holding=bool(o.holding[0]),
            sim_time=o.sim_time,
            depth=None if o.depth is None else {n: v[0] for n, v in o.depth.items()})
