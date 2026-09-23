"""Pick-and-place as a gymnasium environment: the policy picks points, cuRobo moves.

One step is one LEG -- a pick if the gripper is empty, a place if it is
holding something. The action says where the tool should be when the gripper
acts; everything between here and there is sim_env.SimEnv's job:

    move_to(action + descend_m above)   planned by cuRobo, around the map
    move_tool_z(-descend_m)             IK + interpolation, straight down
    grip(close=True | False)
    move_tool_z(+lift_m)                back up

    action = [x, y, z, yaw]   z is grasp_frame's height at the grip; yaw turns
                              the tool about world Z, tool pointing down.

Needs planner_server.py listening, with the same --scene (and --no-mapping
if EnvCfg.mapping is False). Isaac Sim is started by the constructor.

    env = PickPlaceEnv(EnvCfg(mapping=False), headless=True)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

import math
import os
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_env  # noqa: E402
from sim_env import EnvCfg, ResetOptions  # noqa: E402


@dataclass
class TaskCfg:
    max_legs: int = 8           # truncate after this many steps
    scan: bool = True           # rescan at reset (only matters with mapping on)
    clear_map: bool = True      # forget the last episode's map at reset
    # Where the block counts as delivered: on the goal pedestal, released.
    success_xy: float = 0.02
    success_z: float = 0.01
    # Below this the block has fallen off a pedestal onto the table.
    dropped_z: float = 0.06
    # Reward. Progress is potential-based: the drop in block-to-goal distance.
    r_success: float = 1.0
    r_progress: float = 1.0
    r_failed_motion: float = -0.1   # no plan, or no IK for the vertical moves
    r_dropped: float = -1.0
    r_step: float = -0.01


class PickPlaceEnv(gym.Env):
    """Shuttle the scene's payload to the other target, one leg per step."""

    metadata = {"render_modes": []}

    # Action bounds: the cell in front of the arm, both pedestals included.
    LOW = np.array([0.25, -0.55, 0.05, -math.pi], dtype=np.float32)
    HIGH = np.array([0.65, 0.55, 0.35, math.pi], dtype=np.float32)

    def __init__(self, env_cfg: EnvCfg = None, task: TaskCfg = None, headless=True):
        super().__init__()
        self.task = task or TaskCfg()
        sim_env.launch(headless=headless)
        self.sim = sim_env.SimEnv(env_cfg or EnvCfg())
        scene = self.sim.scene
        if len(scene.payload) != 1 or len(scene.targets) != 2:
            raise ValueError("PickPlaceEnv wants one payload and two targets")
        self.block = scene.payload[0][0]
        self.descend = scene.pick["descend_m"]
        self.lift = scene.pick["lift_m"]
        # Where the block rests on each target: its spawn height, under that
        # target -- the same rule SimEnv.reset uses to put it there.
        rest_z = float(scene.payload[0][2][2])
        self.rest = [np.array([t[0], t[1], rest_z], dtype=np.float32)
                     for t in scene.targets]

        self.action_space = gym.spaces.Box(self.LOW, self.HIGH, dtype=np.float32)
        # q(6) gripper(1) tool_pose(7) block_pose(7) holding(1) goal(3)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(25,), dtype=np.float32)
        self._goal = 1
        self._legs = 0
        self._dist = 0.0

    # --- gymnasium ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        start = int(options.get("block_on", self.np_random.integers(2)))
        self._goal = 1 - start
        o = self.sim.reset(ResetOptions(
            block_on=start, slab_pose=options.get("slab_pose"),
            clear_map=self.task.clear_map, scan=self.task.scan))
        self._legs = 0
        self._dist = self._block_to_goal(o)
        return self._vec(o), {"start": start, "goal": self._goal}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), self.LOW, self.HIGH)
        pose = self.tool_pose(*a)
        before = self.sim.observe()
        leg = "place" if before.holding else "pick"
        info = {"leg": leg, "pose": pose}
        reward = self.task.r_step

        # A pick needs an open hand. After a pick that closed on nothing the
        # gripper is still shut, so open it here first, where the arm stands.
        if leg == "pick" and self.sim.gripper_closed:
            self.sim.grip(close=False)

        ok, why = self._leg(pose, close=(leg == "pick"), info=info)
        if not ok:
            reward += self.task.r_failed_motion
            info["failed"] = why

        o = self.sim.observe()
        d = self._block_to_goal(o)
        reward += self.task.r_progress * (self._dist - d)
        self._dist = d
        self._legs += 1

        success = self._delivered(o)
        dropped = (not o.holding) and float(o.objects[self.block][2]) < self.task.dropped_z
        if success:
            reward += self.task.r_success
        if dropped:
            reward += self.task.r_dropped
        terminated = success or dropped or not self.sim.running
        truncated = (not terminated) and self._legs >= self.task.max_legs
        info.update(success=success, dropped=dropped, holding=o.holding,
                    block_to_goal=d, legs=self._legs)
        return self._vec(o), float(reward), terminated, truncated, info

    def close(self):
        self.sim.close()

    # --- the task ------------------------------------------------------------

    @staticmethod
    def tool_pose(x, y, z, yaw):
        """[x, y, z, qw, qx, qy, qz] with the tool pointing down, turned by yaw.

        Tool-down is (0, 1, 0, 0), a half turn about X -- what the scene's own
        targets use. Turning that about world Z by yaw gives
        (0, cos(yaw/2), sin(yaw/2), 0).
        """
        return [float(x), float(y), float(z),
                0.0, math.cos(yaw / 2.0), math.sin(yaw / 2.0), 0.0]

    def _leg(self, pose, close, info):
        """Above, down, grip, up. Returns (ok, reason if not)."""
        above = list(pose)
        above[2] += self.descend
        r = self.sim.move_to(above)
        info.update(solve_ms=r.solve_ms, waypoints=r.waypoints, clearance=r.clearance)
        if not r.ok:
            self.sim.idle(30)
            return False, f"no plan: {r.reason}"
        self.sim.idle(20)
        if not self.sim.move_tool_z(-self.descend).ok:
            return False, "no IK going down"
        self.sim.idle(15)
        g = self.sim.grip(close=close)
        info["grip_settled"] = g.settled
        up = self.sim.move_tool_z(self.lift)
        self.sim.idle(20)
        if not up.ok:
            return False, "no IK going up"
        return True, None

    def _block_to_goal(self, o):
        return float(np.linalg.norm(o.objects[self.block][:3] - self.rest[self._goal]))

    def _delivered(self, o):
        p = o.objects[self.block][:3]
        g = self.rest[self._goal]
        return (not o.holding
                and float(np.linalg.norm(p[:2] - g[:2])) < self.task.success_xy
                and abs(float(p[2] - g[2])) < self.task.success_z)

    def _vec(self, o):
        return np.concatenate([
            o.q, [o.gripper], o.tool_pose, o.objects[self.block],
            [1.0 if o.holding else 0.0], self.rest[self._goal],
        ]).astype(np.float32)
