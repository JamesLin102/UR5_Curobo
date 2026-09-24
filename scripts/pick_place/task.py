"""Pick-and-place as a task: goals, rewards, observations. No simulator in here.

Shared by both gym environments -- pick_place.isaacsim_env.PickPlaceEnv (Isaac
Sim, one cell) and pick_place.lab_env.PickPlaceEnv (Isaac Lab, N cells) -- so
the two score the same episode the same way. Standard library and numpy only.

One step is one LEG -- a pick if the gripper is empty, a place if it is
holding something -- built by legs.leg_ops: above, down, grip, up.

Everything that works on many cells at once takes `env_ids` and arrays with a
leading env axis, so the single-cell environment is just N = 1.
"""

import math
from dataclasses import dataclass

import numpy as np

from cell_api import Observation


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


# Action bounds: the cell in front of the arm, both pedestals included.
LOW = np.array([0.25, -0.55, 0.05, -math.pi], dtype=np.float32)
HIGH = np.array([0.65, 0.55, 0.35, math.pi], dtype=np.float32)

# q(6) gripper(1) tool_pose(7) block_pose(7) holding(1) goal(3)
OBS_DIM = 25

class Scorer:
    """Goals, rewards, dones and observation vectors for N cells.

    The scene has to have exactly one payload and two targets: the block
    shuttles from the target it starts under to the other one.
    """

    def __init__(self, scene, task: TaskCfg, num_envs=1):
        if len(scene.payload) != 1 or len(scene.targets) != 2:
            raise ValueError("pick-and-place wants one payload and two targets")
        self.task = task
        self.block = scene.payload[0][0]
        self.descend = scene.pick["descend_m"]
        self.lift = scene.pick["lift_m"]
        # Where the block rests on each target: its spawn height, under that
        # target -- the same rule the cell's reset uses to put it there.
        rest_z = float(scene.payload[0][2][2])
        self.rest = np.array([[t[0], t[1], rest_z] for t in scene.targets],
                             dtype=np.float32)
        self.goal = np.ones(num_envs, dtype=np.int64)
        self.legs = np.zeros(num_envs, dtype=np.int64)
        self.dist = np.zeros(num_envs, dtype=np.float64)

    # --- arrays, many cells ----------------------------------------------------

    def block_to_goal(self, env_ids, block_pos):
        """(k,) distance from each block (k, 3) to its goal's rest position."""
        d = block_pos - self.rest[self.goal[env_ids]]
        return np.sqrt((d.astype(np.float32) ** 2).sum(axis=-1))

    def begin(self, env_ids, start, block_pos):
        """A new episode in each env: the block is under target `start`."""
        env_ids = np.asarray(env_ids)
        self.goal[env_ids] = 1 - np.asarray(start)
        self.legs[env_ids] = 0
        self.dist[env_ids] = self.block_to_goal(env_ids, block_pos)

    def score(self, env_ids, block_pos, holding, failed, running=True):
        """After a leg: reward and dones per env, as a dict of (k,) arrays."""
        t = self.task
        env_ids = np.asarray(env_ids)
        holding = np.asarray(holding, dtype=bool)
        failed = np.asarray(failed, dtype=bool)
        d = self.block_to_goal(env_ids, block_pos)
        g = self.rest[self.goal[env_ids]]
        dxy = block_pos[:, :2] - g[:, :2]
        success = (~holding
                   & (np.sqrt((dxy.astype(np.float32) ** 2).sum(axis=-1)) < t.success_xy)
                   & (np.abs(block_pos[:, 2] - g[:, 2]) < t.success_z))
        dropped = ~holding & (block_pos[:, 2] < t.dropped_z)
        reward = (t.r_step
                  + np.where(failed, t.r_failed_motion, 0.0)
                  + t.r_progress * (self.dist[env_ids] - d)
                  + np.where(success, t.r_success, 0.0)
                  + np.where(dropped, t.r_dropped, 0.0))
        self.dist[env_ids] = d
        self.legs[env_ids] += 1
        terminated = success | dropped | (not running)
        truncated = ~terminated & (self.legs[env_ids] >= t.max_legs)
        return dict(reward=reward, success=success, dropped=dropped,
                    terminated=terminated, truncated=truncated,
                    block_to_goal=d, legs=self.legs[env_ids].copy())

    def vec(self, env_ids, q, gripper, tool, block, holding):
        """(k, OBS_DIM) float32 observation, one row per env."""
        env_ids = np.asarray(env_ids)
        return np.concatenate([
            q, np.asarray(gripper).reshape(-1, 1), tool, block,
            np.asarray(holding, dtype=np.float32).reshape(-1, 1),
            self.rest[self.goal[env_ids]],
        ], axis=1).astype(np.float32)

    # --- one cell, from an Observation ---------------------------------------

    def begin_one(self, start, o: Observation):
        self.begin([0], [start], o.objects[self.block][None, :3])

    def score_one(self, o: Observation, failed, running=True):
        """score() for env 0, with python scalars out."""
        s = self.score([0], o.objects[self.block][None, :3], [o.holding],
                       [failed], running)
        return {k: v[0].item() for k, v in s.items()}

    def vec_one(self, o: Observation):
        return self.vec([0], o.q[None], [o.gripper], o.tool_pose[None],
                        o.objects[self.block][None], [o.holding])[0]
