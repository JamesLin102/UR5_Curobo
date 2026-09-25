"""Pick-and-place on Isaac Lab: the policy picks points, cuRobo moves.

The legs, rewards and observation are pick_place.task's. One
step is one LEG per environment: a pick if its gripper is empty, a place if
it is holding something.

    action  (num_envs, 4) [x, y, z, yaw], in each cell's own (env-local)
            coordinates -- the scene's; or in
            [-1, 1] with cfg.action_normalized, for RL libraries whose
            policies output roughly N(0, 1)
    obs     {"policy": (num_envs, 25)}, see pick_place.task.OBS_DIM
    extras  {"leg": [per-env dict: leg, failed?, solve_ms, clearance, ...]}

Needs planner_server.py listening with the same --scene (and --no-mapping if
cfg.cell.mapping is False).
"""

from dataclasses import fields

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils import configclass

from cell_api import ResetOptions
from lab.tasks.base import CellEnv, CellEnvCfg
from legs import leg_info, leg_ops, tool_pose
from pick_place.task import HIGH, LOW, OBS_DIM, Scorer, TaskCfg


def _mirror(dc):
    """A configclass with a dataclass's fields and defaults, kept in step with it.

    The task's settings are defined once, in pick_place.task.TaskCfg; this is
    that, in the form Isaac Lab's config tooling (Hydra overrides) accepts.
    """
    ns = {"__annotations__": {f.name: f.type for f in fields(dc)}}
    ns.update({f.name: f.default for f in fields(dc)})
    return configclass(type(f"Lab{dc.__name__}", (), ns))


LabTaskCfg = _mirror(TaskCfg)


@configclass
class PickPlaceEnvCfg(CellEnvCfg):
    action_space = gym.spaces.Box(LOW, HIGH, dtype=np.float32)
    observation_space = OBS_DIM
    task: LabTaskCfg = LabTaskCfg()
    action_normalized: bool = False

    def __post_init__(self):
        if self.action_normalized:
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)


class PickPlaceEnv(CellEnv):
    cfg: PickPlaceEnvCfg

    def __init__(self, cfg: PickPlaceEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.task = TaskCfg(**{f.name: getattr(cfg.task, f.name) for f in fields(TaskCfg)})
        self.scorer = Scorer(self.scene_spec, self.task, self.num_envs)
        self._rng = np.random.default_rng(cfg.seed)
        self._reward = torch.zeros(self.num_envs, device=self.device)

    @property
    def max_episode_length(self):
        return self.task.max_legs

    # --- actions -> programs ---------------------------------------------------

    def _actions(self, actions):
        a = actions.detach().cpu().numpy().astype(np.float64).reshape(self.num_envs, -1)
        if self.cfg.action_normalized:
            a = LOW + (np.clip(a, -1.0, 1.0) + 1.0) / 2.0 * (HIGH - LOW)
        return np.clip(a, LOW, HIGH)

    def programs(self, actions):
        # "reset" in extras means an auto-reset happened during THIS step.
        self.extras.pop("reset", None)
        before = self.cell.observe(range(self.num_envs))
        self._info = []
        out = []
        for e, (x, y, z, yaw) in enumerate(self._actions(actions)):
            pose = tool_pose(x, y, z, yaw)
            pick = not bool(before.holding[e])
            self._info.append({"leg": "pick" if pick else "place", "pose": pose})
            out.append(leg_ops(pose, close=pick, open_first=pick and bool(self.cell.closed[e]),
                               descend=self.scorer.descend, lift=self.scorer.lift))
        return out

    # --- scoring -----------------------------------------------------------------

    def _get_dones(self):
        ids = list(range(self.num_envs))
        o = self.cell.observe(ids)
        failed = np.array([not r.ok for r in self.last], dtype=bool)
        s = self.scorer.score(ids, o.objects[self.scorer.block][:, :3], o.holding, failed,
                              running=self.cell.running)
        for e, (info, r) in enumerate(zip(self._info, self.last)):
            info.update(leg_info(r))
            if not r.ok:
                info["failed"] = r.why
            info.update(success=bool(s["success"][e]), dropped=bool(s["dropped"][e]),
                        holding=bool(o.holding[e]), block_to_goal=float(s["block_to_goal"][e]),
                        legs=int(s["legs"][e]),
                        block=o.objects[self.scorer.block][e, :3].tolist())
        self.extras["leg"] = self._info
        self.extras["success"] = [bool(i["success"]) for i in self._info]
        self.extras["outcome"] = ["failed" if "failed" in i else i["leg"] for i in self._info]
        self._reward = self.to_torch(s["reward"])
        return (self.to_torch(s["terminated"], torch.bool),
                self.to_torch(s["truncated"], torch.bool))

    def _get_rewards(self):
        return self._reward

    def _get_observations(self):
        ids = list(range(self.num_envs))
        o = self.cell.observe(ids)
        vec = self.scorer.vec(ids, o.q, o.gripper, o.tool_pose,
                              o.objects[self.scorer.block], o.holding)
        return {"policy": self.to_torch(vec)}

    # --- resets ------------------------------------------------------------------

    def reset_cells(self, env_ids, options):
        if "block_on" in options:
            # One target for all, or one per environment (indexed by env id).
            v = options["block_on"]
            start = np.array([int(v[e]) if np.ndim(v) else int(v) for e in env_ids])
        else:
            start = self._rng.integers(len(self.scene_spec.targets), size=len(env_ids))
        o = self.cell.reset(env_ids, [ResetOptions(
            block_on=int(s), slab_pose=options.get("slab_pose"),
            clear_map=self.task.clear_map, scan=self.task.scan) for s in start])
        self.scorer.begin(env_ids, start, o.objects[self.scorer.block][:, :3])
        self.extras["reset"] = {"env_ids": list(env_ids), "start": start.tolist(),
                                "goal": self.scorer.goal[env_ids].tolist()}
