"""The planner, as the cell sees it: per environment, never per socket.

    pool = make_pool(cell_cfg, num_envs, log)
    pool.plan(env_ids, q, targets)      -> [reply header per env]
    pool.ik(env_ids, q, targets)        -> [joint list, or None, per env]
    pool.map_frame(env, q, depth, K, camera)
    pool.reset_map(env_ids)

The cell only ever talks to this interface, so how many servers there are and
whether a request is batched is decided here and nowhere else. numpy and the
socket client only -- planning itself stays in planner_server.py.

    SinglePool   one planner_server.py: one map, so one environment
    NullPool     no planner; every plan and IK fails. For tools that only
                 need the stage.

A multi-environment server (batched plan/ik, a map per environment) is one
more class here, chosen by CellCfg.planner_mode.
"""

from typing import List, Optional, Sequence

import numpy as np

from planner_client import Planner
from rig import ROBOTS


class PlannerPool:
    """The interface. Arrays are per request: row k belongs to env_ids[k]."""

    def joint_names(self, home, target) -> List[str]:
        """The planner's arm joint order, which the cell maps onto the simulator's."""
        raise NotImplementedError

    def plan(self, env_ids: Sequence[int], q: np.ndarray, targets: np.ndarray) -> List[dict]:
        """One reply header per env; with "traj" (n x dof float32) when "ok"."""
        raise NotImplementedError

    def ik(self, env_ids, q, targets) -> List[Optional[list]]:
        raise NotImplementedError

    def map_frame(self, env: int, q, depth, K, camera: str) -> None:
        raise NotImplementedError

    def reset_map(self, env_ids) -> None:
        raise NotImplementedError

    def stats(self, env: int) -> dict:
        return {}


class SinglePool(PlannerPool):
    """One planner_server.py. It holds one map, so it can serve one environment."""

    def __init__(self, cell, num_envs, log=print):
        if num_envs != 1:
            raise ValueError(
                f"planner_mode 'single' serves one environment, not {num_envs}: the "
                f"server keeps one map, and cells sharing it would map each other")
        self.client = Planner(cell.scene, host=cell.host, port=cell.port, log=log)

    def joint_names(self, home, target):
        # The reply to any plan carries the joint order, success or not.
        if target is None:
            raise ValueError("the scene has no targets to probe the planner with")
        return list(self.client.plan(home, list(target))["joint_names"])

    def plan(self, env_ids, q, targets):
        return [self.client.plan(q[k], [float(v) for v in targets[k]])
                for k in range(len(env_ids))]

    def ik(self, env_ids, q, targets):
        return [self.client.ik(q[k], [float(v) for v in targets[k]])
                for k in range(len(env_ids))]

    def map_frame(self, env, q, depth, K, camera):
        self.client.map_frame(q, depth, K, camera)

    def reset_map(self, env_ids):
        self.client.reset_map()

    def stats(self, env):
        return self.client.stats()


class NullPool(PlannerPool):
    """No planner. The arm joint order comes from rig.ROBOTS instead."""

    def __init__(self, cell, num_envs, log=print):
        self.robot = cell.robot

    def joint_names(self, home, target):
        return list(ROBOTS[self.robot]["arm"]["joints"])

    def plan(self, env_ids, q, targets):
        return [{"ok": False, "status": "no planner (planner_mode 'none')"} for _ in env_ids]

    def ik(self, env_ids, q, targets):
        return [None for _ in env_ids]

    def map_frame(self, env, q, depth, K, camera):
        pass

    def reset_map(self, env_ids):
        pass


POOLS = {
    "single": SinglePool,
    "none": NullPool,
}


def make_pool(cell, num_envs, log=print) -> PlannerPool:
    if cell.planner_mode not in POOLS:
        raise ValueError(f"unknown planner_mode {cell.planner_mode!r}; "
                         f"one of {', '.join(POOLS)}")
    return POOLS[cell.planner_mode](cell, num_envs, log)
