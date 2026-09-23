"""The planner, as the cell sees it: per environment, never per socket.

    pool = make_pool(cell_cfg, num_envs, log)
    pool.plan(env_ids, q, targets)      -> [reply header per env]
    pool.ik(env_ids, q, targets)        -> [joint list, or None, per env]
    pool.map_frame(env, q, depth, K, camera)
    pool.reset_map(env_ids)

The cell only ever talks to this interface, so how many servers there are and
whether a request is batched is decided here and nowhere else. numpy and the
socket client only -- planning itself stays in planner_server.py.

    ServerPool   planner_server.py processes, one per environment (one map
                 each), or fewer shared ones when mapping is off
    NullPool     no planner; every plan and IK fails. For tools that only
                 need the stage.

A single server that plans for every environment at once (cuRobo's batch
planner, a map per environment) would be one more class here, chosen by
CellCfg.planner_mode, and nothing above this file would change.
"""

from concurrent.futures import ThreadPoolExecutor
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


class ServerPool(PlannerPool):
    """planner_server.py processes on consecutive ports: env e -> server e % k.

    One server holds one map, so with mapping on every environment needs its
    own (k = num_envs: scripts/planner_servers.py starts them). With mapping
    off the world is the static scene for everybody, and k may be smaller --
    down to one server for all.

    Requests made on the same tick go out to their servers together and are
    collected together, so N environments plan in parallel rather than one
    after another.
    """

    def __init__(self, cell, num_envs, log=print):
        k = cell.num_servers or num_envs
        if cell.mapping and k != num_envs:
            raise ValueError(
                f"with mapping on every environment needs its own server (its own "
                f"map): num_servers is {k}, num_envs is {num_envs}")
        if k > num_envs:
            raise ValueError(f"num_servers {k} is more than num_envs {num_envs}")
        self.clients = [Planner(cell.scene, host=cell.host, port=cell.port + i,
                                log=log) for i in range(k)]
        self._threads = ThreadPoolExecutor(max_workers=k) if k > 1 else None

    def _client(self, env):
        return self.clients[env % len(self.clients)]

    def _each(self, env_ids, call):
        """call(client, k) for every request; concurrently across servers.

        Requests for the same server stay in order on its one connection.
        """
        by_client = {}
        for k, e in enumerate(env_ids):
            by_client.setdefault(int(e) % len(self.clients), []).append(k)
        out = [None] * len(env_ids)

        def run(ci, ks):
            for k in ks:
                out[k] = call(self.clients[ci], k)

        if self._threads is None or len(by_client) == 1:
            for ci, ks in by_client.items():
                run(ci, ks)
        else:
            for f in [self._threads.submit(run, ci, ks) for ci, ks in by_client.items()]:
                f.result()
        return out

    def joint_names(self, home, target):
        # The reply to any plan carries the joint order, success or not.
        if target is None:
            raise ValueError("the scene has no targets to probe the planner with")
        return list(self.clients[0].plan(home, list(target))["joint_names"])

    def plan(self, env_ids, q, targets):
        return self._each(env_ids, lambda c, k: c.plan(q[k], [float(v) for v in targets[k]]))

    def ik(self, env_ids, q, targets):
        return self._each(env_ids, lambda c, k: c.ik(q[k], [float(v) for v in targets[k]]))

    def map_frame(self, env, q, depth, K, camera):
        self._client(env).map_frame(q, depth, K, camera)

    def reset_map(self, env_ids):
        for ci in sorted({int(e) % len(self.clients) for e in env_ids}):
            self.clients[ci].reset_map()

    def stats(self, env):
        return self._client(env).stats()


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
    "server": ServerPool,
    "none": NullPool,
}


def make_pool(cell, num_envs, log=print) -> PlannerPool:
    if cell.planner_mode not in POOLS:
        raise ValueError(f"unknown planner_mode {cell.planner_mode!r}; "
                         f"one of {', '.join(POOLS)}")
    return POOLS[cell.planner_mode](cell, num_envs, log)
