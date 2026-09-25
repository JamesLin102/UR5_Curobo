"""The planner, as the cell sees it: per environment, never per socket.

    pool = make_pool(cell_cfg, num_envs, log)
    pool.plan(env_ids, q, targets)      -> [reply header per env]
    pool.plan_joint(env_ids, q, goals)  -> [reply header per env]
    pool.ik(env_ids, q, targets, check) -> [(joint list or None, reason), per env]
    pool.set_world(env_ids, bodies)     -> tell the planner each env's bodies
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
from typing import List, Sequence

import numpy as np

from planner_client import Planner
from rig import ROBOTS


class PlannerPool:
    """The interface. Arrays are per request: row k belongs to env_ids[k]."""

    def joint_names(self, home, target) -> List[str]:
        """The planner's arm joint order, which the cell maps onto the simulator's."""
        raise NotImplementedError

    def plan(self, env_ids: Sequence[int], q: np.ndarray, targets: np.ndarray,
             then=None) -> List[dict]:
        """One reply header per env; with "traj" (n x dof float32) when "ok".

        then[k]: straight moves to solve with it (MoveTo.then); the reply's
        "then" has their joints."""
        raise NotImplementedError

    def plan_joint(self, env_ids, q, goals) -> List[dict]:
        raise NotImplementedError

    def ik(self, env_ids, q, targets, check=None) -> List[tuple]:
        """(joints or None, reason) per env; check[k]: collision-check the straight move."""
        raise NotImplementedError

    def set_world(self, env_ids, bodies) -> None:
        """bodies[k]: [(name, shape, dims, pose)] for env_ids[k], env-local.

        From then on that env plans in a world of the static scene plus these
        (training mode; refused when mapping)."""
        raise NotImplementedError

    def set_exclude(self, env_ids, boxes) -> None:
        """boxes[k]: ((lo), (hi)) the straight-move check leaves out of env_ids[k]'s map."""
        pass

    def map_frame(self, env: int, q, depth, K, camera: str) -> None:
        raise NotImplementedError

    def reset_map(self, env_ids) -> None:
        raise NotImplementedError

    def stats(self, env: int) -> dict:
        return {}

    def voxels(self, env: int):
        """(centres, size) of env's map as the planner has it, or None: no map."""
        return None


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
        self.worlds = set()          # envs whose world was set: their requests name it
        self.exclude = {}            # env -> box the map check leaves out
        k = cell.num_servers or num_envs
        if cell.mapping and k != num_envs:
            raise ValueError(
                f"with mapping on every environment needs its own server (its own "
                f"map): num_servers is {k}, num_envs is {num_envs}")
        if k > num_envs:
            raise ValueError(f"num_servers {k} is more than num_envs {num_envs}")
        self.clients = [Planner(cell.scene, host=cell.host, port=cell.port + i,
                                log=log, mapping=cell.mapping) for i in range(k)]
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

    def _world(self, env):
        return int(env) if int(env) in self.worlds else None

    def plan(self, env_ids, q, targets, then=None):
        then = then if then is not None else [()] * len(env_ids)
        return self._each(env_ids, lambda c, k: c.plan(
            q[k], [float(v) for v in targets[k]], world=self._world(env_ids[k]),
            then=then[k], exclude=self.exclude.get(int(env_ids[k]))))

    def plan_joint(self, env_ids, q, goals):
        return self._each(env_ids, lambda c, k: c.plan_joint(
            q[k], [float(v) for v in goals[k]], world=self._world(env_ids[k])))

    def ik(self, env_ids, q, targets, check=None):
        check = check if check is not None else [False] * len(env_ids)
        return self._each(env_ids, lambda c, k: c.ik_checked(
            q[k], [float(v) for v in targets[k]], world=self._world(env_ids[k]),
            check=check[k], exclude=self.exclude.get(int(env_ids[k]))))

    def set_exclude(self, env_ids, boxes):
        for e, b in zip(env_ids, boxes):
            self.exclude[int(e)] = None if b is None else [list(map(float, b[0])), list(map(float, b[1]))]

    def set_world(self, env_ids, bodies):
        self._each(env_ids, lambda c, k: c.set_world(int(env_ids[k]), bodies[k]))
        self.worlds |= {int(e) for e in env_ids}

    def map_frame(self, env, q, depth, K, camera):
        self._client(env).map_frame(q, depth, K, camera)

    def reset_map(self, env_ids):
        for ci in sorted({int(e) % len(self.clients) for e in env_ids}):
            self.clients[ci].reset_map()

    def stats(self, env):
        return self._client(env).stats()

    def voxels(self, env):
        return self._client(env).voxels()


class NullPool(PlannerPool):
    """No planner. The arm joint order comes from rig.ROBOTS instead."""

    def __init__(self, cell, num_envs, log=print):
        self.robot = cell.robot

    def joint_names(self, home, target):
        return list(ROBOTS[self.robot]["arm"]["joints"])

    def plan(self, env_ids, q, targets, then=None):
        return [{"ok": False, "status": "no planner (planner_mode 'none')"} for _ in env_ids]

    def plan_joint(self, env_ids, q, goals):
        return self.plan(env_ids, q, goals)

    def ik(self, env_ids, q, targets, check=None):
        return [(None, "no planner") for _ in env_ids]

    def set_world(self, env_ids, bodies):
        pass

    def map_frame(self, env, q, depth, K, camera):
        pass

    def reset_map(self, env_ids):
        pass


class BatchPool(ServerPool):
    """ServerPool on servers started with --batch: an env step's plans go out
    as ONE batch per server (plan_batch), not one request per environment.

    Env e is on server e % k, in slot e // k of its batch; each server's
    --batch must be at least ceil(num_envs / k). Training mode only: every
    environment's world must have been set (set_world). Everything else --
    straight moves not solved with a plan, plans home -- is as ServerPool.
    """

    def _slot(self, env):
        return int(env) // len(self.clients)

    def set_world(self, env_ids, bodies):
        self._each(env_ids, lambda c, k: c.set_world(int(env_ids[k]), bodies[k],
                                                     slot=self._slot(env_ids[k])))
        self.worlds |= {int(e) for e in env_ids}

    def plan(self, env_ids, q, targets, then=None):
        then = then if then is not None else [()] * len(env_ids)
        if any(int(e) not in self.worlds for e in env_ids):
            return super().plan(env_ids, q, targets, then)
        by_client = {}
        for k, e in enumerate(env_ids):
            by_client.setdefault(int(e) % len(self.clients), []).append(k)
        out = [None] * len(env_ids)

        def run(ci, ks):
            reqs = [{"slot": self._slot(env_ids[k]), "q": q[k], "target": targets[k],
                     "then": then[k]} for k in ks]
            for k, reply in zip(ks, self.clients[ci].plan_batch(reqs)):
                out[k] = reply

        if self._threads is None or len(by_client) == 1:
            for ci, ks in by_client.items():
                run(ci, ks)
        else:
            for f in [self._threads.submit(run, ci, ks) for ci, ks in by_client.items()]:
                f.result()
        return out


POOLS = {
    "server": ServerPool,
    "batch": BatchPool,
    "none": NullPool,
}


def make_pool(cell, num_envs, log=print) -> PlannerPool:
    if cell.planner_mode not in POOLS:
        raise ValueError(f"unknown planner_mode {cell.planner_mode!r}; "
                         f"one of {', '.join(POOLS)}")
    return POOLS[cell.planner_mode](cell, num_envs, log)
