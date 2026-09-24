"""Pick-and-place as a gymnasium environment: the policy picks points, cuRobo moves.

One step is one LEG -- a pick if the gripper is empty, a place if it is
holding something. The action says where the tool should be when the gripper
acts; everything between here and there is sim.sim_env.SimEnv's job:

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

import os
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import legs  # noqa: E402
from pick_place import task as pick_place_task  # noqa: E402
from pick_place.task import OBS_DIM, Scorer, TaskCfg  # noqa: E402,F401
from sim import sim_env  # noqa: E402
from sim.sim_env import EnvCfg, ResetOptions  # noqa: E402


class PickPlaceEnv(gym.Env):
    """Shuttle the scene's payload to the other target, one leg per step."""

    metadata = {"render_modes": []}

    # Action bounds: the cell in front of the arm, both pedestals included.
    LOW = pick_place_task.LOW
    HIGH = pick_place_task.HIGH

    def __init__(self, env_cfg: EnvCfg = None, task: TaskCfg = None, headless=True):
        super().__init__()
        self.task = task or TaskCfg()
        sim_env.launch(headless=headless)
        self.sim = sim_env.SimEnv(env_cfg or EnvCfg())
        # Goals, rewards and the observation vector: pick_place.task, shared
        # with the Isaac Lab environment so both score an episode the same way.
        self.scorer = Scorer(self.sim.scene, self.task)
        self.block = self.scorer.block
        self.descend = self.scorer.descend
        self.lift = self.scorer.lift
        self.rest = self.scorer.rest

        self.action_space = gym.spaces.Box(self.LOW, self.HIGH, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

    # --- gymnasium ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        start = int(options.get("block_on", self.np_random.integers(2)))
        o = self.sim.reset(ResetOptions(
            block_on=start, slab_pose=options.get("slab_pose"),
            clear_map=self.task.clear_map, scan=self.task.scan))
        self.scorer.begin_one(start, o)
        return self.scorer.vec_one(o), {"start": start, "goal": int(self.scorer.goal[0])}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), self.LOW, self.HIGH)
        pose = self.tool_pose(*a)
        before = self.sim.observe()
        leg = "place" if before.holding else "pick"
        info = {"leg": leg, "pose": pose}

        # A pick needs an open hand. After a pick that closed on nothing the
        # gripper is still shut, so open it here first, where the arm stands.
        if leg == "pick" and self.sim.gripper_closed:
            self.sim.grip(close=False)

        ok, why = self._leg(pose, close=(leg == "pick"), info=info)
        if not ok:
            info["failed"] = why

        o = self.sim.observe()
        s = self.scorer.score_one(o, failed=not ok, running=self.sim.running)
        info.update(success=s["success"], dropped=s["dropped"], holding=o.holding,
                    block_to_goal=s["block_to_goal"], legs=s["legs"])
        return (self.scorer.vec_one(o), float(s["reward"]), s["terminated"],
                s["truncated"], info)

    def close(self):
        self.sim.close()

    # --- the task ------------------------------------------------------------

    tool_pose = staticmethod(legs.tool_pose)

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
