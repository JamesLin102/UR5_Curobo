"""CellEnv: the DirectRLEnv every task here is built on.

It owns the parts that are the same whatever the task: the cell is spawned
from cfg.cell.robot / cfg.cell.scene, the planner is connected before the
stage is built (so a missing server fails in seconds), and one env step runs
one PROGRAM per environment -- a whole macro action, hundreds of physics
steps -- in lockstep: an environment that finishes early holds still until
the slowest one is done.

A task subclasses CellEnv and supplies:

    programs(actions)      -> one list of cell_api ops per environment
    reset_cells(env_ids, options)
    _get_dones / _get_rewards / _get_observations, from self.last (the
                           ProgramResults of this step's programs)

and, each step, in self.extras, what lab/eval.py counts:

    "success"   [bool] per environment: did this step achieve the task
    "outcome"   [str] per environment: what the step came to, in a word or two

Optionally, for lab.viz (--viz, the videos):

    viz_views(e)           -> the camera views to show for env e, or None: the
                           cameras' current ones
    viz_draw(viewer, e)    draws the "perception" layer; returns status lines

Used on its own, CellEnv is a cell with no task: actions are ignored and each
step just holds for one tick. That is what the checks under tools/ use.

Reset options: gymnasium's reset(options=...) is ignored by DirectRLEnv, so
it is kept here and handed to reset_cells(); `next_reset_options` does the
same for the auto-reset that step() performs on a finished environment. An
option may be one value for every environment or a list indexed by env id.
"""

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

import scenes
from cell_api import ResetOptions
from rig import SIM_DT

from ..cell import LabCell
from ..cell_cfg import CellCfg
from ..planner_pool import make_pool
from ..scene_cfg import spawn_cell


@configclass
class CellEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=SIM_DT, render_interval=1)
    # One env step is one program: the physics steps inside it are the
    # cell's, so the env itself steps once per action, and holds.
    decimation: int = 1
    episode_length_s: float = SIM_DT      # unused: tasks count their own steps
    # Env spacing leaves the neighbouring cells outside each cell's mapper
    # grid (pick_place: x -0.55..1.25, y +/-0.90), so no camera maps its
    # neighbour.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=False, filter_collisions=True)
    action_space = 1
    observation_space = 1
    state_space = 0
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 1.6, 1.4), lookat=(0.35, 0.0, 0.35))
    ui_window_class_type = None
    wait_for_textures = False
    cell: CellCfg = CellCfg()


class CellEnv(DirectRLEnv):
    cfg: CellEnvCfg

    def __init__(self, cfg: CellEnvCfg, render_mode=None, **kwargs):
        # Not `spec`: gymnasium.make() sets env.spec to its own EnvSpec.
        self.scene_spec = scenes.load(cfg.cell.scene)
        # Connect first: if the server is not there, fail before the stage is
        # built rather than a minute into it.
        self.pool = make_pool(cfg.cell, cfg.scene.num_envs,
                              log=lambda m: print(f"[lab] {m}", flush=True))
        self._reset_options = {}
        self.next_reset_options = {}
        self.last = []
        super().__init__(cfg, render_mode, **kwargs)
        self.cell = LabCell(self)

    # --- DirectRLEnv -----------------------------------------------------------

    def _setup_scene(self):
        self.handles = spawn_cell(self.scene, self.scene_spec, self.cfg.cell)

    def reset(self, seed=None, options=None):
        self._reset_options = dict(options or {})
        return super().reset(seed=seed, options=options)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        options = {**self.next_reset_options, **self._reset_options}
        self.next_reset_options, self._reset_options = {}, {}
        ids = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
        self.reset_cells(ids, options)

    def _pre_physics_step(self, actions):
        programs = self.programs(actions)
        ids = list(range(self.num_envs))
        self.last = self.cell.run(ids, programs) if programs is not None else []

    def _apply_action(self):
        pass    # the program set every target; they hold

    def _get_observations(self):
        return {"policy": torch.zeros((self.num_envs, 1), device=self.device)}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        false = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return false, false.clone()

    # --- what a task supplies --------------------------------------------------

    def programs(self, actions):
        """One op list per environment, or None to just hold."""
        return None

    def reset_cells(self, env_ids, options):
        self.cell.reset(env_ids, ResetOptions(
            block_on=int(options.get("block_on", 0)), slab_pose=options.get("slab_pose"),
            clear_map=options.get("clear_map", True), scan=options.get("scan", True)))

    # --- helpers for tasks -----------------------------------------------------

    def to_torch(self, x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)
