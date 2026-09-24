"""The Isaac Lab backend: the same cell as sim/sim_env.py, built on Isaac Lab.

    lab/robots.py      rig.ROBOTS[key]      -> ArticulationCfg, for any robot
    lab/scene_cfg.py   scenes.SceneSpec     -> spawned cell under /World/envs/env_*
    lab/cell.py        LabCell: the primitives, over N environments at once;
                       CellView: one of them as a cell_api.CellLike
    lab/programs.py    cell_api ops interpreted one physics tick at a time
    lab/tasks/         CellEnv, the DirectRLEnv every task is built on, and the
                       registry of tasks (each an example's lab_env.py) per robot

cuRobo stays in planner_server.py, reached over the socket, for the same reason
as on the Isaac Sim side: Isaac Lab runs on Isaac Sim 5.1 and its Warp 1.8.2,
and cuRobo 0.8 needs Warp >= 1.13. Nothing under lab/ imports cuRobo.

Importing this package only registers the gym ids, which needs gymnasium and
rig and nothing from Isaac. Everything else here must be imported after
isaaclab.app.AppLauncher has started the simulator.
"""

from . import tasks  # noqa: F401  (registers the gym ids)
