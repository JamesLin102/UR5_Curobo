"""Task registry: every task, for every robot in rig.ROBOTS, as a gym id.

    Isaac-{Task}-{Robot}-v0        e.g. Isaac-PickPlace-Ur5Robotiq-v0

Adding a robot to rig.ROBOTS registers it for every task; adding a task is an
example's lab_env.py (a CellEnv subclass, see lab/tasks/base.py) plus one line
in TASKS. The scene is not part of the id -- it is `cell.scene` in the env
config, and defaults to scenes.DEFAULT.

Nothing here imports Isaac: the entry points are strings, and each robot's
config is built by a factory that Isaac Lab calls (parse_env_cfg) only once
the simulator is running.
"""

import importlib

import gymnasium as gym

from rig import ROBOTS

# name -> (env class, env config class), as "module:attr". Strings, so the
# backend never imports an example; gymnasium imports it on make().
TASKS = {
    "PickPlace": ("pick_place.lab_env:PickPlaceEnv",
                  "pick_place.lab_env:PickPlaceEnvCfg"),
    "Grasp": ("grasp.lab_env:GraspEnv",
              "grasp.lab_env:GraspEnvCfg"),
}


def camel(key):
    """rig.ROBOTS key -> the robot's part of a gym id: ur5_robotiq -> Ur5Robotiq."""
    return "".join(p[:1].upper() + p[1:] for p in key.split("_"))


def task_id(task, robot):
    return f"Isaac-{task}-{camel(robot)}-v0"


def ids():
    """Every registered id, for --help and error messages."""
    return sorted(task_id(t, r) for t in TASKS for r in ROBOTS)


def _cfg_factory(cfg_entry, robot):
    """A callable that builds the task's config for `robot`.

    A plain function rather than functools.partial: Isaac Lab's registry
    loader calls inspect.getfile() on a callable entry point, which refuses a
    partial.
    """
    def make_cfg():
        mod, attr = cfg_entry.split(":")
        cfg = getattr(importlib.import_module(mod), attr)()
        cfg.cell.robot = robot
        return cfg

    make_cfg.__name__ = make_cfg.__qualname__ = f"make_{robot}_cfg"
    return make_cfg


def register():
    for task, (env_entry, cfg_entry) in TASKS.items():
        for robot in ROBOTS:
            tid = task_id(task, robot)
            if tid in gym.registry:
                continue
            gym.register(
                id=tid,
                entry_point=env_entry,
                disable_env_checker=True,
                kwargs={"env_cfg_entry_point": _cfg_factory(cfg_entry, robot)},
            )


register()
