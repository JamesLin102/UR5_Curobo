"""Task registry: every task, for every robot in rig.ROBOTS, as a gym id.

    Isaac-{Task}-{Robot}-v0        e.g. Isaac-PickPlace-Ur5Robotiq-v0

Adding a robot to rig.ROBOTS registers it for every task; adding a task is an
example's lab_env.py (a CellEnv subclass, see lab/tasks/base.py) plus an entry
in TASKS. The scene is not part of the id -- it is `cell.scene` in the env
config, which the task's config sets.

An entry names, as "module:attr" strings:

    env       the CellEnv subclass
    cfg       its config class
    rsl_rl    optional: its PPO runner config, which makes it trainable by
              lab/train.py and loadable by lab/eval.py
    oracle    optional: a function env -> actions tensor, a policy that knows
              what the task hides, for --policy oracle and the checks

They go to gymnasium's registry as the entry point and the kwargs Isaac Lab's
own tools read ("env_cfg_entry_point", "rsl_rl_cfg_entry_point"), plus
"oracle_entry_point".

Nothing here imports Isaac: the entry points are strings, and each robot's
config is built by a factory that Isaac Lab calls (parse_env_cfg) only once
the simulator is running.
"""

import importlib

import gymnasium as gym

from rig import ROBOTS

# Strings, so the backend never imports an example; gymnasium imports them on
# make(), and the train and eval scripts when they need them.
TASKS = {
    "PickPlace": dict(env="pick_place.lab_env:PickPlaceEnv",
                      cfg="pick_place.lab_env:PickPlaceEnvCfg"),
    "Grasp": dict(env="grasp.lab_env:GraspEnv",
                  cfg="grasp.lab_env:GraspEnvCfg",
                  rsl_rl="grasp.lab_rl_cfg:GraspPPORunnerCfg",
                  oracle="grasp.lab_policy:oracle_actions"),
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
        cfg = load(cfg_entry)()
        cfg.cell.robot = robot
        return cfg

    make_cfg.__name__ = make_cfg.__qualname__ = f"make_{robot}_cfg"
    return make_cfg


def load(entry):
    """The object a "module:attr" string names."""
    mod, attr = entry.split(":")
    return getattr(importlib.import_module(mod), attr)


def entry(tid, key):
    """A registered id's "rsl_rl_cfg_entry_point" / "oracle_entry_point", or None."""
    return gym.spec(tid).kwargs.get(key)


def register():
    for task, e in TASKS.items():
        for robot in ROBOTS:
            tid = task_id(task, robot)
            if tid in gym.registry:
                continue
            kwargs = {"env_cfg_entry_point": _cfg_factory(e["cfg"], robot)}
            if "rsl_rl" in e:
                kwargs["rsl_rl_cfg_entry_point"] = e["rsl_rl"]
            if "oracle" in e:
                kwargs["oracle_entry_point"] = e["oracle"]
            gym.register(id=tid, entry_point=e["env"], disable_env_checker=True, kwargs=kwargs)


register()
