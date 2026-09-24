"""The command line every Isaac Lab entry point shares, and how it becomes a config.

    ap = argparse.ArgumentParser(); add_args(ap); args = ap.parse_args()
    app = launch(args)                      # before anything else from Isaac
    tid, cfg = make_env_cfg(args)           # the registered task's config, overridden
    env = gymnasium.make(tid, cfg=cfg)

The flags mirror isaacsim_client.py's, so the two backends are driven the
same way; AppLauncher adds its own (--headless, --device, --enable_cameras, ...).
Choices come from the registries -- rig.ROBOTS, scenes, lab.tasks.TASKS -- so a
new robot, scene or task shows up here without an edit.
"""

from isaaclab.app import AppLauncher

import scenes
from rig import DEFAULT_ROBOT, ROBOTS

from .tasks import TASKS, task_id


def add_args(ap, task="PickPlace"):
    ap.add_argument("--task", default=task, choices=sorted(TASKS))
    ap.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
    ap.add_argument("--scene", default=scenes.DEFAULT, choices=scenes.available(),
                    help="an example with a scene.py under scripts/; must match the server")
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--no-mapping", action="store_true")
    ap.add_argument("--no-overhead", action="store_true",
                    help="robot-mounted cameras only: leave out the scene's fixed ones")
    ap.add_argument("--map-every", type=int, default=None,
                    help="fuse a frame every N sim steps")
    ap.add_argument("--depth-lag", type=int, default=None,
                    help="sim steps the depth image trails the physics by")
    ap.add_argument("--markers", choices=("overlay", "usd", "off"), default=None)
    ap.add_argument("--camera-class", choices=("camera", "tiled"), default=None)
    ap.add_argument("--planner", choices=("server", "none"), default=None,
                    help="'none': build and step the cell without a planner server")
    ap.add_argument("--num-servers", type=int, default=None,
                    help="planner servers to use (default: one per env; with "
                         "--no-mapping fewer may be shared)")
    ap.add_argument("--replicate-physics", action="store_true",
                    help="clone env_0's physics to the others instead of parsing each")
    ap.add_argument("--merge-inertial", action="store_true",
                    help="also merge fixed links WITH mass (Isaac Lab's default import)")
    ap.add_argument("--force-usd", action="store_true",
                    help="reconvert the URDF even if the cached USD looks current")
    AppLauncher.add_app_launcher_args(ap)


def launch(args):
    """Start the simulator. Mapping needs cameras, and cameras need rendering."""
    if not args.no_mapping:
        args.enable_cameras = True
    return AppLauncher(args).app


def make_env_cfg(args):
    """(gym id, env config) for the args, from the registry like parse_env_cfg."""
    import gymnasium as gym

    tid = task_id(args.task, args.robot)
    cfg = gym.spec(tid).kwargs["env_cfg_entry_point"]()
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.scene.replicate_physics = args.replicate_physics
    apply_cell_args(cfg.cell, args)
    return tid, cfg


def apply_cell_args(cell, args):
    cell.robot = args.robot
    cell.scene = args.scene
    cell.mapping = not args.no_mapping
    cell.overhead = not args.no_overhead
    cell.merge_inertial = args.merge_inertial
    cell.force_usd = args.force_usd
    for flag, field in (("map_every", "map_every"), ("depth_lag", "depth_lag"),
                        ("markers", "markers"), ("camera_class", "camera_class"),
                        ("planner", "planner_mode"), ("num_servers", "num_servers")):
        value = getattr(args, flag)
        if value is not None:
            setattr(cell, field, value)
    return cell
