"""The command line every Isaac Lab entry point shares, and how it becomes a config.

    ap = argparse.ArgumentParser(); add_args(ap); args = ap.parse_args()
    app = launch(args)                      # before anything else from Isaac
    tid, cfg = make_env_cfg(args)           # the registered task's config, overridden
    servers = start_servers(cfg)            # the planner servers that config needs
    env = gymnasium.make(tid, cfg=cfg)

AppLauncher adds its own flags (--headless, --device, --enable_cameras, ...).
Choices come from the registries -- rig.ROBOTS, scenes, lab.tasks.TASKS -- so a
new robot, scene or task shows up here without an edit. A flag left out keeps
what the task's config says (grasp's scene and its mapping off, say); --set
reaches any other field of it: --set task.bank=eval cell.map_every=4.

Every entry point starts the simulator through launch(), with these flags or
its own: it is where the things every Kit process needs are done once.
"""

import argparse
import ast
import signal

from isaaclab.app import AppLauncher

import scenes
import planner_servers
from rig import DEFAULT_ROBOT, ROBOTS

from . import viz
from .tasks import TASKS, task_id


def add_args(ap, task="PickPlace"):
    ap.add_argument("--task", default=task, choices=sorted(TASKS))
    ap.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
    ap.add_argument("--scene", default=None, choices=scenes.available(),
                    help="an example with a scene.py under scripts/ (default: the task's)")
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--mapping", action=argparse.BooleanOptionalAction, default=None,
                    help="the cameras map and the planner avoids the map (--mapping), or "
                         "the planner sees the scene's static world (--no-mapping); "
                         "default: the task's")
    ap.add_argument("--no-overhead", action="store_true",
                    help="robot-mounted cameras only: leave out the scene's fixed ones")
    ap.add_argument("--map-every", type=int, default=None,
                    help="fuse a frame every N sim steps")
    ap.add_argument("--depth-lag", type=int, default=None,
                    help="sim steps the depth image trails the physics by")
    ap.add_argument("--markers", choices=("overlay", "usd", "off"), default=None)
    ap.add_argument("--camera-class", choices=("camera", "tiled"), default=None)
    ap.add_argument("--planner", choices=("server", "batch", "none"), default=None,
                    help="'batch': each server plans all its envs at once (mapping off); "
                         "'none': build and step the cell without a planner server")
    ap.add_argument("--num-servers", type=int, default=None,
                    help="planner servers to use (default: one per env; with "
                         "--no-mapping fewer may be shared)")
    ap.add_argument("--port", type=int, default=None,
                    help="the first planner server's port (default rig.PORT)")
    ap.add_argument("--set", nargs="+", default=[], metavar="FIELD=VALUE",
                    help="any env config field, dotted: task.bank=eval cell.map_every=4")
    ap.add_argument("--replicate-physics", action="store_true",
                    help="clone env_0's physics to the others instead of parsing each")
    ap.add_argument("--merge-inertial", action="store_true",
                    help="also merge fixed links WITH mass (Isaac Lab's default import)")
    ap.add_argument("--force-usd", action="store_true",
                    help="reconvert the URDF even if the cached USD looks current")
    viz.add_args(ap)
    AppLauncher.add_app_launcher_args(ap)


def launch(args, cameras=None):
    """Start the simulator; returns the app. args: AppLauncher's, as a namespace or a dict.

    cameras: True or False to say; None leaves --enable_cameras as it is,
    except that with add_args()' flags they are on unless --no-mapping: mapping
    needs cameras, cameras need rendering, and which the task wants is only
    known once its config can be imported, after this. A cell that maps nothing
    builds no cameras, and then nothing renders anyway. Also, before Kit starts,
    viser for --viz (viz.preload); and after, Ctrl-C back to Python:
    SimulationApp takes it and exits on the spot, which skips an entry point's
    `finally` -- and with it stopping the planner servers, which run in their
    own session and never see the terminal's Ctrl-C.
    """
    if isinstance(args, dict):
        if cameras is not None:
            args["enable_cameras"] = cameras
    else:
        if cameras is None and hasattr(args, "mapping"):
            cameras = args.mapping is not False or args.enable_cameras
        if cameras is not None:
            args.enable_cameras = cameras
        viz.preload(args)
    app = AppLauncher(args).app
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return app


def make_env_cfg(args):
    """(gym id, env config) for the args, from the registry like parse_env_cfg."""
    import gymnasium as gym

    tid = task_id(args.task, args.robot)
    cfg = gym.spec(tid).kwargs["env_cfg_entry_point"]()
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.scene.replicate_physics = args.replicate_physics
    apply_cell_args(cfg.cell, args)
    apply_overrides(cfg, args.set)
    return tid, cfg


def apply_cell_args(cell, args):
    cell.robot = args.robot
    if args.no_overhead:
        cell.overhead = False
    if args.merge_inertial:
        cell.merge_inertial = True
    if args.force_usd:
        cell.force_usd = True
    for flag, field in (("scene", "scene"), ("mapping", "mapping"),
                        ("map_every", "map_every"), ("depth_lag", "depth_lag"),
                        ("markers", "markers"), ("camera_class", "camera_class"),
                        ("planner", "planner_mode"), ("num_servers", "num_servers"),
                        ("port", "port")):
        value = getattr(args, flag)
        if value is not None:
            setattr(cell, field, value)
    return cell


def apply_overrides(cfg, sets):
    """--set: "a.b.c=value" onto cfg.a.b.c; value as a Python literal, else a string."""
    for item in sets:
        path, _, text = item.partition("=")
        if not _:
            raise SystemExit(f"--set {item!r}: want FIELD=VALUE")
        *parents, leaf = path.split(".")
        obj = cfg
        for name in parents:
            obj = getattr(obj, name)
        if not hasattr(obj, leaf):
            raise SystemExit(f"--set {item!r}: the config has no {path!r}")
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = text
        setattr(obj, leaf, value)


def server_flags(cfg):
    """(how many planner servers, their flags) for an env config.

    Mapping on: one per environment, since a server holds one cell's map.
    Off: cell.num_servers of them (0: one per environment), each seeing the
    static scene -- plus --batch, sized to its share, for planner_mode "batch".
    """
    cell, n = cfg.cell, cfg.scene.num_envs
    k = n if cell.mapping else (cell.num_servers or n)
    flags = ["--scene", cell.scene, "--robot", cell.robot]
    if not cell.mapping:
        flags.append("--no-mapping")
    if cell.planner_mode == "batch":
        flags += ["--batch", str(-(-n // k))]
    return k, flags


def start_servers(cfg, log=print, logfile=None):
    """The planner servers an env config needs, started and listening (planner_servers.launch)."""
    if cfg.cell.planner_mode == "none":
        return None
    k, flags = server_flags(cfg)
    return planner_servers.launch(k, *flags, port=cfg.cell.port, log=log, logfile=logfile)
