"""Is the Isaac Lab backend still built out of the registries, and nothing else?

    python tools/check_lab_modularity.py            # all of it
    python tools/check_lab_modularity.py --static   # the import rules only, no simulator

Three things, each guarding a promise the layout makes:

  imports     who may import whom, read off the source (ast), so a shortcut
              taken today fails here rather than months later:
                shared layer  (cell_api, urdf_frames, sim_usd, pick_place_task,
                               demo_loop, rig, proto, planner_client, scenes/)
                              imports neither backend, nor cuRobo
                lab/          imports neither the Isaac Sim backend nor cuRobo
                Isaac Sim     (sim_env, isaacsim_client, pick_place_env)
                              imports neither lab/ nor Isaac Lab nor cuRobo
                planner_server imports no simulator at all
  registry    every task is registered for every robot in rig.ROBOTS
  builders    the cell builds, steps and observes, with no planner, for every
              scene under scenes/ AND for a bare scene with only the required
              SceneSpec fields -- no payload, no unmapped body, no camera --
              so nothing in lab/ quietly assumes pick_place's contents

Exit status 0 means every check passed.
"""

import argparse
import ast
import glob
import os
import subprocess
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS)

BARE = "_bare"       # the synthetic scene, registered in the child that builds it

FAILED = []


def report(ok, what):
    print(f"[modularity] {'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        FAILED.append(what)


# --- imports -------------------------------------------------------------------

SHARED = ["cell_api", "urdf_frames", "sim_usd", "pick_place_task", "demo_loop", "rig",
          "proto", "planner_client"]
ISAAC_SIM = ["sim_env", "isaacsim_client", "pick_place_env"]
RULES = [
    # (files, forbidden top-level modules)
    ([f"{m}.py" for m in SHARED] + ["scenes/*.py"],
     {"lab", "curobo", "isaaclab", "isaaclab_tasks"} | set(ISAAC_SIM)),
    (["lab/*.py", "lab/tasks/*.py"], {"curobo"} | set(ISAAC_SIM)),
    ([f"{m}.py" for m in ISAAC_SIM], {"lab", "curobo", "isaaclab", "isaaclab_tasks"}),
    (["planner_server.py"], {"isaacsim", "isaaclab", "omni", "pxr", "lab"} | set(ISAAC_SIM)),
]


def imported(path):
    """Top-level module names a file imports, anywhere in it (lazy imports too)."""
    tree = ast.parse(open(path).read(), path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def check_imports():
    for patterns, forbidden in RULES:
        for pattern in patterns:
            for path in sorted(glob.glob(os.path.join(SCRIPTS, pattern))):
                bad = imported(path) & forbidden
                rel = os.path.relpath(path, ROOT)
                report(not bad, f"{rel} imports {sorted(bad) if bad else 'nothing it must not'}")


def check_registry():
    import gymnasium as gym

    import lab.tasks as tasks
    from rig import ROBOTS

    want = {tasks.task_id(t, r) for t in tasks.TASKS for r in ROBOTS}
    have = {i for i in gym.registry if i in want}
    for tid in sorted(want):
        report(tid in have, f"registered {tid}")


# --- builders (one child process per scene) ------------------------------------


def bare_scene():
    """Only what SceneSpec requires: a floor to stand on and one goal."""
    import scenes
    from scenes import SceneSpec

    home = list(scenes.load(scenes.DEFAULT).home)
    return SceneSpec(
        obstacles=[("table", [1.0, 1.0, 0.10], [0.0, 0.0, -0.06, 1, 0, 0, 0], (0.5, 0.5, 0.5))],
        targets=[[0.4, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0]],
        home=home, scan_poses=[home], cameras={}, mapper={})


def child(scene_name):
    from isaaclab.app import AppLauncher

    import scenes
    if scene_name == BARE:
        mod = types.ModuleType(f"scenes.{BARE}")
        mod.SCENE = bare_scene()
        sys.modules[mod.__name__] = mod
    spec = scenes.load(scene_name)
    AppLauncher(dict(headless=True, enable_cameras=bool(spec.cameras)))

    from lab.tasks.base import CellEnv, CellEnvCfg

    cfg = CellEnvCfg()
    cfg.cell.scene = scene_name
    cfg.cell.planner_mode = "none"
    cfg.cell.mapping = bool(spec.cameras)
    cfg.cell.verbose = False
    env = CellEnv(cfg)
    cell = env.cell
    cell.idle([0], 10)
    o = cell.observe([0], images=bool(cell.cams))
    print(f"[modularity] RESULT {scene_name}: built and stepped; "
          f"{cell.robot.num_bodies} bodies, cameras {sorted(cell.cams)}, "
          f"payload {sorted(cell.payload)}, unmapped {sorted(cell.bodies)}, "
          f"tool at {[round(float(v), 3) for v in o.tool_pose[0, :3]]}", flush=True)
    os._exit(0)


def check_builders():
    import scenes

    for name in scenes.available() + [BARE]:
        try:
            r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--child", name],
                               cwd=ROOT, timeout=900, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
            line = next((ln for ln in r.stdout.splitlines()
                         if ln.startswith("[modularity] RESULT")), None)
            if line:
                print(line.replace("RESULT ", ""), flush=True)
            elif "Traceback" in r.stdout:
                print(r.stdout[r.stdout.rindex("Traceback"):][-3000:], flush=True)
            report(r.returncode == 0 and line is not None, f"cell builds for scene {name!r}")
        except subprocess.TimeoutExpired:
            report(False, f"cell builds for scene {name!r} (timed out)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--static", action="store_true", help="import rules and registry only")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        try:
            child(args.child)
        except BaseException:
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            os._exit(1)

    check_imports()
    check_registry()
    if not args.static:
        check_builders()
    print(f"[modularity] {'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}", flush=True)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
