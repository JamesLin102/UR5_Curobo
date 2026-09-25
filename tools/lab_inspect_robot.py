"""What did Isaac Lab make of the robot? Layout, gains, pins, cameras -- no planner.

    python tools/lab_inspect_robot.py --headless                   # no cameras
    python tools/lab_inspect_robot.py --headless --mapping         # + camera checks
    python tools/lab_inspect_robot.py --headless --merge-inertial  # Isaac Lab's merge
    python tools/lab_inspect_robot.py --headless --robot KEY       # any rig.ROBOTS entry
    python tools/lab_inspect_robot.py --headless --num_envs 4 --replicate-physics

Builds the cell through the task-free lab.tasks.base.CellEnv with no planner
(every plan would fail; nothing here plans), then reports:

  bodies and joints   what the URDF import produced, with masses
  gains               PhysX's drive gains per joint, SI, against rig.ROBOTS x DEG
  frames              which body the tool frame and each camera ended up on
  stage               the linkage pins and the pad material binding
  HOME                how far the arm drifts holding HOME (the Isaac Sim backend
                      measured 4.4 mrad once fixed joints were merged)
  gripper             closing on nothing: the spread across the linkage's
                      joints as a fraction of a full close (0.003 with the
                      pin working, >= 0.118 without it)

With several environments HOME and the gripper are checked in every one --
which is how to tell whether the pins survive --replicate-physics.
  warp                which Warp this process runs -- Isaac Sim's 1.8.2, or
                      the process is set up wrong

Exit status 0 means every check passed.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lab import app as lab_app  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
lab_app.add_args(ap)
ap.add_argument("--mapping", action="store_true", help="build the cameras and check them")
ap.add_argument("--hold-steps", type=int, default=120)
ARGS = ap.parse_args()
ARGS.no_mapping = not ARGS.mapping
ARGS.planner = "none"
APP = lab_app.launch(ARGS)

import numpy as np  # noqa: E402

from rig import ROBOTS  # noqa: E402

FAILED = []


def report(ok, what):
    print(f"[inspect] {'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        FAILED.append(what)


def say(msg):
    print(f"[inspect] {msg}", flush=True)


def main():
    import warp
    from pxr import UsdPhysics, UsdShade

    import isaaclab.sim as sim_utils
    from lab.robots import DEG, drive_groups
    from lab.tasks.base import CellEnv, CellEnvCfg

    cfg = CellEnvCfg()
    cfg.sim.device = ARGS.device
    cfg.scene.num_envs = ARGS.num_envs
    cfg.scene.replicate_physics = ARGS.replicate_physics
    lab_app.apply_cell_args(cfg.cell, ARGS)
    env = CellEnv(cfg)
    cell = env.cell
    robot = cell.robot
    stage = sim_utils.get_current_stage()
    root = f"{env.scene.env_prim_paths[0]}/Robot"

    say(f"warp {warp.__version__} from {os.path.dirname(warp.__file__)}")
    report(warp.__version__.startswith("1.8"), f"warp is Isaac Sim's own ({warp.__version__})")

    masses = robot.root_physx_view.get_masses()[0].cpu().numpy()
    say(f"{robot.num_bodies} bodies (merge_inertial={cfg.cell.merge_inertial}):")
    for name, m in zip(robot.body_names, masses):
        say(f"    {name:40s} {m:8.4f} kg")
    say(f"{robot.num_joints} joints: {', '.join(robot.joint_names)}")

    k_sim = robot.root_physx_view.get_dof_stiffnesses()[0].cpu().numpy()
    d_sim = robot.root_physx_view.get_dof_dampings()[0].cpu().numpy()
    usd = {}
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(root):
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive and drive.GetStiffnessAttr().Get() is not None:
            usd[prim.GetName()] = (drive.GetStiffnessAttr().Get(), drive.GetDampingAttr().Get())
    for group, (joints, k, d) in drive_groups(cfg.cell.robot).items():
        for j in joints:
            i = robot.joint_names.index(j)
            uk, ud = usd.get(j, (float("nan"), float("nan")))
            good = abs(k_sim[i] - k * DEG) < 1e-3 * max(1.0, k * DEG) and \
                abs(d_sim[i] - d * DEG) < 1e-3 * max(1.0, d * DEG)
            report(good, f"{group:9s} {j:40s} PhysX {k_sim[i]:10.2f} / {d_sim[i]:8.2f} SI"
                         f"   USD {uk:8.2f} / {ud:6.2f} per deg   (rig {k:g} / {d:g})")

    body, pos, _ = cell.handles.tool
    say(f"tool frame {ROBOTS[cfg.cell.robot]['tool_frame']}: body {body}, "
        f"offset {np.round(pos, 5).tolist()}")
    for name, (b, T) in cell.handles.camera_mounts.items():
        say(f"camera {name}: body {b}, offset {np.round(T[:3, 3], 5).tolist()}")
    grip = ROBOTS[cfg.cell.robot]["gripper"]
    for link in grip["pad_links"]:
        col = stage.GetPrimAtPath(f"{root}/{link}/collisions")
        mat = None
        if col.IsValid():
            mat, _ = UsdShade.MaterialBindingAPI(col).ComputeBoundMaterial("physics")
        say(f"pad {link}: collisions prim {'found' if col.IsValid() else 'MISSING'}"
            f"{' (instanceable)' if col.IsValid() and col.IsInstance() else ''}, "
            f"physics material {mat.GetPath() if mat else None}")
    if cell.payload:
        report(len(cell.handles.pads) == len(grip["pad_links"]),
               f"grip material on {len(cell.handles.pads)} of {len(grip['pad_links'])} pads")
    if grip.get("linkage"):
        report(len(cell.handles.pins) == 2, f"linkage pins: {cell.handles.pins}")

    from cell_api import Grip

    # HOME drift, holding still, in every environment.
    ids = list(range(cell.num_envs))
    home = np.asarray(cell.spec.home)
    worst = np.zeros(len(ids))
    for _ in range(ARGS.hold_steps):
        cell.idle(ids, 1)
        worst = np.maximum(worst, np.abs(cell.observe(ids).q - home).max(axis=1))
    e = int(worst.argmax())
    report(worst.max() < 0.01, f"holding HOME for {ARGS.hold_steps} steps: worst joint "
                               f"{worst.max() * 1000:.1f} mrad off"
                               + (f" (worst of {len(ids)} envs: env {e})" if len(ids) > 1 else ""))

    # Gripper closing on nothing: does the 4-bar stay together, everywhere?
    results = cell.run(ids, [[Grip(close=True)] for _ in ids])
    for e, r in zip(ids, results):
        g = r.results[0][1]
        frac = np.array([v / grip["closed"] for v in g.spread.values()])
        where = f"env {e}: " if len(ids) > 1 else ""
        if e == 0:
            say("  closed: " + " ".join(f"{n}={v:+.3f}" for n, v in g.spread.items()))
        report(g.settled, f"{where}gripper settled after {g.steps} steps, error {g.error:.4f} rad")
        report(frac.max() - frac.min() < 0.02,
               f"{where}gripper spread {frac.max() - frac.min():.3f} of a full close "
               f"({frac.min():.3f}..{frac.max():.3f}); pinned and working is ~0.003")
    cell.run(ids, [[Grip(close=False)] for _ in ids])

    say(f"{'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.stdout.flush()
    os._exit(1 if FAILED else 0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # A Kit app often hangs on the way out after an exception, holding the
        # GPU. Report and leave.
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
