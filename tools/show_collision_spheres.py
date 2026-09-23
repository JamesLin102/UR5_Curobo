"""Serve cuRobo's Viser viewer with the robot mesh and its collision spheres.

cuRobo plans against spheres, not meshes, so "does the config still fit the
robot" is a question about spheres. This puts both in a browser at once, which
is the quickest way to see a link that is under-covered or a gripper whose
auto-fitted spheres have drifted from its geometry.

No Isaac Sim, so it costs seconds rather than minutes. Run one per port to
compare two robots side by side in two tabs:

    python tools/show_collision_spheres.py ur5_robotiq
    python tools/show_collision_spheres.py ur5_robotiq --port 8081

Ctrl-C to stop. The per-link breakdown is printed as well, because a radius
that looks odd on screen is easier to chase from the numbers.
"""

import argparse
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import scenes  # noqa: E402
from rig import DEFAULT_ROBOT, ROBOTS  # noqa: E402

from curobo.types import ContentPath, JointState  # noqa: E402
from curobo.viewer import ViserVisualizer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def report(config_path: str) -> None:
    """Per-link sphere counts and radii, straight from the config.

    A NEGATIVE radius is how cuRobo disables a sphere; the planner filters
    those out (`radius > 0`) and so does the viewer. They are counted
    separately here so the total matches what you actually see on screen.
    """
    kin = yaml.safe_load(open(config_path))["robot_cfg"]["kinematics"]
    spheres = kin["collision_spheres"]
    active = disabled = 0
    print(f"  {'link':<28s} {'spheres':>7s}  {'radius (mm)':>14s}")
    for name in kin["collision_link_names"]:
        s = spheres.get(name) or []
        live = [x["radius"] for x in s if x["radius"] > 0]
        dead = len(s) - len(live)
        active += len(live)
        disabled += dead
        if not live:
            note = f"({dead} disabled)" if dead else ""
            print(f"  {name:<28s} {0:>7d}  {note:>14s}")
            continue
        r = sorted(live)
        span = f"{r[0] * 1000:.0f}" if r[0] == r[-1] else f"{r[0] * 1000:.0f}-{r[-1] * 1000:.0f}"
        tail = f"{span}  (+{dead} disabled)" if dead else span
        print(f"  {name:<28s} {len(live):>7d}  {tail:>14s}")
    extra = f"  (+{disabled} disabled, not drawn)" if disabled else ""
    print(f"  {'total':<28s} {active:>7d}{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("robot", nargs="?", default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--scene", default=scenes.DEFAULT, choices=scenes.available(),
                    help="whose home pose to display (default: %(default)s)")
    ap.add_argument("--q", type=float, nargs="+", default=None,
                    help="joint positions in rad, overriding the scene's home pose")
    args = ap.parse_args()

    spec = ROBOTS[args.robot]
    config_path = f"{ROOT}/{spec['config']}"
    q = args.q if args.q is not None else scenes.load(args.scene).home
    # The viewer drives every actuated joint in the URDF, which on a robot with
    # a locked gripper is wider than the cspace. Name the arm joints explicitly
    # and let cuRobo fill the locked ones in at their locked values.
    cspace = yaml.safe_load(open(config_path))["robot_cfg"]["kinematics"]["cspace"]
    arm_joints = list(cspace["joint_names"])

    print(f"[{args.robot}] {os.path.basename(config_path)}")
    report(config_path)

    viz = ViserVisualizer(
        content_path=ContentPath(
            robot_config_absolute_path=config_path,
            robot_urdf_absolute_path=f"{ROOT}/{spec['urdf']}",
            robot_asset_absolute_path=f"{ROOT}/{spec['assets']}",
        ),
        add_robot_to_scene=True,
        visualize_robot_spheres=True,
        # No draggable goal gizmos: this view is for looking at the robot, and
        # they clutter it.
        add_control_frames=False,
        connect_port=args.port,
    )
    # from_position wants a tensor -- it multiplies the positions to derive a
    # zero velocity, which a plain list cannot do.
    if len(q) != len(arm_joints):
        raise SystemExit(
            f"--q has {len(q)} values, {args.robot} has {len(arm_joints)} arm joints")
    viz.set_joint_state(JointState.from_position(
        torch.tensor([q], device="cuda", dtype=torch.float32),
        joint_names=arm_joints))

    print(f"[{args.robot}] http://localhost:{args.port}   (Ctrl-C to stop)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
