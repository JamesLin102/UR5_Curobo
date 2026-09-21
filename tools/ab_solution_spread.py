"""Measure how far the planner's solutions wander, over a long target cycle.

Headless A/B harness: chains plans between the two demo targets exactly as the
Isaac Sim loop does (each plan starts where the last one ended), then reports
failure rate and how far each joint drifts from its nominal posture. Use it to
compare a config before and after constraining the solution space.

Run:
    python tools/ab_solution_spread.py --config configs/ur5e_robotiq_2f_85.yml \
        --urdf assets/robot/ur_description/ur5e_robotiq_2f_85.urdf \
        --tool grasp_frame --cycles 60 --label before
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from scene_def import HOME, OBSTACLES, SIM_DT, TARGETS  # noqa: E402

from curobo.types import ContentPath, GoalToolPose, JointState, Pose  # noqa: E402
from curobo.kinematics import Kinematics, KinematicsCfg  # noqa: E402
from curobo.scene import Cuboid, Scene  # noqa: E402
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: E402
from curobo._src.robot.loader.util import load_robot_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--tool", default="tool0")
    ap.add_argument("--cycles", type=int, default=60)
    ap.add_argument("--label", default="run")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    content = ContentPath(
        robot_config_absolute_path=f"{ROOT}/{args.config}",
        robot_urdf_absolute_path=f"{ROOT}/{args.urdf}",
        robot_asset_absolute_path=f"{ROOT}/assets/robot/ur_description",
    )
    kin = Kinematics(KinematicsCfg.from_content_path(content))
    scene = Scene(cuboid=[Cuboid(name=n, dims=d, pose=p) for n, d, p, _ in OBSTACLES])
    planner = MotionPlanner(
        MotionPlannerCfg.create(
            robot=load_robot_yaml(content),
            scene_model=scene,
            use_cuda_graph=True,
            interpolation_dt=SIM_DT,
            random_seed=args.seed,
        )
    )
    planner.warmup()
    names = list(kin.joint_names)

    q = list(HOME)
    fails, visited, lengths, pose_err = 0, [], [], []
    for i in range(args.cycles):
        target = TARGETS[i % 2]
        js = JointState.from_position(
            torch.tensor([q], device="cuda", dtype=torch.float32), joint_names=names
        )
        res = planner.plan_pose(
            GoalToolPose.from_poses({args.tool: Pose.from_list(target)}), js
        )
        if res is None or not bool(res.success[0]):
            fails += 1
            continue
        traj = res.interpolated_trajectory.position[0, 0]
        n = int(res.interpolated_last_tstep[0])
        n = max(2, min(n, traj.shape[0]))
        lengths.append(n)
        q = traj[n - 1].cpu().numpy().tolist()
        visited.append(np.rad2deg(np.abs(traj[:n].cpu().numpy())).max(axis=0))

        st = kin.compute_kinematics(
            JointState.from_position(
                torch.tensor([q], device="cuda", dtype=torch.float32), joint_names=names
            )
        )
        reached = st.tool_poses[args.tool].position.cpu().numpy()[0]
        pose_err.append(np.linalg.norm(reached - np.array(target[:3])) * 1000)

    visited = np.array(visited) if visited else np.zeros((1, len(names)))
    print(f"\n=== {args.label} ===")
    print(f"  cycles            : {args.cycles}")
    print(f"  failures          : {fails}  ({100 * fails / args.cycles:.1f}%)")
    print(f"  pose error (mm)   : max {max(pose_err) if pose_err else 0:.3f}")
    print(f"  trajectory length : median {int(np.median(lengths)) if lengths else 0} waypoints")
    print(f"  {'joint':<22s} {'max |angle| reached':>20s}")
    for j, name in enumerate(names):
        peak = visited[:, j].max()
        flag = "   <-- beyond 180" if peak > 180.0 else ""
        print(f"  {name:<22s} {peak:17.1f} deg{flag}")
    beyond = int((visited > 180.0).any(axis=1).sum())
    print(f"  cycles touching a wound-up pose (>180 deg on any joint): "
          f"{beyond}/{len(visited)}")


if __name__ == "__main__":
    main()
