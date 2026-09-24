"""Solve the grasp scene's HOME and scan poses from its camera views, and check them.

    python tools/grasp_scan_poses.py            # solve, check, print what to paste
    python tools/grasp_scan_poses.py --check    # check the poses grasp/scene.py has

cuRobo only (the planner-server side of the Warp split); no simulator.

Each VIEW in grasp/scene.py is a camera eye and a point to look at. For each
one this finds joint angles that put the wrist camera's OPTICAL frame there,
with collision-aware IK against the table plus NO_GO -- the box any cylinder
could occupy -- so the pose is safe whatever the layout. The roll about the
view axis is free: every 30 degrees is tried, IK is seeded around the
previous pose and returns every configuration it finds, and the one closest to
the previous pose in joint space is kept. Without that the solver's best-cost
answer is as likely to swing the base half a turn round the back as to move
the wrist (measured: 567 deg of joint travel for a view 113 deg away).

Then the check, which is what the scan actually does: the cell blends joint
space LINEARLY from one scan pose to the next (lab/programs._Scan), so the
blend is sampled and every collision sphere of the arm is measured against
NO_GO along it, against the planner's floor guard (SceneSpec.keep_out: the arm
has to start every plan from outside it), and above the table.

Exit status 0 means every pose was solved and every check passed.
"""

import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch  # noqa: E402

from grasp import scene as G  # noqa: E402
from urdf_frames import matrix_to_quat, quat_to_matrix  # noqa: E402

# The arm must stay this far out of NO_GO, sphere surface to box surface.
MARGIN = 0.02
BLEND_SAMPLES = 60
ROLLS_DEG = range(0, 360, 30)
SEED_SPREAD = 0.5          # rad, how far around the previous pose IK is seeded
# Spheres this close to the base's axis are the base's own, which sit inside
# the table on purpose (rig.ROBOTS[...]["arm"]["mask_only_links"]); they are
# left out of the height-above-the-table figure.
BASE_RADIUS = 0.12


def look_at(eye, target, roll_deg):
    """4x4 world pose of an OPTICAL frame at `eye` looking at `target`.

    Optical: +Z along the view, +X image right, +Y image down. `roll_deg`
    turns the image about the view axis; 0 puts image-down along the
    horizontal direction of world -X where that is defined.
    """
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    ref = np.array([-1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = ref - z * (ref @ z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    r = math.radians(roll_deg)
    xr = math.cos(r) * x + math.sin(r) * y
    yr = np.cross(z, xr)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = xr, yr, z, eye
    return T


def to_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(pose[3:])
    T[:3, 3] = pose[:3]
    return T


def to_list(T):
    return list(T[:3, 3]) + list(matrix_to_quat(T[:3, :3]))


def box_distance(centres, radii, dims, pose):
    """Signed distance from each sphere's surface to an axis-aligned box's surface."""
    lo = np.asarray(pose[:3]) - np.asarray(dims) / 2
    hi = np.asarray(pose[:3]) + np.asarray(dims) / 2
    q = np.maximum(lo - centres, 0) + np.maximum(centres - hi, 0)
    outside = np.linalg.norm(q, axis=1)
    inside = np.minimum(np.min(np.minimum(centres - lo, hi - centres), axis=1), 0)
    return np.where(outside > 0, outside, inside) - radii


class Arm:
    """cuRobo kinematics + a collision-aware IK for the grasp scene."""

    def __init__(self):
        import planner_server as ps
        from curobo.scene import Cuboid, Scene
        from curobo.types import GoalToolPose, JointState, Pose

        self.JointState, self.GoalToolPose, self.Pose = JointState, GoalToolPose, Pose
        self.kin, self.planner, _, _, self.tool = ps.build("ur5_robotiq", G.SCENE,
                                                           use_cuda_graph=False)
        world = Scene(cuboid=[Cuboid(name=n, dims=d, pose=p) for n, d, p, _ in
                              list(G.SCENE.obstacles) + list(G.SCENE.keep_out)]
                      + [Cuboid(name="no_go", dims=G.NO_GO[0], pose=G.NO_GO[1])])
        self.planner.update_world(world)
        self.names = list(self.kin.joint_names)
        self.rng = np.random.default_rng(0)

    def fk(self, q):
        js = self.JointState.from_position(torch.tensor([q], device="cuda", dtype=torch.float32),
                                           joint_names=self.names)
        k = self.kin.compute_kinematics(js)
        poses = {}
        for f in ("camera_link", self.tool):
            p = k.tool_poses[f]
            poses[f] = to_matrix(np.concatenate([p.position.view(-1).cpu().numpy(),
                                                 p.quaternion.view(-1).cpu().numpy()]))
        sph = k.robot_spheres.reshape(-1, 4).cpu().numpy()
        return poses, sph[sph[:, 3] > 0]

    def ik(self, tool_T, seed):
        """Every collision-free configuration IK finds for the tool pose, (k, dof)."""
        js = self.JointState.from_position(torch.tensor([seed], device="cuda", dtype=torch.float32),
                                           joint_names=self.names)
        goal = self.GoalToolPose.from_poses({self.tool: self.Pose.from_list(to_list(tool_T))})
        n = self.planner.ik_solver.config.num_seeds
        seeds = np.asarray(seed) + self.rng.normal(0.0, SEED_SPREAD, size=(n, len(seed)))
        seeds[0] = seed
        res = self.planner.ik_solver.solve_pose(
            goal, current_state=js, return_seeds=n,
            seed_config=torch.tensor(seeds[None], device="cuda", dtype=torch.float32))
        if res is None or not bool(res.success.any()):
            return np.zeros((0, len(self.names)))
        return res.solution[res.success].view(-1, len(self.names)).cpu().numpy().astype(float)


def clearance(arm, q):
    """(to NO_GO, to the planner's keep-out guard, the arm's lowest sphere above the table) at q."""
    _, sph = arm.fk(q)
    d = box_distance(sph[:, :3], sph[:, 3], *G.NO_GO)
    arm_only = sph[np.linalg.norm(sph[:, :2], axis=1) > BASE_RADIUS]
    guard = min((float(box_distance(arm_only[:, :3], arm_only[:, 3], dims, pose).min())
                 for _, dims, pose, _ in G.SCENE.keep_out), default=np.inf)
    return (float(d.min()), guard,
            float((arm_only[:, 2] - arm_only[:, 3]).min()) - G.TABLE_TOP)


def solve(arm):
    seed = list(G.HOME)
    # The fixed transform from the tool frame to the camera's optical frame.
    poses, _ = arm.fk(seed)
    tool_to_cam = np.linalg.inv(poses[arm.tool]) @ poses["camera_link"]
    out = []
    for name, eye, target in G.VIEWS:
        best = None
        for roll in ROLLS_DEG:
            cam_T = look_at(eye, target, roll)
            for q in arm.ik(cam_T @ np.linalg.inv(tool_to_cam), seed):
                cost = float(np.abs(q - np.asarray(seed)).sum())
                if best is None or cost < best[0]:
                    best = (cost, roll, q)
        if best is None:
            print(f"[poses] FAIL {name}: no roll of the view has a collision-free IK solution")
            return None
        _, roll, q = best
        got = arm.fk(q)[0]["camera_link"]
        want = look_at(eye, target, roll)
        err_mm = 1000 * np.linalg.norm(got[:3, 3] - want[:3, 3])
        err_deg = math.degrees(math.acos(np.clip(got[:3, 2] @ want[:3, 2], -1, 1)))
        print(f"[poses] {name:7s} roll {roll:3d} deg   camera off by {err_mm:.1f} mm, "
              f"view axis off by {err_deg:.2f} deg, {math.degrees(best[0]):.0f} deg of "
              f"joint travel from the last pose")
        out.append((name, [round(float(v), 4) for v in q]))
        seed = list(q)
    return out


def check(arm, named):
    """Each pose and each joint-linear blend of the scan, against NO_GO and the table."""
    ok = True
    worst = {}
    cycle = named + [named[0]]            # the scan ends back at HOME
    for (a, qa), (b, qb) in zip(cycle, cycle[1:]):
        qa, qb = np.asarray(qa), np.asarray(qb)
        d_min, g_min, z_min = 1e9, 1e9, 1e9
        for t in np.linspace(0.0, 1.0, BLEND_SAMPLES):
            d, g, z = clearance(arm, (1 - t) * qa + t * qb)
            d_min, g_min, z_min = min(d_min, d), min(g_min, g), min(z_min, z)
        good = d_min >= MARGIN and g_min >= 0
        ok &= good
        worst[f"{a} -> {b}"] = (d_min, g_min, z_min)
        print(f"[poses] {'ok  ' if good else 'FAIL'} blend {a:>7s} -> {b:<7s}  "
              f"NO_GO {1000 * d_min:+6.0f} mm   guard {1000 * g_min:+6.0f} mm   "
              f"arm {1000 * z_min:+5.0f} mm above the table")
    for name, q in named:
        cam = arm.fk(q)[0]["camera_link"]
        # How much table the view covers, and at what resolution, straight down
        # or not: the distance along the view axis to the table plane.
        view = cam[:3, 2]
        rng = -cam[2, 3] / view[2] if view[2] < -1e-6 else float("inf")
        print(f"[poses]      {name:7s} camera at [{cam[0, 3]:+.3f} {cam[1, 3]:+.3f} "
              f"{cam[2, 3]:+.3f}], {rng:.2f} m to the table along the view, "
              f"{1000 * 2 * rng * math.tan(math.radians(34.5)) / 640:.1f} mm/pixel there")
    return ok, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="check grasp/scene.py's poses, no solving")
    args = ap.parse_args()
    arm = Arm()
    if args.check:
        named = [("home", G.HOME)] + [(f"scan{i}", q) for i, q in enumerate(G.SCAN_POSES)
                                      if list(q) != list(G.HOME)]
    else:
        named = solve(arm)
        if named is None:
            sys.exit(1)
    ok, _ = check(arm, named)
    if not args.check:
        print("\n# paste into scripts/grasp/scene.py")
        print(f"HOME = {named[0][1]}")
        print("SCAN_POSES = [")
        for name, q in named + [named[0]]:
            print(f"    {q},   # {name}")
        print("]")
    print(f"[poses] {'ALL PASSED' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
