"""Can the arm actually grip the cube everywhere grasp/scene.py lets it stand, without hitting a cylinder?

    python tools/grasp_reach.py                 # both parts, ~ a few minutes
    python tools/grasp_reach.py --layouts 500   # more random layouts
    python tools/grasp_reach.py --grid-only

cuRobo only, no simulator: kinematics and collision, not physics. Whether the
fingers then HOLD the cube is a separate question for the simulator.

A grasp is what legs.leg_ops does, taken apart:

  plan    HOME -> the pre-grasp, DESCEND over the grip, with the planner's
          collision-aware plan_pose against the table (and, with cylinders, the
          cylinders -- training mode, where the planner is told)
  down    IK for the grip pose seeded from the pre-grasp, as MoveZ asks the
          server; the joints are interpolated between, so that joint-space
          path is what is checked
  up      the same from the grip to lift_m above it
  clear   along down and up, every collision sphere of the arm against every
          cylinder (the real cylinder, not the planner's box) and against the
          table; the cube itself is what the fingers are meant to reach

Part 1, the grid: the empty table, the cube at every point of a grid over
CUBE_XY, the tool at every yaw the policy can ask for ([-90, 90) deg). Says
where the arm can grip at all.

Part 2, random layouts: the cube anywhere in CUBE_XY at any yaw, 1-4
cylinders in CYL_XY at least 15 mm from it and from each other, and the two
grips an oracle would try (tool square to either pair of faces). Says how often
a layout has a grip, broken down by how close the nearest cylinder stands.

The planner sees each cylinder as cuRobo's own stand-in for one, its bounding
box (Cylinder.get_cuboid, 70 x 70 mm): that is what training mode will give it.
"""

import argparse
import math
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import torch  # noqa: E402

from grasp import scene as G  # noqa: E402
from legs import tool_pose  # noqa: E402
from rig import DEFAULT_ROBOT, ROBOTS  # noqa: E402

MIN_GAP = 0.015        # cylinder to cube, cylinder to cylinder, surface to surface
INTERP = 70            # cell_api.MOVE_Z_STEPS: the path the arm takes, sampled here
GRID_YAWS = np.radians(np.arange(-90, 90, 15))
# The fingers are meant to come down to the cube, which stands on the table,
# so their spheres (knuckles, fingers, tips) are held to the cylinders but not
# to the table; everything from the gripper body up is held to both.
TABLE_MARGIN = 0.0
FINGER_LINKS = ("knuckle", "finger")


def say(msg):
    print(f"[reach] {msg}", flush=True)


class Arm:
    def __init__(self):
        import planner_server as ps
        from curobo.scene import Cuboid, Scene
        from curobo.types import GoalToolPose, JointState, Pose

        self.Cuboid, self.Scene = Cuboid, Scene
        self.JointState, self.GoalToolPose, self.Pose = JointState, GoalToolPose, Pose
        self.kin, self.planner, self.free_ik, _, self.tool = ps.build(
            "ur5_robotiq", G.SCENE, use_cuda_graph=False)
        self.names = list(self.kin.joint_names)
        # The planner's static world, as planner_server.static_scene builds it.
        self.table = [Cuboid(name=n, dims=d, pose=p)
                      for n, d, p, _ in list(G.SCENE.obstacles) + list(G.SCENE.keep_out)]
        self.finger = None
        self.set_cylinders([])

    def set_cylinders(self, xy):
        from curobo.scene import Cylinder
        cyl = [Cylinder(name=f"c{i}", radius=G.CYL_RADIUS, height=G.CYL_HEIGHT,
                        pose=[x, y, G.TABLE_TOP + G.CYL_HEIGHT / 2, 1, 0, 0, 0])
               for i, (x, y) in enumerate(xy)]
        self.planner.update_world(self.Scene(cuboid=self.table + [c.get_cuboid() for c in cyl]))
        self.cylinders = list(xy)

    def _js(self, q):
        return self.JointState.from_position(
            torch.tensor(np.atleast_2d(q), device="cuda", dtype=torch.float32),
            joint_names=self.names)

    def plan(self, start_q, pose):
        goal = self.GoalToolPose.from_poses({self.tool: self.Pose.from_list(list(pose))})
        res = self.planner.plan_pose(goal, self._js(start_q))
        if res is None or not bool(res.success[0]):
            return None
        itraj = res.interpolated_trajectory
        n = int(res.interpolated_last_tstep[0]) if res.interpolated_last_tstep is not None \
            else itraj.position.shape[2]
        names = list(itraj.joint_names) if itraj.joint_names is not None else self.names
        keep = [names.index(j) for j in self.names]
        return itraj.position[0, 0, n - 1, keep].cpu().numpy().astype(float)

    def ik(self, seed_q, pose):
        goal = self.GoalToolPose.from_poses({self.tool: self.Pose.from_list(list(pose))})
        res = self.free_ik.solve_pose(goal, current_state=self._js(seed_q))
        if res is None or not bool(res.success.any()):
            return None
        return res.solution[res.success].view(-1, len(self.names))[0].cpu().numpy().astype(float)

    def spheres(self, qs):
        k = self.kin.compute_kinematics(self._js(qs))
        s = k.robot_spheres.cpu().numpy().reshape(len(np.atleast_2d(qs)), -1, 4)
        if self.finger is None:
            n = s.shape[1]
            self.finger = self._link_mask(
                n, lambda l: l.startswith("robotiq_85_") and any(f in l for f in FINGER_LINKS))
            # The base's own spheres sit inside the table on purpose: they are
            # there to mask the base out of depth, and the planner drops them
            # (rig.ROBOTS[...]["arm"]["mask_only_links"]).
            mask_only = set(ROBOTS[DEFAULT_ROBOT]["arm"]["mask_only_links"])
            self.base = self._link_mask(n, lambda l: l in mask_only)
        return s

    def _link_mask(self, n, pick):
        """Spheres hanging off the links `pick(link name)` accepts."""
        kc = self.kin.config.kinematics_config
        mask = np.zeros(n, dtype=bool)
        for link in kc.all_link_names:
            if pick(link):
                try:
                    idx = np.asarray(kc.get_sphere_index_from_link_name(link).cpu()).ravel()
                except Exception:
                    continue
                mask[idx[idx < n]] = True
        return mask

    def path_clearance(self, q0, q1):
        """(min to any cylinder, min of the arm above the table) along the joint blend."""
        qs = np.linspace(q0, q1, INTERP)
        s = self.spheres(qs)
        c, r = s[..., :3].reshape(-1, 3), s[..., 3].reshape(-1)
        live = r > 0
        d_cyl = np.inf
        for x, y in self.cylinders:
            radial = np.linalg.norm(c[:, :2] - [x, y], axis=1) - G.CYL_RADIUS
            axial = np.abs(c[:, 2] - (G.TABLE_TOP + G.CYL_HEIGHT / 2)) - G.CYL_HEIGHT / 2
            q = np.stack([radial, axial], axis=1)
            d = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0) - r
            d_cyl = min(d_cyl, float(d[live].min()))
        arm = live & ~np.tile(self.finger | self.base, INTERP)
        d_table = float((c[arm, 2] - r[arm]).min()) - G.TABLE_TOP
        return d_cyl, d_table


def try_grip(arm, x, y, yaw):
    """None if the grip works, else why not. Also returns the clearances seen."""
    grip = tool_pose(x, y, G.GRASP_Z, yaw)
    above = list(grip)
    above[2] += G.DESCEND
    lift = list(grip)
    lift[2] += G.SCENE.pick["lift_m"]
    q_pre = arm.plan(G.HOME, above)
    if q_pre is None:
        return "no plan", None
    q_grip = arm.ik(q_pre, grip)
    if q_grip is None:
        return "no IK down", None
    q_lift = arm.ik(q_grip, lift)
    if q_lift is None:
        return "no IK up", None
    if np.abs(q_grip - q_pre).max() > 1.0 or np.abs(q_lift - q_grip).max() > 1.0:
        return "IK on another branch", None
    c1, t1 = arm.path_clearance(q_pre, q_grip)
    c2, t2 = arm.path_clearance(q_grip, q_lift)
    clear = (min(c1, c2), min(t1, t2))
    if clear[0] < 0:
        return "hits a cylinder", clear
    if clear[1] < TABLE_MARGIN:
        return "arm hits the table", clear
    return None, clear


def grid(arm):
    xs = np.round(np.arange(G.CUBE_XY[0][0], G.CUBE_XY[0][1] + 1e-9, 0.05), 3)
    ys = np.round(np.arange(G.CUBE_XY[1][0], G.CUBE_XY[1][1] + 1e-9, 0.07), 3)
    arm.set_cylinders([])
    why = Counter()
    table = {}
    lowest = np.inf
    for x in xs:
        for y in ys:
            ok = 0
            for yaw in GRID_YAWS:
                r, clear = try_grip(arm, x, y, yaw)
                why[r or "ok"] += 1
                ok += r is None
                if clear is not None:
                    lowest = min(lowest, clear[1])
            table[(x, y)] = ok
    say(f"part 1, empty table: {len(xs)} x {len(ys)} cube positions x {len(GRID_YAWS)} tool yaws "
        f"({math.degrees(GRID_YAWS[0]):.0f}..{math.degrees(GRID_YAWS[-1]):.0f} deg)")
    say("  grips that work, of " + str(len(GRID_YAWS)) + ", by cube position "
        "(rows x, columns y):")
    say("        " + " ".join(f"{y:+.2f}" for y in ys))
    for x in xs:
        say(f"  {x:.2f} " + " ".join(f"{table[(x, y)]:5d}" for y in ys))
    say(f"  outcomes: {dict(why)}")
    say(f"  lowest the arm came to the table while gripping: {1000 * lowest:+.0f} mm")
    return why


def sample_layout(rng, n_cyl, near_p=0.0, near_gap=(0.015, 0.12)):
    """The cube, and up to n_cyl cylinders that keep MIN_GAP from it and each other.

    near_p: the chance each cylinder is put near the cube -- its surface
    near_gap from the cube's, in a random direction -- rather than anywhere in
    CYL_XY. Uniform placement mostly leaves the cube in the open.
    """
    x = rng.uniform(*G.CUBE_XY[0])
    y = rng.uniform(*G.CUBE_XY[1])
    yaw = rng.uniform(-math.pi / 4, math.pi / 4)
    corners = np.array([[sx, sy] for sx in (-1, 1) for sy in (-1, 1)]) * G.CUBE_SIZE / 2
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    cube_poly = corners @ R.T + [x, y]
    cyl = []
    for _ in range(200):
        if len(cyl) == n_cyl:
            break
        if rng.random() < near_p:
            ang = rng.uniform(-math.pi, math.pi)
            d = G.CUBE_SIZE / 2 + G.CYL_RADIUS + rng.uniform(*near_gap)
            c = np.array([x + d * math.cos(ang), y + d * math.sin(ang)])
            if not (G.CYL_XY[0][0] <= c[0] <= G.CYL_XY[0][1]
                    and G.CYL_XY[1][0] <= c[1] <= G.CYL_XY[1][1]):
                continue
        else:
            c = np.array([rng.uniform(*G.CYL_XY[0]), rng.uniform(*G.CYL_XY[1])])
        # Distance from the cylinder's axis to the cube's square, exactly.
        local = (c - [x, y]) @ R
        q = np.abs(local) - G.CUBE_SIZE / 2
        to_cube = np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0)
        if to_cube - G.CYL_RADIUS < MIN_GAP:
            continue
        if any(np.linalg.norm(c - o) - 2 * G.CYL_RADIUS < MIN_GAP for o in cyl):
            continue
        cyl.append(c)
    gap = min((np.linalg.norm(np.maximum(np.abs((c - [x, y]) @ R) - G.CUBE_SIZE / 2, 0))
               - G.CYL_RADIUS for c in cyl), default=np.inf)
    return (x, y, yaw), [tuple(c) for c in cyl], gap, cube_poly


def layouts(arm, n, seed):
    rng = np.random.default_rng(seed)
    buckets = [(0.015, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, np.inf)]
    stats = {b: Counter() for b in buckets}
    why = Counter()
    for i in range(n):
        n_cyl = int(rng.integers(1, G.MAX_CYLINDERS + 1))
        (x, y, yaw), cyl, gap, _ = sample_layout(rng, n_cyl)
        arm.set_cylinders(cyl)
        ok = 0
        for face in (0.0, math.pi / 2):
            t = (yaw + face + math.pi / 2) % math.pi - math.pi / 2     # into [-90, 90)
            r, _ = try_grip(arm, x, y, t)
            why[r or "ok"] += 1
            ok += r is None
        b = next(b for b in buckets if b[0] <= gap < b[1])
        stats[b][ok] += 1
        if (i + 1) % 50 == 0:
            say(f"  {i + 1}/{n} layouts")
    say(f"part 2, {n} random layouts, 1-{G.MAX_CYLINDERS} cylinders, the two face-square grips each")
    say("  nearest cylinder to the cube   layouts   both grips   one   none")
    for b, c in stats.items():
        tot = sum(c.values())
        if tot:
            hi = "and more" if b[1] == np.inf else f"- {1000 * b[1]:.0f} mm"
            say(f"  {1000 * b[0]:4.0f} {hi:10s}             {tot:5d}   {c[2]:10d}   {c[1]:3d}   {c[0]:4d}")
    say(f"  outcomes over every grip tried: {dict(why)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layouts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-only", action="store_true")
    args = ap.parse_args()
    arm = Arm()
    grid(arm)
    if not args.grid_only:
        layouts(arm, args.layouts, args.seed)


if __name__ == "__main__":
    main()
