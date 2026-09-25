"""Generate configs/ur5_robotiq.yml from assets/robot/ur5_robotiq/ur5_robotiq.urdf.

Every link is fitted from its own mesh. Nothing is spliced in from another
robot's config: the arm meshes here are UR5 (CB3), and cuRobo's shipped,
hand-tuned sphere sets are for the e-Series, so borrowing them would put
spheres in the wrong places -- silently, since nothing checks that a sphere
lands on its link.

Run:
    python tools/build_ur5_robotiq_config.py
    python tools/check_robot_cfg.py
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from rig import ROBOTS  # noqa: E402

from curobo.robot_builder import RobotBuilder  # noqa: E402
from curobo._src.geom.sphere_fit.fit_spheres import fit_spheres_to_mesh  # noqa: E402
from curobo._src.geom.sphere_fit.types import SphereFitType  # noqa: E402
from curobo._src.geom.sphere_fit.metrics import compute_sphere_fit_metrics  # noqa: E402

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = f"{ROOT}/assets/robot/ur5_robotiq"
URDF = f"{ASSETS}/ur5_robotiq.urdf"
OUT = f"{ROOT}/configs/ur5_robotiq.yml"

ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

# Weights and limits for how this project wants the arm to move. Copied from
# cuRobo 0.7.7's hand-tuned ur5e.yml, which used to be read from
# configs/ur5e.yml at build time; they are properties of the motion, not of
# which UR it is, so they carry across. Its geometry deliberately does not.
TUNED_CSPACE = {
    "cspace_distance_weight": [1.0, 1.0, 1.0, 1.5, 1.5, 1.5],
    "null_space_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "max_jerk": 500.0,
    "max_acceleration": 12.0,
    "position_limit_clip": 0.1,
}

# Sphere budget for the arm, copied from cuRobo's own hand-tuned ur5e.yml.
# Its spheres cannot be reused -- the CB3 link meshes are a different shape,
# by up to 28 mm on wrist_1 -- but its STYLE can, and that style is few big
# spheres that protrude rather than many small ones that follow the surface:
# 27 spheres for the whole arm, one 100 mm ball for the shoulder, none at all
# on the base. Fewer spheres is faster to collision-check, and a sphere that
# sticks out is safe; one that falls short is not.
#
# The automatic count, for comparison, was 87 over the same six links.
#
# The base DOES get spheres, and the planner must not see them.
#
# Both halves of that matter. Any sphere on the base sits inside the table
# cuboid for every configuration -- measured at 17.1 mm of penetration -- so a
# planner that checks them calls the robot in collision whatever it is asked;
# eight base spheres were tried and every plan failed with no map loaded at
# all. cuRobo's own ur5e.yml leaves the base bare for this reason.
#
# But the spheres are also what RobotSegmenter masks the cameras with, and a
# base with no spheres is a base the cameras MAP. Measured after removing
# them: 13 700 of the map's 14 000 voxels sat between z = 0.02 and 0.15, which
# is exactly the base (0..0.024) and the shoulder (0.024..0.157) -- a
# permanent blob of the robot's own body, in the space its shoulder and
# forearm have to pass through, and it took the demo from 0 failures to 74.
#
# The self-hit check could not see it either: it counts occupied voxels inside
# the robot's spheres, and the one link without spheres is invisible to it. It
# reported "0 on the robot" throughout.
#
# So they live here, and planner_server drops them from the PLANNER's copy of
# the config only. See build() there.
SEGMENTER_ONLY = list(ROBOTS["ur5_robotiq"]["arm"]["mask_only_links"])

ARM_SPHERES = {
    "base_link_inertia": 8,
    "shoulder_link": 2,
    "upper_arm_link": 8,
    "forearm_link": 9,
    "wrist_1_link": 4,
    "wrist_2_link": 4,
    "wrist_3_link": 5,
}

# MORPHIT is a randomised optimiser and occasionally collapses -- one run put
# a single 5 mm sphere on wrist_3, covering 0.4% of it. Refuse anything that
# does not cover the link, and keep the best of several attempts rather than
# whatever the first one produced. cuRobo's shipped ur5e spheres score 0.774
# to 1.000 on this same metric, so this bar is above its own tuning.
MIN_COVERAGE = 0.90
# Eight, not four: a clamped link's fit is much more variable, because MORPHIT
# sometimes returns fewer spheres than asked and one clamped sphere cannot
# cover a shoulder. A run that came back with a single 62 mm ball at 0.648 was
# caught by the coverage bar below and cost a whole rebuild.
ATTEMPTS = 8

# Links whose spheres may not reach below their own geometry.
#
# The shoulder is the reason. MorphIt covers it with one 80 mm ball, which is
# 20 mm wider than the metal and hangs to base z = +4 mm -- and the CB3
# shoulder sits at z = 89 mm, 73 mm lower than the e-Series one cuRobo's
# config was tuned on, so that ball ends up 15 mm off the table instead of
# 62 mm. The real table is a cuboid the planner is told about and 15 mm is
# enough for it; the MAPPED table is a 2 cm voxel grid on top of that, and it
# is not. Measured live: with the ball, the collision-aware IK refused BOTH
# goals and every plan failed, with the map showing nothing inside the robot.
#
# upper_arm_link needs it for the same reason and was found the same way: with
# the shoulder fixed, the goals were still refused, and its spheres reached
# 14 mm below its own metal -- 40 mm of real clearance over the table becoming
# 26 mm of sphere clearance, which the mapped table's 2 cm voxels then closed.
#
# Only these two links come near the table. The wrist spheres protrude just as
# far but do it 470 mm up, where it costs nothing.
#
# Clipping is done by clamping radii, not by cuRobo's clip_plane, whose 20 mm
# buffer discards whole spheres and drops the shoulder to 0.40 coverage.
FLOOR_CLAMP = ["shoulder_link", "upper_arm_link"]

# A clamped link cannot reach MIN_COVERAGE -- across runs the shoulder lands
# at 0.78-0.83 and the upper arm at 0.83-0.86, with the uncovered band along
# their lower side, which is the side deliberately pulled back. cuRobo's own
# ur5e.yml scores 0.774 on its worst link, so this bar is its bar rather than
# a new one, and what it really guards against is the collapsed fit: one run
# put a single 62 mm ball on the shoulder at 0.648.
MIN_COVERAGE_CLAMPED = 0.78

# The wrist stack and gripper get budgets too, so every link's sphere count is
# stated here rather than left to whatever the estimator happened to pick that
# run -- and so every link goes through the same best-of-ATTEMPTS guard. One
# ungtuarded run gave the wrist camera 0.888 coverage where four attempts give
# 0.92; the difference was luck, not geometry.
TOOL_SPHERES = {
    "robotiq_ft300_sensor": 6,
    "robotiq_wrist_camera_link": 8,
    "camera_mount": 16,
    "realsense_link": 3,
    "robotiq_85_base_link": 5,
    "robotiq_85_left_knuckle_link": 3,
    "robotiq_85_right_knuckle_link": 3,
    "robotiq_85_left_finger_link": 2,
    "robotiq_85_right_finger_link": 2,
    "robotiq_85_left_inner_knuckle_link": 4,
    "robotiq_85_right_inner_knuckle_link": 4,
    "robotiq_85_left_finger_tip_link": 3,
    "robotiq_85_right_finger_tip_link": 3,
}
BUDGET = {**ARM_SPHERES, **TOOL_SPHERES}

# The FT 300 coupling is a dia 75 x 13 mm washer, and no inscribed-sphere fit
# can cover a washer: the largest sphere that fits inside 13 mm of plate has a
# 6.5 mm radius, so the fitter puts down four 2 mm beads and covers 0.2% of
# it. The way out is the one cuRobo's own tuning takes everywhere -- let the
# spheres stick out. A ring of eight 16 mm spheres plus one in the middle
# covers 98.3% of the plate and reaches 9.5 mm past its faces, into wrist_3
# behind it and the FT 300 in front, both of which this link already ignores.
# Nothing protrudes sideways past the dia 75 edge, so the planner sees the
# real outline.
def _coupling_spheres(n=8, rho=0.0245, radius=0.016, z=0.0065):
    import math
    out = [{"center": [0.0, 0.0, z], "radius": radius}]
    for k in range(n):
        a = 2 * math.pi * k / n
        out.append({"center": [rho * math.cos(a), rho * math.sin(a), z],
                    "radius": radius})
    return out


HAND_PLACED = {"robotiq_ft300_mounting_plate": _coupling_spheres()}

# 0 rad is fully open, which is the pose to plan against: the widest the
# gripper ever is, so a path that clears with the fingers open clears with
# them anywhere. Locking is not the same as the joints being fixed -- cuRobo
# still places the finger links, and their spheres, at these angles.
GRIPPER_LOCK_RAD = 0.0

# The wrist stack and the camera bracket are bolted into one rigid assembly,
# so contact between them is by design, not a collision to plan around.
RIGID_GROUP = [
    "wrist_3_link", "flange", "tool0",
    "robotiq_ft300_mounting_plate", "robotiq_ft300_sensor",
    "robotiq_wrist_camera_link",
    "camera_mount", "realsense_link",
    "robotiq_85_base_link",
    "robotiq_85_left_knuckle_link", "robotiq_85_left_finger_link",
    "robotiq_85_left_inner_knuckle_link", "robotiq_85_left_finger_tip_link",
    "robotiq_85_right_knuckle_link", "robotiq_85_right_finger_link",
    "robotiq_85_right_inner_knuckle_link", "robotiq_85_right_finger_tip_link",
]


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def link_mesh(link):
    """The link's collision geometry, in the LINK's own frame.

    Which is the frame collision spheres are written in, so a mesh loaded any
    other way would be compared against the spheres in the wrong place.
    """
    parts = []
    for el in link.findall("collision"):
        geo = el.find("geometry")
        m, cyl, box = geo.find("mesh"), geo.find("cylinder"), geo.find("box")
        if m is not None:
            g = trimesh.load(os.path.join(ASSETS, m.get("filename")), force="mesh")
            if m.get("scale"):
                g.apply_scale([float(v) for v in m.get("scale").split()])
        elif cyl is not None:
            # The wrist stack is described with primitives rather than meshes;
            # they are the collision geometry, so they are what the spheres
            # have to be judged against.
            g = trimesh.creation.cylinder(radius=float(cyl.get("radius")),
                                          height=float(cyl.get("length")))
        elif box is not None:
            g = trimesh.creation.box(
                extents=[float(v) for v in box.get("size").split()])
        else:
            continue
        o = el.find("origin")
        T = np.eye(4)
        if o is not None:
            T[:3, 3] = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
            T[:3, :3] = _rpy(*[float(v) for v in (o.get("rpy") or "0 0 0").split()])
        g.apply_transform(T)
        parts.append(g)
    if not parts:
        return None
    return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]


def _np(x):
    import torch
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def fit_budgeted(mesh, budget, clamp_floor=False):
    """Best of ATTEMPTS fits at a fixed sphere count, by coverage.

    With clamp_floor, every sphere's radius is cut back so it does not reach
    below the mesh's own lowest point, and the metrics are recomputed from the
    clamped spheres -- so the number reported is the number that ships.
    """
    floor = float(mesh.bounds[0][2])
    best = None
    for _ in range(ATTEMPTS):
        r = fit_spheres_to_mesh(mesh, num_spheres=budget,
                                fit_type=SphereFitType.MORPHIT,
                                iterations=400, compute_metrics=not clamp_floor)
        C = _np(r.centers).astype(float)
        R = _np(r.radii).astype(float).ravel()
        if clamp_floor:
            R = np.minimum(R, np.maximum(C[:, 2] - floor, 1e-4))
            keep = R > 0.004
            C, R = C[keep], R[keep]
            if not len(C):
                continue
            metrics = compute_sphere_fit_metrics(mesh, C, R)
        else:
            metrics = r.metrics
        # Rank by sphere COUNT first, then coverage. MORPHIT sometimes
        # returns fewer spheres than asked for, and on a clamped link one
        # survivor cannot cover anything -- the shoulder came back as a single
        # ball at 0.65 twice. Coverage alone does not separate those from a
        # genuine two-sphere fit, because a fat lone ball scores respectably
        # until the clamp cuts it down.
        rank = (min(len(C), budget), metrics.coverage)
        if best is None or rank > best[0]:
            best = (rank, metrics, C, R)
    return best[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sphere-density", type=float, default=1.0)
    args = ap.parse_args()

    builder = RobotBuilder(urdf_path=URDF, asset_path=ASSETS,
                           tool_frames=["grasp_frame", "camera_link"])
    print(f"fitting collision spheres (density {args.sphere_density})...")
    builder.fit_collision_spheres(sphere_density=args.sphere_density,
                                  use_collision_mesh=True, compute_metrics=True)
    print(f"  {builder.num_spheres} spheres over {len(builder.collision_link_names)} links")
    builder.compute_collision_matrix()
    builder.save(builder.build(), OUT)

    built = yaml.safe_load(open(OUT))
    k = built.get("robot_cfg", built)["kinematics"]

    # Re-fit the arm to the ur5e budget, and report what every link scored.
    urdf_links = {l.get("name"): l for l in ET.parse(URDF).getroot().findall("link")}
    print(f"  {'link':<36s} {'n':>3s} {'r (mm)':>9s} {'cover':>6s} {'protr':>6s} {'gap95':>8s}")
    failed = []
    for name in k["collision_link_names"]:
        mesh = link_mesh(urdf_links[name])
        if mesh is None:
            print(f"  {name:<36s}   -   (no collision mesh)")
            continue
        if name in HAND_PLACED:
            k["collision_spheres"][name] = [dict(x) for x in HAND_PLACED[name]]
            s = k["collision_spheres"][name]
            C = np.array([x["center"] for x in s])
            R = np.array([x["radius"] for x in s])
            m = compute_sphere_fit_metrics(mesh, C, R)
            n, lo, hi = len(s), R.min(), R.max()
            if m.coverage < MIN_COVERAGE:
                failed.append((name, m.coverage))
        elif name in BUDGET:
            clamped = name in FLOOR_CLAMP
            m, C, R = fit_budgeted(mesh, BUDGET[name], clamp_floor=clamped)
            k["collision_spheres"][name] = [
                {"center": [float(v) for v in c], "radius": float(rad)}
                for c, rad in zip(C, R)
            ]
            n, lo, hi = len(R), R.min(), R.max()
            bar = MIN_COVERAGE_CLAMPED if clamped else MIN_COVERAGE
            if m.coverage < bar:
                failed.append((name, m.coverage))
        else:
            raise SystemExit(f"{name} has collision geometry but no sphere budget")
        print(f"  {name:<36s} {n:>3d} {lo * 1000:4.0f}-{hi * 1000:<4.0f} "
              f"{m.coverage:>6.3f} {m.protrusion:>6.3f} {m.surface_gap_p95 * 1000:>6.1f}mm")
    if failed:
        raise SystemExit("coverage below %.2f: %s" % (MIN_COVERAGE, failed))

    k["tool_frames"] = ["grasp_frame", "camera_link"]
    k["lock_joints"] = {j: GRIPPER_LOCK_RAD * mult
                        for j, mult in ROBOTS["ur5_robotiq"]["gripper"]["joints"].items()}

    ignore = {kk: list(v) for kk, v in (k.get("self_collision_ignore") or {}).items()}
    present = set(k["collision_link_names"])
    for link in RIGID_GROUP:
        if link not in present:
            continue
        others = [o for o in RIGID_GROUP if o != link and o in present]
        ignore[link] = sorted(set(ignore.get(link, [])) | set(others))
    k["self_collision_ignore"] = ignore

    cs = k["cspace"]
    if cs["joint_names"] != ARM_JOINTS:
        names = cs["joint_names"]
        width = len(names)   # capture first: joint_names is itself one of the
                             # per-joint lists being trimmed
        keep = [i for i, n in enumerate(names) if n in ARM_JOINTS]
        for key, val in list(cs.items()):
            if key != "joint_names" and isinstance(val, list) and len(val) == width:
                cs[key] = [val[i] for i in keep]
        cs["joint_names"] = [names[i] for i in keep]
    assert cs["joint_names"] == ARM_JOINTS, cs["joint_names"]
    cs.update(TUNED_CSPACE)
    k["format_version"] = 2.0
    # RobotBuilder records the absolute paths it was given. Every caller hands
    # cuRobo absolute paths through ContentPath, which overwrites these two on
    # load, so they are informational -- and an absolute one would name this
    # machine's mount point. Relative to the project root instead.
    k["urdf_path"] = os.path.relpath(URDF, ROOT)
    k["asset_root_path"] = os.path.relpath(ASSETS, ROOT)

    out = built if "robot_cfg" in built else {"robot_cfg": built}
    yaml.safe_dump(out, open(OUT, "w"), sort_keys=False, default_flow_style=None)
    print(f"wrote {OUT}")
    print(f"  cspace     : {cs['joint_names']}")
    print(f"  locked     : {len(k['lock_joints'])} gripper joints at {GRIPPER_LOCK_RAD} rad")
    print(f"  tool_frames: {k['tool_frames']}")


if __name__ == "__main__":
    main()
