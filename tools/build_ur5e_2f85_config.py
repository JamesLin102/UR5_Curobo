"""Generate configs/ur5e_robotiq_2f_85.yml with cuRobo 0.8's RobotBuilder.

The arm's collision spheres are the hand-tuned ones from cuRobo 0.7.7's
ur5e.yml -- they are known good and already verified to plan. Only the 2F-85
gripper links get auto-fitted spheres, because no tuned set exists for them.

Run:
    python tools/build_ur5e_2f85_config.py
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from rig import ROBOTS  # noqa: E402

from curobo.robot_builder import RobotBuilder  # noqa: E402

GRIPPER_COUPLING = ROBOTS["ur5e_2f85"]["gripper_joints"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = f"{ROOT}/assets/robot/ur_description/ur5e_robotiq_2f_85.urdf"
ASSETS = f"{ROOT}/assets/robot/ur_description"
ARM_YML = f"{ROOT}/configs/ur5e.yml"
OUT = f"{ROOT}/configs/ur5e_robotiq_2f_85.yml"

# The D435i body, as spheres along its 90 mm length (which runs along Y,
# parallel to the finger opening). Hand-placed rather than
# auto-fitted: it is a plain box in the URDF, not a mesh, and three spheres of
# r=18 mm cover the 25x25 mm cross-section with its corners.
CAMERA_SPHERES = [
    {"center": [0.0, -0.030, 0.0], "radius": 0.018},
    {"center": [0.0, 0.000, 0.0], "radius": 0.018},
    {"center": [0.0, 0.030, 0.0], "radius": 0.018},
]

# The FT 300 and the Wrist Camera are 75 mm discs / an 87.5 x 75 puck stacked
# on the tool axis. Both are simple convex bodies whose outer dimensions are
# known exactly, so they get hand-placed spheres rather than a mesh fit -- and
# the camera has no mesh to fit in the first place.
#
# Sphere centres are in the LINK's OWN frame, which is the trap here: both of
# these links sit partway up the tool axis, and ft300_sensor is additionally
# flipped 180 deg about Y by the official mounting transform. Writing centres
# in tool0 coordinates puts them nowhere near the body -- the first attempt
# floated the camera's spheres 48 mm up inside the gripper, where they were
# invisible and useless.
#
# ft300_coupling is the one link in the stack whose joint is identity relative
# to tool0, so ALL the FT 300 spheres live on it and its own frame doubles as
# tool0's. That sidesteps the flipped sensor frame entirely; ft300_sensor
# keeps its mesh for rendering and carries no spheres.
#
# FT 300: measured 75 mm across, occupying z 0..41.5 mm above tool0. Three
# rings of four cover the square-ish footprint without one fat sphere
# swallowing the flange.
FT300_SPHERES = [
    {"center": [dx, dy, z], "radius": 0.024}
    for z in (0.008, 0.024, 0.038)
    for dx, dy in ((-0.021, -0.021), (-0.021, 0.021), (0.021, -0.021), (0.021, 0.021))
]

# Wrist Camera: the box is centred on its own link origin, so the spheres are
# centred at local z=0. 87.5 mm along X, 75 mm along Y, 22.4 mm thick.
WRIST_CAM_SPHERES = [
    {"center": [dx, dy, 0.0], "radius": 0.023}
    for dx in (-0.021, 0.0, 0.021)
    for dy in (-0.015, 0.015)
]

# Neighbours in the stack touch by design; the whole wrist assembly is one
# rigid body as far as self-collision goes.
STACK_IGNORES = {
    "ft300_coupling": ["wrist_3_link", "tool0", "ft300_sensor", "wrist_camera"],
    "ft300_sensor": ["wrist_3_link", "tool0", "ft300_coupling", "wrist_camera",
                     "robotiq_arg2f_base_link"],
    "wrist_camera": ["tool0", "ft300_coupling", "ft300_sensor", "camera_mount",
                     "robotiq_arg2f_base_link",
                     "left_outer_knuckle", "left_inner_knuckle",
                     "right_outer_knuckle", "right_inner_knuckle"],
}

# The camera is bolted alongside the gripper, so it touches these by design.
# The camera is bolted to the gripper: one rigid assembly. Every link of it
# ignores the camera, fingers included, so a bracket that touches the body is
# not reported as a collision.
CAMERA_IGNORES = [
    "robotiq_arg2f_base_link", "wrist_3_link", "tool0", "wrist_camera", "ft300_sensor",
    "left_outer_knuckle", "left_inner_knuckle", "left_outer_finger", "left_inner_finger",
    "right_outer_knuckle", "right_inner_knuckle", "right_outer_finger", "right_inner_finger",
]

# The gripper is articulated in the URDF but LOCKED here, so cuRobo plans 6
# DOF and not 12. franka.yml does the same with its two finger joints.
#
# Locking is not the same as the joints being fixed: cuRobo still places the
# finger links -- and therefore their collision spheres -- at these angles, so
# the planner knows the gripper's real shape.
#
# ALL SIX are listed. The URDF carries no <mimic> tags (PhysX rejects them and
# breaks the articulation, see build_ur5e_2f85_urdf.py), so nothing derives
# the followers for us; the coupling comes from rig.ROBOTS, which is also what
# the simulator drives from.
#
# 0 rad is fully open, which is the pose to plan against: it is the widest the
# gripper ever is, so a path that clears with the fingers open clears with
# them anywhere. A plan made at one lock value is NOT valid at another.
GRIPPER_LOCK_RAD = 0.0

GRIPPER_LINKS = [
    "robotiq_arg2f_base_link",
    "left_outer_knuckle", "left_outer_finger", "left_inner_finger",
    "left_inner_knuckle",
    "right_outer_knuckle", "right_outer_finger", "right_inner_finger",
    "right_inner_knuckle",
]


def main():
    builder = RobotBuilder(urdf_path=URDF, asset_path=ASSETS,
                           tool_frames=["grasp_frame", "camera_link"])
    print("fitting collision spheres to the 2F-85 meshes...")
    builder.fit_collision_spheres(sphere_density=1.0, compute_metrics=True)
    print(f"  {builder.num_spheres} spheres over {len(builder.collision_link_names)} links")
    builder.compute_collision_matrix()
    cfg = builder.build()
    builder.save(cfg, OUT)

    # Splice: keep the arm exactly as the verified ur5e.yml has it, keep the
    # builder's output only for the gripper.
    arm = yaml.safe_load(open(ARM_YML))["robot_cfg"]["kinematics"]
    built = yaml.safe_load(open(OUT))
    k = built.get("robot_cfg", built)["kinematics"]

    arm_spheres = arm["collision_spheres"]
    merged = {name: arm_spheres[name] for name in arm["collision_link_names"] if name in arm_spheres}
    kept, dropped = [], []
    for name in GRIPPER_LINKS:
        if name in k["collision_spheres"] and k["collision_spheres"][name]:
            merged[name] = k["collision_spheres"][name]
            kept.append(name)
        else:
            dropped.append(name)

    merged["camera_mount"] = [dict(x) for x in CAMERA_SPHERES]
    merged["ft300_coupling"] = [dict(x) for x in FT300_SPHERES]
    merged["wrist_camera"] = [dict(x) for x in WRIST_CAM_SPHERES]
    k["collision_spheres"] = merged
    k["collision_link_names"] = (arm["collision_link_names"] + kept
                                 + ["camera_mount", "ft300_coupling",
                                    "wrist_camera"])
    for field in ("self_collision_buffer",):
        k.setdefault(field, {})
        k[field].update(arm.get(field, {}))
    # Arm-internal ignores are the tuned ones; keep the builder's computed
    # entries for every pair that involves the gripper.
    ignore = {kk: list(v) for kk, v in (k.get("self_collision_ignore") or {}).items()}
    for kk, v in (arm.get("self_collision_ignore") or {}).items():
        ignore[kk] = sorted(set(ignore.get(kk, [])) | set(v))
    k["self_collision_ignore"] = ignore
    # Carry the arm's tuned cspace settings across, so regenerating this file
    # does not silently undo the wrist weighting.
    for key in ("default_joint_position", "cspace_distance_weight", "null_space_weight"):
        if key in arm["cspace"]:
            k["cspace"][key] = arm["cspace"][key]
    ignore["camera_mount"] = sorted(set(ignore.get("camera_mount", [])) | set(CAMERA_IGNORES))
    for link, others in STACK_IGNORES.items():
        ignore[link] = sorted(set(ignore.get(link, [])) | set(others))
    k["self_collision_ignore"] = ignore
    k["tool_frames"] = ["grasp_frame", "camera_link"]
    k["lock_joints"] = {j: GRIPPER_LOCK_RAD * mult
                        for j, mult in GRIPPER_COUPLING.items()}
    # cspace stays the six arm joints; the builder may have picked up the
    # gripper's driving joint as a seventh, so drop it explicitly rather than
    # trusting lock_joints to have already done it.
    arm_joints = arm["cspace"]["joint_names"]
    cs = k["cspace"]
    if cs.get("joint_names") != arm_joints:
        names = cs["joint_names"]
        width = len(names)          # capture BEFORE trimming anything: joint_names
                                    # is itself one of the per-joint lists, and
                                    # shrinking it mid-loop silently stops the
                                    # rest from matching
        keep = [i for i, n in enumerate(names) if n in arm_joints]
        for key, val in list(cs.items()):
            if key != "joint_names" and isinstance(val, list) and len(val) == width:
                cs[key] = [val[i] for i in keep]
        cs["joint_names"] = [names[i] for i in keep]
    k["format_version"] = 2.0

    out = built if "robot_cfg" in built else {"robot_cfg": built}
    yaml.safe_dump(out, open(OUT, "w"), sort_keys=False, default_flow_style=None)
    print(f"wrote {OUT}")
    print(f"  arm links (tuned spheres) : {arm['collision_link_names']}")
    print(f"  gripper links (auto-fit)  : {kept}")
    print(f"  camera_mount              : {len(CAMERA_SPHERES)} hand-placed spheres")
    print(f"  ft300_coupling            : {len(FT300_SPHERES)} hand-placed spheres "
          f"(the whole FT 300; ft300_sensor is mesh-only)")
    print(f"  wrist_camera              : {len(WRIST_CAM_SPHERES)} hand-placed spheres")
    print(f"  gripper                   : {len(GRIPPER_COUPLING)} joints locked at "
          f"{GRIPPER_LOCK_RAD} rad (open); cspace = {k['cspace']['joint_names']}")
    if dropped:
        print(f"  gripper links with no mesh: {dropped}")


if __name__ == "__main__":
    main()
