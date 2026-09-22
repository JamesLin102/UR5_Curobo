"""Build ur5e_robotiq_2f_85.urdf by splicing the UR5e arm with a Robotiq 2F-85.

cuRobo 0.7.7 shipped a UR5e with a 2F-*140*, and 2F-85 geometry only on a
Kinova arm. Neither is what we run. This takes the arm from ur5e.urdf and the
2F-85 link/joint chain from kinova_gen3_7dof.urdf, re-parents the gripper onto
tool0, and adds a grasp frame at the finger pads.

The gripper joints stay FIXED, exactly as cuRobo's own Kinova 2F-85 model has
them: the fingers are rigid collision geometry for planning, not an actuated
mechanism. Modelling the real 4-bar linkage needs a loop-closure joint that
URDF cannot express, and PhysX handles it badly.

Run:
    python tools/build_ur5e_2f85_urdf.py
"""

import copy
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from rig import ROBOTS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V7 = "/home/eencku/curobo-0.7.7/src/curobo/content/assets/robot"

ARM = f"{ROOT}/assets/robot/ur_description/ur5e.urdf"
DONOR = f"{V7}/kinova/kinova_gen3_7dof.urdf"
OUT = f"{ROOT}/assets/robot/ur_description/ur5e_robotiq_2f_85.urdf"

GRIPPER_LINKS = [
    "robotiq_arg2f_base_link",
    "left_outer_knuckle", "left_outer_finger", "left_inner_finger",
    "left_inner_finger_pad", "left_inner_knuckle",
    "right_outer_knuckle", "right_outer_finger", "right_inner_finger",
    "right_inner_finger_pad", "right_inner_knuckle",
]

# --- the wrist stack, in the order it is bolted on the real robot ----------
#     tool0 -> FT 300 -> Wrist Camera -> 2F-85
#
# FT 300 force-torque sensor. Meshes are the official ros-industrial ones
# (BSD, LICENSE kept beside them); measured rather than taken from the
# datasheet:
#     coupling plate  75.0 x 74.9 x 13.0 mm
#     sensor body     75.0 x 75.0 x 34.8 mm
# The official xacro seats the sensor at z=+41.5 mm flipped 180 deg about Y,
# so the plate's 13 mm and the sensor's recess overlap and the assembly adds
# 41.5 mm along the tool axis. Reproduced here exactly.
# The coupling plate ships as robotiq_ft300-G-062-COUPLING_G-50-4M6-1D6_
# 20181119.STL and is stored here as ft300_coupling.STL, renamed on purpose.
# Isaac Sim's URDF importer turns a mesh FILENAME into a USD prim name without
# sanitising it, and "-" is not legal in one, so the hyphenated name produces a
# null prim and the ENTIRE import fails with "RuntimeError: Used null prim" --
# naming no file. (It does sanitise link and joint names, and says so in the
# log, which is what makes the omission easy to miss.) Isolated by bisection:
# swapping just this mesh for a box imports fine, renaming it imports fine,
# converting it to .obj does not.
FT300_MESH = "../robotiq/ft300/meshes"
FT300_EXT = ".STL"
FT300_SENSOR_Z = 0.0415
FT300_SENSOR_RPY = "0 3.14159265 0"

# Robotiq Wrist Camera. Robotiq ships NO public URDF or mesh for this one --
# the ROS wiki says outright that the Wrist Camera is not supported -- so it
# is a box at the documented outer dimensions, the same treatment the D435i
# gets below. 87.5 x 75 x 22.4 mm body; mounted between an FT sensor and a
# 2-finger gripper it adds 13.5 mm to the chain, which is the number that
# actually matters for where the gripper ends up.
CAM_PUCK = (0.0875, 0.075, 0.0224)
WRIST_CAM_ADDED_Z = 0.0135
# Rotation of the camera puck about the tool axis, read off the hardware
# photos: the camera's two LEDs and its lens sit in a row along the puck's
# LONG edge, and that row runs parallel to the direction the fingers open.
# ATTACH yaws the gripper 90 deg relative to tool0, which puts the finger
# opening along tool0's X -- so the puck's long edge (its own X, 87.5 mm) also
# has to lie along tool0 X, i.e. no extra yaw.
WRIST_CAM_YAW = 0.0

# Gripper sits on top of both. Both Robotiq 2F models bolt through the same
# 50 mm pattern, so the 2F-140 build's 90 deg yaw carries over.
ATTACH = {"xyz": f"0.0 0.0 {FT300_SENSOR_Z + WRIST_CAM_ADDED_Z:.4f}",
          "rpy": "0.0 0.0 1.57"}

# --- gripper articulation -------------------------------------------------
#
# cuRobo 0.7.7's Kinova model, which the links here are copied from, makes
# every 2F-85 joint FIXED. The origins are nevertheless identical to the
# ros-industrial robotiq_2f_85_gripper_visualization macro -- checked joint by
# joint -- so restoring the articulation moves nothing at angle 0; it only adds
# the axes, limits and mimics that were dropped.
#
# The real 2F-85 is two mirrored 4-bar linkages, and a 4-bar needs a loop
# closure that URDF cannot express. Every Robotiq ROS package approximates it
# with one driving joint and <mimic> followers.
#
# The joints are made revolute here, but NO <mimic> tags are written. PhysX
# refuses to build the constraint -- "the revolute joint at ... needs a finite
# limit set to be used by the mimic joint feature", though every one of them
# gets lower="0" upper="0.8757" below -- and that failure takes the whole
# articulation down with it: the fingers visibly come apart in Isaac Sim while
# cuRobo, which resolves mimics itself and never touches PhysX, shows a
# perfectly correct gripper. Verified by stripping the tags: the error count
# goes 1 -> 0 and the articulation builds.
#
# The coupling therefore lives in rig.ROBOTS[...]["gripper_joints"], which
# both the config builder and the simulator read. See HANDOVER section 6.
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 0.8

# joint -> travel MAGNITUDE (rad), upstream's numbers. The SIGN comes from
# rig's coupling: a joint with multiplier -1 travels negative, so a limit of
# lower="0" would pin it there. That is exactly what happened -- the four
# positive joints tracked a close command perfectly while both inner_finger
# joints sat at -0.000, clamped by their own limit.
GRIPPER_SPAN = {
    "left_outer_knuckle_joint": 0.8,
    "right_outer_knuckle_joint": 0.81,
    "left_inner_knuckle_joint": 0.8757,
    "right_inner_knuckle_joint": 0.8757,
    "left_inner_finger_joint": 0.8757,
    "right_inner_finger_joint": 0.8757,
}
GRIPPER_COUPLING = ROBOTS["ur5e_2f85"]["gripper_joints"]
# outer_finger and inner_finger_pad stay fixed: they are rigid offsets within
# the linkage, not degrees of freedom.

# Finger-pad height above the gripper base, summed along the 2F-85 chain:
#   base->outer_knuckle  +0.054904
#   ->outer_finger       -0.0041
#   ->inner_finger       +0.0471
#   ->pad                +0.03242
GRASP_Z = 0.054904 - 0.0041 + 0.0471 + 0.03242  # = 0.130324


def retarget_mesh(filename: str) -> str:
    """Point a donor mesh path at this project's copy of the 2F-85 meshes.

    The .dae files are swapped for .obj: Isaac Sim's asset converter segfaults
    inside tinyxml2 on these particular COLLADA files. trimesh reads them fine,
    so tools/convert_2f85_meshes.py re-exports each one as .obj, which both
    Isaac Sim and cuRobo load without complaint.
    """
    tail = filename.split("kortex_description/", 1)[-1]
    if tail.endswith(".dae"):
        tail = tail[: -len(".dae")] + ".obj"
    return f"../kinova/kortex_description/{tail}"


# Intel RealSense D435i, bolted to the side of the gripper looking along the
# tool axis. Body is 90 x 25 x 25 mm; it is real hardware that can hit things,
# so it gets collision geometry rather than being a bare frame.
# Long axis along Y, i.e. parallel to the finger opening direction, which
# is how the common 2F-85 + D435i brackets are built.
CAM_BODY = (0.025, 0.090, 0.025)

# Bolted flat against the FRONT face of the gripper, the way a real D435i
# bracket sits: offset along X (the 2F-85's fingers separate along Y and are
# thin in X, so X is the face), long axis horizontal, looking the same way the
# fingers point. It touches the gripper body by design -- camera_mount ignores
# self-collision against every gripper link, because the whole assembly is one
# rigid body, so overlap there costs nothing.
CAM_MOUNT_XYZ = (-0.050, 0.0, 0.020)

# Tilt the view back toward the tool axis (negative = toward optical +Y, the
# side the grasp point lies on). Untilted it sits 52 deg off-centre, outside
# the 34.5 deg half-frame; -0.50 rad brings it to ~8 deg.
CAM_TILT_RAD = 0.0


def _fixed_joint(robot, name, parent, child, xyz, rpy="0 0 0"):
    j = ET.SubElement(robot, "joint")
    j.set("name", name)
    j.set("type", "fixed")
    ET.SubElement(j, "parent").set("link", parent)
    ET.SubElement(j, "child").set("link", child)
    o = ET.SubElement(j, "origin")
    o.set("xyz", xyz)
    o.set("rpy", rpy)
    return j


def _mesh_link(robot, name, visual_mesh, collision_mesh):
    link = ET.SubElement(robot, "link")
    link.set("name", name)
    for tag, mesh in (("visual", visual_mesh), ("collision", collision_mesh)):
        el = ET.SubElement(link, tag)
        geom = ET.SubElement(el, "geometry")
        ET.SubElement(geom, "mesh").set("filename", mesh)
    return link


def add_wrist_stack(robot: ET.Element) -> None:
    """Bolt the FT 300 and the Wrist Camera between tool0 and the gripper.

    Order matches the hardware: tool0 -> FT 300 coupling plate -> FT 300
    sensor body -> Wrist Camera -> 2F-85. The gripper's own attach transform
    (ATTACH) already carries the summed height, so this function only has to
    place the two new bodies.

    The camera is a box because no public mesh exists for it; the FT 300 uses
    the real meshes. Both get collision geometry -- this is 5.5 cm of hardware
    between the flange and the gripper, and the planner has to know it is
    there.
    """
    # --- FT 300 -----------------------------------------------------------
    _mesh_link(
        robot, "ft300_coupling",
        f"{FT300_MESH}/visual/ft300_coupling{FT300_EXT}",
        f"{FT300_MESH}/collision/ft300_coupling{FT300_EXT}",
    )
    _fixed_joint(robot, "ft300_coupling_joint", "tool0", "ft300_coupling", "0 0 0")

    _mesh_link(
        robot, "ft300_sensor",
        f"{FT300_MESH}/visual/robotiq_ft300{FT300_EXT}",
        f"{FT300_MESH}/collision/robotiq_ft300{FT300_EXT}",
    )
    _fixed_joint(robot, "ft300_sensor_joint", "ft300_coupling", "ft300_sensor",
                 f"0 0 {FT300_SENSOR_Z:.4f}", FT300_SENSOR_RPY)

    # --- Wrist Camera -----------------------------------------------------
    puck = ET.SubElement(robot, "link")
    puck.set("name", "wrist_camera")
    for tag in ("visual", "collision"):
        el = ET.SubElement(puck, tag)
        geom = ET.SubElement(el, "geometry")
        ET.SubElement(geom, "box").set(
            "size", " ".join(f"{v:.4f}" for v in CAM_PUCK))
    # Centred on the added height, so the body straddles the joint between the
    # sensor and the gripper the way the real puck does.
    _fixed_joint(robot, "wrist_camera_joint", "tool0", "wrist_camera",
                 f"0 0 {FT300_SENSOR_Z + WRIST_CAM_ADDED_Z / 2:.4f}",
                 f"0 0 {WRIST_CAM_YAW:.4f}")


def add_wrist_camera(robot: ET.Element) -> None:
    """Add the D435i body plus its optical frame to the gripper base.

    ``camera_link`` is the OPTICAL frame, matching what cuRobo's mapper kernels
    expect: +Z forward down the view axis, +X right, +Y down.

    Isaac Sim's Camera wrapper does NOT take a raw USD orientation -- it takes
    ROS body axes, where the view direction is +X, +Y is left and +Z is up. The
    demo script carries that conversion (a 120 degree rotation, not a flip);
    measure it there rather than assuming, because an earlier 180-about-X guess
    was a no-op on the view axis and left the camera staring sideways.
    """
    mount = ET.SubElement(robot, "link")
    mount.set("name", "camera_mount")
    for tag in ("visual", "collision"):
        el = ET.SubElement(mount, tag)
        geom = ET.SubElement(el, "geometry")
        box = ET.SubElement(geom, "box")
        box.set("size", " ".join(f"{v:.4f}" for v in CAM_BODY))
    mj = ET.SubElement(robot, "joint")
    mj.set("name", "camera_mount_joint")
    mj.set("type", "fixed")
    ET.SubElement(mj, "parent").set("link", "robotiq_arg2f_base_link")
    ET.SubElement(mj, "child").set("link", "camera_mount")
    o = ET.SubElement(mj, "origin")
    o.set("xyz", " ".join(f"{v:.4f}" for v in CAM_MOUNT_XYZ))
    o.set("rpy", "0 0 0")

    ET.SubElement(robot, "link").set("name", "camera_link")
    cj = ET.SubElement(robot, "joint")
    cj.set("name", "camera_optical_joint")
    cj.set("type", "fixed")
    ET.SubElement(cj, "parent").set("link", "camera_mount")
    ET.SubElement(cj, "child").set("link", "camera_link")
    o = ET.SubElement(cj, "origin")
    o.set("xyz", f"0 0 {CAM_BODY[2] / 2:.4f}")  # lens sits on the front face
    # Yaw 90 degrees about the optical axis so image-horizontal (optical +X,
    # the 640 px side) runs along the camera body's long edge, the way a
    # landscape sensor actually sits. Without it the frame arrives rotated a
    # quarter turn. The SIGN matters and +90 gives an upside-down frame, so
    # this is -90. Roll (CAM_TILT_RAD) tips the view toward the grasp point.
    o.set("rpy", f"{CAM_TILT_RAD:.4f} 0 -1.5708")


def main():
    arm = ET.parse(ARM)
    robot = arm.getroot()
    robot.set("name", "ur5e_robotiq_2f_85")

    donor = ET.parse(DONOR).getroot()
    donor_links = {l.get("name"): l for l in donor.findall("link")}
    donor_joints = {j.get("name"): j for j in donor.findall("joint")}

    for name in GRIPPER_LINKS:
        link = copy.deepcopy(donor_links[name])
        for mesh in link.iter("mesh"):
            mesh.set("filename", retarget_mesh(mesh.get("filename")))
        robot.append(link)

    for jname, joint in donor_joints.items():
        child = joint.find("child").get("link")
        if child not in GRIPPER_LINKS:
            continue
        j = copy.deepcopy(joint)
        if jname == "gripper_base_joint":
            # Re-parent the whole gripper from the Kinova flange onto tool0.
            j.find("parent").set("link", "tool0")
            o = j.find("origin")
            o.set("xyz", ATTACH["xyz"])
            o.set("rpy", ATTACH["rpy"])
        # Restore the articulation the donor dropped, or clean up after it.
        if jname in GRIPPER_SPAN:
            j.set("type", "revolute")
            for extra in list(j.findall("limit")) + list(j.findall("axis")) \
                    + list(j.findall("mimic")):
                j.remove(extra)
            ET.SubElement(j, "axis").set("xyz", "1 0 0")
            span = GRIPPER_SPAN[jname]
            lo, hi = (0.0, span) if GRIPPER_COUPLING[jname] > 0 else (-span, 0.0)
            lim = ET.SubElement(j, "limit")
            lim.set("lower", f"{lo}"); lim.set("upper", f"{hi}")
            lim.set("velocity", "2.0"); lim.set("effort", "1000")
        elif j.get("type") == "fixed":
            # Donor left a stale <limit> on a fixed joint; drop it.
            for extra in list(j.findall("limit")) + list(j.findall("axis")):
                j.remove(extra)
        robot.append(j)

    grasp = ET.SubElement(robot, "link")
    grasp.set("name", "grasp_frame")
    gj = ET.SubElement(robot, "joint")
    gj.set("name", "grasp_joint")
    gj.set("type", "fixed")
    ET.SubElement(gj, "parent").set("link", "robotiq_arg2f_base_link")
    ET.SubElement(gj, "child").set("link", "grasp_frame")
    o = ET.SubElement(gj, "origin")
    o.set("xyz", f"0 0 {GRASP_Z:.6f}")
    o.set("rpy", "0 0 0")

    add_wrist_stack(robot)
    add_wrist_camera(robot)

    ET.indent(arm, space="  ")
    arm.write(OUT, encoding="utf-8", xml_declaration=True)
    print(f"wrote {OUT}")
    print(f"  gripper links : {len(GRIPPER_LINKS)}")
    print(f"  gripper joints: {len(GRIPPER_SPAN)} revolute, limits signed by "
          f"rig's coupling, no <mimic> (PhysX rejects it)")
    print(f"  grasp_frame   : {GRASP_Z:.6f} m above the gripper base")


if __name__ == "__main__":
    main()
