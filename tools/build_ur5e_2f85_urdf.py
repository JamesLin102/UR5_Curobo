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
import xml.etree.ElementTree as ET

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

# Same coupling as the 2F-140 build: both Robotiq 2F models bolt to tool0
# through the identical 50 mm pattern, so reuse that URDF's attach transform.
ATTACH = {"xyz": "0.0 0.0 0.0", "rpy": "0.0 0.0 1.57"}

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
        # Donor left a stale <limit> on a fixed joint; drop it.
        if j.get("type") == "fixed":
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

    add_wrist_camera(robot)

    ET.indent(arm, space="  ")
    arm.write(OUT, encoding="utf-8", xml_declaration=True)
    print(f"wrote {OUT}")
    print(f"  gripper links : {len(GRIPPER_LINKS)}")
    print(f"  grasp_frame   : {GRASP_Z:.6f} m above the gripper base")


if __name__ == "__main__":
    main()
