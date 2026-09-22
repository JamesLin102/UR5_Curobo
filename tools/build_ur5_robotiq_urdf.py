"""Turn the upstream UR5 + Robotiq + D435i description into this project's URDF.

The input, `assets/robot/ur5_robotiq/ur5_robotiq.raw.urdf`, is xacro's own
output for `ur5_robotiq.urdf.xacro` -- the mir_ur5_humble stack with the MiR
chassis left out. Regenerating it needs the upstream ROS packages on disk, so
it is vendored; see assets/robot/ur5_robotiq/PROVENANCE.md.

Nothing here touches geometry. Every edit below is mechanical, and that is
deliberate: the model this replaces was assembled by hand from photographs and
its link positions were wrong. If a number in the chain looks off, it is wrong
upstream, not here.

Run:
    python tools/build_ur5_robotiq_urdf.py
"""

import math
import os
import xml.etree.ElementTree as ET

import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = f"{ROOT}/assets/robot/ur5_robotiq"
RAW = f"{ASSETS}/ur5_robotiq.raw.urdf"
OUT = f"{ASSETS}/ur5_robotiq.urdf"

# Upstream namespaces the arm with "ur_". Dropping it gives the joint names
# every other UR config in this project already uses (shoulder_pan_joint and
# friends), so scenes and saved poses carry over unchanged.
ARM_PREFIX = "ur_"

# The 2F-85 linkage, as multipliers of the one driven joint. Read off the
# <mimic> tags in the raw URDF -- which are then removed, because PhysX
# refuses to build a mimic constraint and takes the whole articulation down
# with it when it fails.
LEADER = "robotiq_85_left_knuckle_joint"
COUPLING = {
    "robotiq_85_left_knuckle_joint": 1.0,
    "robotiq_85_right_knuckle_joint": -1.0,
    "robotiq_85_left_inner_knuckle_joint": 1.0,
    "robotiq_85_right_inner_knuckle_joint": -1.0,
    "robotiq_85_left_finger_tip_joint": -1.0,
    "robotiq_85_right_finger_tip_joint": 1.0,
}
SPAN = 0.8  # rad, the leader's own upper limit; 0 is fully open

# Grasp frame: on the tool axis, level with the middle of the finger pads.
# Measured off left_finger_tip.stl -- its inner face spans 166.3..204.3 mm
# from tool0, so the centre is 185.3, which is 130.324 mm above the gripper
# base. (The hand-built model put it at the same place, so grasp poses in the
# scenes do not move.)
GRASP_Z = 0.130324
GRASP_PARENT = "robotiq_85_base_link"

# cuRobo asks for the camera by one link name. The D435i's colour optical
# frame already uses the optical convention cuRobo expects (+Z forward, +X
# right, +Y down), so this is an alias, not a transform.
CAMERA_PARENT = "realsense_color_optical_frame"

# The Robotiq coupling mesh is mounted back to front upstream (and in
# ROS-Industrial's own FT 300 macro, which places it at identity on the
# flange). Measured off the STL, the part is stepped:
#
#     z  0.0 -  4.0   dia 31.5   locating spigot
#     z  4.0 -  6.5   dia 75     collar, the FT 300's own diameter
#     z  7.0 - 13.0   dia 63     body, the UR flange's own diameter
#
# so dia 63 is the robot face and the spigot locates the SENSOR, not the arm.
# Drawn as-is it puts the dia 75 collar against a dia 63 flange -- a 6 mm
# overhang right under the wrist -- and buries the body 6.3 mm inside the
# sensor. Flipped, the faces match on both sides and the stack-up closes:
# body 0..6.5, sensor mesh starts at 6.7.
#
# Only the drawn geometry moves. The link frame stays on the flange face and
# ft300_sensor stays 41.5 mm from it, which is ROS-Industrial's figure and not
# ours to change, so no kinematics change.
FLIP_VISUAL = {"robotiq_ft300_mounting_plate": (0.013, "0 3.14159265 0")}

# cuRobo's sphere fitter finds a link's interior with a Warp signed-distance
# query, which needs a mesh it can sign. The D435i bracket is not watertight,
# so every grid point reads as outside, the fit finds nothing, and it falls
# back to scattering 2 mm beads over the surface -- spheres far smaller than
# the hardware, which the planner would believe. Its convex hull is watertight
# and fits cleanly. A hull can only over-cover, never under-cover, and this
# one is a lump bolted beside the gripper where a few extra millimetres of
# clearance cost nothing. Visual geometry is untouched: the viewer still shows
# the real bracket next to the spheres.
HULL_COLLISION = ["camera_mount"]
HULL_DIR = "hull"

# Five of the six UR joints ship with a +/-360 degree range. With 720 degrees
# to play in the planner picks IK branches that wind a wrist right round:
# legal, ugly to watch, and it sometimes leaves the arm somewhere the next
# target cannot be reached from. Real cells restrict these for cable
# management anyway. The gripper joints are far inside this and untouched.
#
# This was a separate step in the old build (tools/clip_joint_limits.py, run
# by hand between the two builders); folded in here so regenerating cannot
# lose it.
JOINT_LIMIT_DEG = 180.0

# Isaac Sim's URDF importer derives USD prim names from these and does not
# sanitise them; a hyphen aborts the whole import with "Used null prim".
BAD = "-"


def rename(name: str) -> str:
    if name.startswith(ARM_PREFIX):
        name = name[len(ARM_PREFIX):]
    return name.replace(BAD, "_")


def main():
    tree = ET.parse(RAW)
    robot = tree.getroot()

    # Simulator and controller plumbing. cuRobo ignores it, Isaac Sim ignores
    # it, and it refers to hardware that is not part of this project.
    dropped = 0
    for tag in ("ros2_control", "gazebo", "transmission"):
        for el in robot.findall(tag):
            robot.remove(el)
            dropped += 1

    for el in robot.iter():
        if el.tag in ("link", "joint") and el.get("name"):
            el.set("name", rename(el.get("name")))
        if el.tag in ("parent", "child") and el.get("link"):
            el.set("link", rename(el.get("link")))
        if el.tag == "mesh" and el.get("filename"):
            f = el.get("filename")
            # package://<pkg>/... maps to <pkg>/... because the asset tree is
            # laid out by package name. The RealSense body comes through as a
            # file:// URL instead, because its xacro was included by path.
            if f.startswith("package://"):
                f = f[len("package://"):]
            elif "realsense2_description/" in f:
                f = "realsense2_description/" + f.split("realsense2_description/")[-1]
            el.set("filename", f)

    links = {l.get("name"): l for l in robot.findall("link")}
    for name, (dz, rpy) in FLIP_VISUAL.items():
        for el in links[name].findall("visual"):
            if el.find("geometry/mesh") is None:
                continue
            o = el.find("origin")
            if o is None:
                o = ET.SubElement(el, "origin")
            xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
            xyz[2] += dz
            o.set("xyz", " ".join(f"{v:.6f}" for v in xyz))
            o.set("rpy", rpy)
        print(f"  turned {name} around; robot face now on the flange")

    for name in HULL_COLLISION:
        link = links[name]
        hull = None
        for col in link.findall("collision"):
            m = col.find("geometry/mesh")
            if m is None:
                continue
            mesh = trimesh.load(os.path.join(ASSETS, m.get("filename")), force="mesh")
            if m.get("scale"):
                mesh.apply_scale([float(v) for v in m.get("scale").split()])
            hull = mesh.convex_hull if hull is None else trimesh.util.concatenate(
                [hull, mesh.convex_hull]).convex_hull
            link.remove(col)
        if hull is None:
            raise SystemExit(f"{name} has no collision mesh to hull")
        rel = f"{HULL_DIR}/{name}.stl"
        os.makedirs(f"{ASSETS}/{HULL_DIR}", exist_ok=True)
        hull.export(f"{ASSETS}/{rel}")
        col = ET.SubElement(link, "collision")
        ET.SubElement(ET.SubElement(col, "geometry"), "mesh").set("filename", rel)
        print(f"  hulled {name} -> {rel} "
              f"({len(hull.faces)} faces, {hull.volume * 1e6:.0f} cm3)")

    joints = {j.get("name"): j for j in robot.findall("joint")}

    # The linkage: every joint driven explicitly, limits signed by its
    # multiplier so the simulator cannot drive one the wrong way.
    lead_limit = joints[LEADER].find("limit")
    effort, velocity = lead_limit.get("effort"), lead_limit.get("velocity")
    for name, mult in COUPLING.items():
        j = joints[name]
        j.set("type", "revolute")
        for m in j.findall("mimic"):
            j.remove(m)
        limit = j.find("limit")
        if limit is None:
            limit = ET.SubElement(j, "limit")
        lo, hi = sorted((0.0, SPAN * mult))
        limit.set("lower", f"{lo:.4f}")
        limit.set("upper", f"{hi:.4f}")
        limit.set("effort", effort)
        limit.set("velocity", velocity)

    clipped = []
    lim = math.radians(JOINT_LIMIT_DEG)
    for j in robot.findall("joint"):
        limit = j.find("limit")
        if j.get("type") != "revolute" or limit is None:
            continue
        lo, hi = float(limit.get("lower")), float(limit.get("upper"))
        new = (max(lo, -lim), min(hi, lim))
        if new != (lo, hi):
            limit.set("lower", f"{new[0]:.6f}")
            limit.set("upper", f"{new[1]:.6f}")
            clipped.append(j.get("name"))

    def frame(name, parent, xyz="0 0 0"):
        ET.SubElement(robot, "link").set("name", name)
        j = ET.SubElement(robot, "joint")
        j.set("name", f"{name}_joint")
        j.set("type", "fixed")
        ET.SubElement(j, "parent").set("link", parent)
        ET.SubElement(j, "child").set("link", name)
        o = ET.SubElement(j, "origin")
        o.set("xyz", xyz)
        o.set("rpy", "0 0 0")

    frame("grasp_frame", GRASP_PARENT, f"0 0 {GRASP_Z:.6f}")
    frame("camera_link", CAMERA_PARENT)

    names = [el.get("name") for el in robot.iter() if el.tag in ("link", "joint")]
    assert len(names) == len(set(names)), "renaming collided"
    assert not any(BAD in n for n in names), "a hyphen survived"

    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="utf-8", xml_declaration=True)
    print(f"wrote {OUT}")
    print(f"  dropped {dropped} ros2_control/gazebo/transmission blocks")
    print(f"  links {len(robot.findall('link'))}, joints {len(robot.findall('joint'))}")
    print(f"  {len(COUPLING)} gripper joints, revolute, no <mimic>; "
          f"leader {rename(LEADER)}")
    print(f"  grasp_frame {GRASP_Z:.6f} m above {GRASP_PARENT}")
    print(f"  camera_link aliases {CAMERA_PARENT}")
    print(f"  clipped {len(clipped)} joints to +/-{JOINT_LIMIT_DEG:.0f} deg: "
          f"{', '.join(clipped)}")


if __name__ == "__main__":
    main()
