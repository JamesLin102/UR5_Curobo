"""Clip revolute joint limits in a URDF.

The stock UR5e description gives five of six joints a +/-360 degree range. With
720 degrees to play in, the planner is free to pick IK branches that wind a
wrist right round: collision-free and inside limits, so perfectly legal, but
ugly to watch and occasionally leaving the arm somewhere the next target cannot
be reached from. Real cells restrict these anyway for cable management.

Keeps a .orig backup the first time it touches a file, so the baseline stays
recoverable.

Run:
    python tools/clip_joint_limits.py assets/robot/ur_description/ur5e.urdf --deg 180
"""

import argparse
import math
import os
import shutil
import xml.etree.ElementTree as ET


def clip(urdf_path: str, limit_deg: float, skip=()) -> None:
    backup = urdf_path + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(urdf_path, backup)
        print(f"  backed up -> {os.path.basename(backup)}")

    tree = ET.parse(urdf_path)
    lim_rad = math.radians(limit_deg)
    changed = []
    for joint in tree.getroot().findall("joint"):
        if joint.get("type") != "revolute" or joint.get("name") in skip:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lo, hi = float(limit.get("lower")), float(limit.get("upper"))
        new_lo, new_hi = max(lo, -lim_rad), min(hi, lim_rad)
        if (new_lo, new_hi) != (lo, hi):
            limit.set("lower", f"{new_lo:.6f}")
            limit.set("upper", f"{new_hi:.6f}")
            changed.append(
                f"    {joint.get('name'):<22s} "
                f"{math.degrees(lo):7.1f}/{math.degrees(hi):<7.1f} -> "
                f"{math.degrees(new_lo):7.1f}/{math.degrees(new_hi):<7.1f}"
            )

    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    print(f"  {os.path.basename(urdf_path)}: {len(changed)} joints clipped to +/-{limit_deg:g} deg")
    for line in changed:
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf", nargs="+")
    ap.add_argument("--deg", type=float, default=180.0)
    args = ap.parse_args()
    for path in args.urdf:
        clip(path, args.deg)
