"""Convert a cuRobo v1 (0.7.x) robot yml to the v2 (0.8.x) schema."""
import sys, yaml

DROP = {  # v1-only keys that v2's KinematicsLoaderCfg does not accept
    "usd_path", "usd_robot_root", "isaac_usd_path",
    "usd_flip_joints", "usd_flip_joint_limits",
    "ee_link", "link_names",
}
CSPACE_RENAME = {"retract_config": "default_joint_position"}

def convert(src, dst):
    d = yaml.safe_load(open(src))
    k = d["robot_cfg"]["kinematics"]
    ee = k.get("ee_link")
    extra = k.get("link_names") or []
    for key in DROP:
        k.pop(key, None)
    # v2 replaces ee_link/link_names with an explicit tool_frames list
    k["tool_frames"] = [ee] + [l for l in extra if l != ee] if ee else None
    cs = k.get("cspace", {}) or {}
    for old, new in CSPACE_RENAME.items():
        if old in cs:
            cs[new] = cs.pop(old)
    k["format_version"] = 2.0
    yaml.safe_dump(d, open(dst, "w"), sort_keys=False, default_flow_style=None)
    print(f"converted {src} -> {dst}\n  tool_frames = {k['tool_frames']}")

convert(sys.argv[1], sys.argv[2])
