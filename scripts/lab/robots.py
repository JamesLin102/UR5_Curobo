"""Isaac Lab robots, derived from rig.ROBOTS: one function for every robot.

    cfg = make_robot_cfg("ur5_robotiq", CellCfg(), home=scene.home)
    cfg = cfg.replace(prim_path="/World/envs/env_.*/Robot")

Nothing here knows which robot it is building. The URDF, the joints, how they
are driven, which of them a loop closure owns: all of it is read from the
rig entry, so a second robot is a second entry there and no edit here.

Two things differ from the Isaac Sim side's URDF import and are handled here:

  merging   Isaac Lab's converter pins URDF importer 2.4.31 and tells it to
            merge fixed links WITH mass into their parent too. The Isaac Sim
            side's bundled 2.4.30 merges only massless frames. The converter
            below makes that choice a config flag (CellCfg.merge_inertial),
            defaulting to the Isaac Sim side's, so both backends simulate the
            same bodies.
  gains     rig.ROBOTS stores drive gains as USD writes them, per DEGREE for a
            revolute joint. Isaac Lab actuators take SI, per radian, and
            convert back on the way into USD. The factor is DEG below; without
            it the arm would be 57x softer than on the Isaac Sim side.
            And they are gains for the drive TYPE the rig names (the Isaac Sim
            importer's "acceleration"), which the converter is told as well:
            Isaac Lab's own default, "force", reads the same numbers as a
            different, far stiffer robot.
"""

import math
import os
from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverter
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.utils import configclass

from rig import DEFAULT_ROBOT, ROBOTS

from .cell_cfg import CellCfg

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEG = 180.0 / math.pi   # USD angular drive gains are per degree; SI is per radian


class RigUrdfConverter(UrdfConverter):
    """UrdfConverter with the merge of links-with-mass as a choice, not a given."""

    def _get_urdf_import_config(self):
        config = super()._get_urdf_import_config()
        config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_ignore_inertia)
        return config


@sim_utils.clone
def spawn_from_rig_urdf(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """isaaclab's spawn_from_urdf, through RigUrdfConverter."""
    converter = RigUrdfConverter(cfg)
    return _spawn_from_usd_file(prim_path, converter.usd_path, cfg, translation, orientation)


@configclass
class RigUrdfFileCfg(sim_utils.UrdfFileCfg):
    func: Callable = spawn_from_rig_urdf
    # A field rather than a converter argument, so it is part of the config
    # hash that decides whether the cached USD is still current.
    merge_fixed_ignore_inertia: bool = False


def drive_groups(robot_key):
    """{group: (joint names, stiffness, damping)} in rig units (per degree).

    arm        the planned joints
    gripper    the gripper joints something DRIVES
    followers  the gripper joints a loop closure owns, if the rig names any:
               released (0 / 0) so the drive does not fight the pin
    """
    spec = ROBOTS[robot_key]
    arm, grip = spec["arm"], spec["gripper"]
    follower = grip.get("pinned_follower")
    followers = [j for j in grip["joints"] if follower and follower in j]
    driven = [j for j in grip["joints"] if j not in followers]
    groups = {
        "arm": (list(arm["joints"]), arm["drive_stiffness"], arm["drive_damping"]),
        "gripper": (driven, grip["drive_stiffness"], grip["drive_damping"]),
    }
    if followers:
        groups["followers"] = (followers, 0.0, 0.0)
    return groups


def usd_dir(robot_key, cell):
    """Where the converted USD is cached: per robot, per merge mode.

    Outside the repository: the checkout may sit on a FUSE mount that leaves
    .fuse_hidden files behind whenever the converter rewrites a file.
    """
    mode = "merged_inertial" if cell.merge_inertial else "merged_frames"
    return os.path.join(os.path.expanduser(cell.usd_dir), robot_key, mode)


def make_robot_cfg(robot_key, cell, home=None):
    """ArticulationCfg for rig.ROBOTS[robot_key], resting at `home` (arm joints)."""
    spec = ROBOTS[robot_key]
    groups = drive_groups(robot_key)
    # Exact-name patterns: the converter matches gain keys with re.search, so
    # a bare name would also hit every joint it is a substring of.
    stiffness = {f"^{j}$": k * DEG for joints, k, _ in groups.values() for j in joints}
    damping = {f"^{j}$": d * DEG for joints, _, d in groups.values() for j in joints}

    arm_joints = groups["arm"][0]
    grip = spec["gripper"]
    joint_pos = {j: grip["open"] * m for j, m in grip["joints"].items()}
    if home is not None:
        if len(home) != len(arm_joints):
            raise ValueError(f"home has {len(home)} values, {robot_key} has "
                             f"{len(arm_joints)} arm joints")
        joint_pos.update(dict(zip(arm_joints, map(float, home))))

    pos_iters, vel_iters = spec["solver_iterations"]
    return ArticulationCfg(
        spawn=RigUrdfFileCfg(
            asset_path=os.path.join(ROOT, spec["urdf"]),
            usd_dir=usd_dir(robot_key, cell),
            usd_file_name=f"{robot_key}.usd",
            force_usd_conversion=cell.force_usd,
            fix_base=True,
            merge_fixed_joints=True,
            merge_fixed_ignore_inertia=cell.merge_inertial,
            self_collision=False,
            collider_type="convex_hull",
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type=spec["drive_type"],
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=stiffness, damping=damping),
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=pos_iters,
                solver_velocity_iteration_count=vel_iters,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=joint_pos),
        actuators={
            name: ImplicitActuatorCfg(joint_names_expr=joints,
                                      stiffness=k * DEG, damping=d * DEG)
            for name, (joints, k, d) in groups.items()
        },
    )


# The Isaac Lab asset-constant idiom, for code that wants the robot on its own
# in a scene of its own: the default robot with the default CellCfg, arm at
# zero. A cell builds its robot through make_robot_cfg instead, from the
# scene's HOME.
UR5_ROBOTIQ_CFG = make_robot_cfg(DEFAULT_ROBOT, CellCfg())
