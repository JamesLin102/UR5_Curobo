"""A scenes.SceneSpec, spawned as Isaac Lab assets under every environment.

    handles = spawn_cell(scene, spec, cell_cfg)     # in DirectRLEnv._setup_scene

Everything that belongs to a cell is spawned under the environment namespace,
so N cells are N copies and each one keeps the scene's own coordinates
relative to its env origin:

    /World/envs/env_<i>/Robot                   the arm (rig.ROBOTS)
    /World/envs/env_<i>/obstacles/<name>        static colliders the planner knows
    /World/envs/env_<i>/<name>                  unmapped bodies: rendered, no collision
    /World/envs/env_<i>/<name>                  payloads: rigid bodies
    /World/envs/env_<i>/Robot/<body>/<camera>   cameras on a robot frame ("link")
    /World/envs/env_<i>/<camera>_cam            fixed cameras ("pose")

Shared by all of them: the ground, the light, the grip material.

Each kind of SceneSpec entry has one builder, and cameras are built by kind
(CAMERA_BUILDERS), so a scene that leaves out an optional entry, or adds a
camera of an existing kind, needs nothing here.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg, TiledCameraCfg

from rig import ROBOTS
from sim_usd import (GRIP_MATERIAL, GRIP_MATERIAL_PATH, apply_linkage, apply_urdf_colors,
                     bind_pad_material, body_ancestor)
from urdf_frames import split_transform

from .robots import ROOT, make_robot_cfg

ENV_NS = "/World/envs/env_.*"
GROUND = "/World/ground"

# The USD camera's default horizontal aperture, in the same units as its focal
# length. Only their ratio matters: it fixes the horizontal field of view.
APERTURE = 20.955


@dataclass
class CellHandles:
    """What _setup_scene made, for the cell to drive once physics is running."""
    robot: Articulation
    urdf: str
    tool: Tuple[str, np.ndarray, np.ndarray]          # (body, offset pos, offset quat wxyz)
    payload: Dict[str, RigidObject] = field(default_factory=dict)
    unmapped: Dict[str, str] = field(default_factory=dict)     # name -> prim path pattern
    cameras: Dict[str, Camera] = field(default_factory=dict)
    camera_mounts: Dict[str, Tuple[str, np.ndarray]] = field(default_factory=dict)  # link cams
    pins: List[str] = field(default_factory=list)     # loop closures made, per env robot
    pads: List[str] = field(default_factory=list)     # pad links that got the grip material


def _colour(rgb):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(c) for c in rgb))


def _spawn_obstacles(spec):
    """Static colliders: a collision shape and no rigid body, like FixedCuboid."""
    for env in sim_utils.find_matching_prim_paths(ENV_NS):
        sim_utils.create_prim(f"{env}/obstacles", "Xform")
    for name, dims, pose, rgb in spec.obstacles:
        cfg = sim_utils.CuboidCfg(size=tuple(dims), visual_material=_colour(rgb),
                                  collision_props=sim_utils.CollisionPropertiesCfg())
        cfg.func(f"{ENV_NS}/obstacles/{name}", cfg,
                 translation=tuple(pose[:3]), orientation=tuple(pose[3:]))


def _spawn_unmapped(spec):
    """Bodies only the simulator knows: rendered into depth, never collided with."""
    out = {}
    for name, dims, pose, rgb in spec.unmapped:
        if spec.shape(name) == "cylinder":
            cfg = sim_utils.CylinderCfg(radius=float(dims[0]) / 2, height=float(dims[2]),
                                        axis="Z", visual_material=_colour(rgb))
        else:
            cfg = sim_utils.CuboidCfg(size=tuple(dims), visual_material=_colour(rgb))
        cfg.func(f"{ENV_NS}/{name}", cfg,
                 translation=tuple(pose[:3]), orientation=tuple(pose[3:]))
        out[name] = f"{ENV_NS}/{name}"
    return out


def _spawn_payload(spec):
    """Rigid bodies to pick up, sharing the grip material with the finger pads."""
    out = {}
    for name, dims, pose, rgb, mass in spec.payload:
        cfg = RigidObjectCfg(
            prim_path=f"{ENV_NS}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=tuple(dims),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=float(mass)),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(**GRIP_MATERIAL),
                physics_material_path=GRIP_MATERIAL_PATH,
                visual_material=_colour(rgb),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pose[:3]),
                                                      rot=tuple(pose[3:])),
        )
        out[name] = RigidObject(cfg)
    return out


# --- cameras, by kind ----------------------------------------------------------


def _camera_cfg(cell, cam, prim_path, pos, rot):
    """Depth camera matching the SceneSpec entry. The pose is an OPTICAL frame.

    Isaac Lab's "ros" convention is +Z forward, +X right, +Y down -- exactly
    the optical frame the scene (and cuRobo's mapper) uses, so the scene's
    pose goes in as it is.
    """
    focal = APERTURE / (2.0 * math.tan(math.radians(cam["horizontal_fov_deg"]) / 2.0))
    kind = {"camera": CameraCfg, "tiled": TiledCameraCfg}[cell.camera_class]
    return kind(
        prim_path=prim_path,
        width=int(cam["width"]),
        height=int(cam["height"]),
        data_types=["distance_to_image_plane"] + (["rgb"] if cam.get("rgb") else []),
        update_period=0.0,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal, horizontal_aperture=APERTURE,
            clipping_range=(float(cam["near"]), float(cam["far"]))),
        offset=kind.OffsetCfg(pos=tuple(float(v) for v in pos),
                              rot=tuple(float(v) for v in rot), convention="ros"),
    )


def _link_camera(name, cam, ctx):
    """On a robot frame. The frame may have been merged into a body: find which."""
    body, T = body_ancestor(ctx["stage"], ctx["robot0"], cam["link"], ctx["urdf"])
    pos, rot = split_transform(T)
    ctx["mounts"][name] = (body, T)
    return _camera_cfg(ctx["cell"], cam, f"{ENV_NS}/Robot/{body}/{name}", pos, rot)


def _fixed_camera(name, cam, ctx):
    """Fixed in the cell, at the scene's pose relative to the env origin."""
    if not ctx["cell"].overhead:
        return None
    pose = cam["pose"]
    return _camera_cfg(ctx["cell"], cam, f"{ENV_NS}/{name}_cam", pose[:3], pose[3:])


# SceneSpec camera kind (the key that says where its pose comes from) -> builder.
CAMERA_BUILDERS = {
    "link": _link_camera,
    "pose": _fixed_camera,
}


def _spawn_cameras(spec, ctx):
    cams = {}
    for name, cam in spec.cameras.items():
        kind = next(k for k in CAMERA_BUILDERS if k in cam)
        cfg = CAMERA_BUILDERS[kind](name, cam, ctx)
        if cfg is not None:
            cams[name] = cfg.class_type(cfg)
    return cams


# --- the cell --------------------------------------------------------------------


def spawn_cell(scene, spec, cell):
    """Spawn one cell per environment. Call from DirectRLEnv._setup_scene.

    Order matters: the robot first (cameras hang off its bodies), cloning next,
    and the USD edits last, on every environment's robot, so they are there
    whether each environment is a copy or a replica of env_0.
    """
    robot_spec = ROBOTS[cell.robot]
    urdf = os.path.join(ROOT, robot_spec["urdf"])
    stage = sim_utils.get_current_stage()

    robot = Articulation(make_robot_cfg(cell.robot, cell, spec.home)
                         .replace(prim_path=f"{ENV_NS}/Robot"))
    robot0 = f"{scene.env_prim_paths[0]}/Robot"
    tool_body, T = body_ancestor(stage, robot0, robot_spec["tool_frame"], urdf)
    tool = (tool_body, *split_transform(T))

    ground = sim_utils.GroundPlaneCfg(color=None)
    ground.func(GROUND, ground)
    light = sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0)
    light.func("/World/DistantLight", light)

    _spawn_obstacles(spec)
    unmapped = _spawn_unmapped(spec)
    payload = _spawn_payload(spec)

    handles = CellHandles(robot=robot, urdf=urdf, tool=tool, payload=payload,
                          unmapped=unmapped)
    if cell.mapping:
        ctx = dict(stage=stage, robot0=robot0, urdf=urdf, cell=cell,
                   mounts=handles.camera_mounts)
        handles.cameras = _spawn_cameras(spec, ctx)

    if scene.cfg.replicate_physics:
        scene.clone_environments(copy_from_source=False)

    for path in sim_utils.find_matching_prim_paths(f"{ENV_NS}/Robot"):
        apply_urdf_colors(stage, path, urdf)
        handles.pins = apply_linkage(stage, path, robot_spec["gripper"], urdf, cell.robot)
        if payload:
            handles.pads = bind_pad_material(stage, path, robot_spec["gripper"]["pad_links"])

    scene.articulations["robot"] = robot
    for name, obj in payload.items():
        scene.rigid_objects[name] = obj
    # The cameras are deliberately NOT registered in scene.sensors; the cell
    # updates them itself. Registered, scene.reset() would call Camera.reset(),
    # which re-reads the camera's pose through an XformPrimView -- and on the
    # GPU pipeline that view's first read copies the prim's USD world
    # transform into Fabric. USD is not updated as the robot moves, so a
    # camera riding a robot link is then rendered with a stale offset for as
    # long as it stays. Measured: after the first env.reset() on cuda:0 the
    # wrist depth sat 34-45 mm off forward kinematics for a whole episode and
    # the map filled with phantom obstacles; on CPU the view reads USD
    # directly and nothing is copied. Nothing here needs what reset() does
    # (frame counters, the pose it re-reads).
    if scene.device == "cpu" or (not scene.cfg.replicate_physics and scene.cfg.filter_collisions):
        scene.filter_collisions([GROUND])
    return handles
