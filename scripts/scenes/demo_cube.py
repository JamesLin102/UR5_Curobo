"""The shipped demo: one slab the cameras have to discover for themselves.

Two targets on opposite sides of a cell, and an obstacle between them that the
planner is never told about. Avoidance of it is therefore proof that the live
map is driving the plan -- see HANDOVER section 7 for the measurements, and
scenes/base.py for what a scene may define.

Copy this file to add your own scene, then run both processes with
``--scene <your module name>``.
"""

from .base import SceneSpec, WatchBox


# Rest pose. shoulder_lift and elbow are raised relative to the obvious
# [0, -2.2, 1.9, ...]: with the FT 300 and the Wrist Camera in the stack the
# tool sits 55 mm further out, and that pose put grasp_frame at z=0.350 --
# exactly the top face of the slab below, at x=0.256 inside its 0.24..0.36
# span. Measured clearance of the nearest collision sphere to the slab was
# -11 mm, i.e. the gripper started inside it. This pose clears by +140 mm with
# the tool at z=0.550. The wrist joints are untouched, so the tool still
# points down.
HOME = [0.0, -1.8, 1.25, -1.383, -1.57, 0.0]

# name, dims (full extents, m), pose [x, y, z, qw, qx, qy, qz], rgb
# Deliberately sparse. An earlier version had a wall at z<=0.70 under a shelf
# at z=0.755, leaving a 5.5 cm corridor -- fine for a planner working from
# exact cuboids, but mapped surfaces are inflated by the ESDF resolution, so
# that corridor closed the moment the map got dense and every plan failed.
# Keep the cell open and let the draggable cube be the only real obstacle.
OBSTACLES = [
    ("table", [2.00, 2.00, 0.10], [0.00, 0.0, -0.06, 1, 0, 0, 0], (0.45, 0.47, 0.50)),
]

# Two targets on opposite sides of the wall, tool pointing down.
TARGETS = [
    [0.45, -0.32, 0.25, 0.0, 1.0, 0.0, 0.0],
    [0.45,  0.32, 0.25, 0.0, 1.0, 0.0, 0.0],
]

# A cube that moves through the cell while the sim runs (see CUBE_SWEEP). The
# planner is never told where it is: the wrist camera has to find it, so a
# change in the planned route when the cube arrives is the proof that the live
# map is driving collision avoidance. It is a visual prim, not a physics body,
# so it can also be dragged by hand in the viewport without fighting PhysX --
# which does mean the arm passes through it rather than hitting it.
DRAG_CUBE = (
    # A low, wide slab standing on the table, not a tall post. The shape was
    # measured, not guessed, because two constraints fight each other:
    #   * the camera only maps up to roughly its own altitude (~0.40 m here),
    #     so anything above that line is invisible -- raising the obstacle from
    #     0.50 m to 0.70 m added exactly zero voxels inside its bounding box;
    #   * with EXACT geometry, an obstacle here only diverts the route above
    #     ~0.45 m, and then only if it is narrow.
    # A 0.35 m slab is entirely inside what the camera can see while still
    # blocking laterally. x=0.30 sits in the middle of the camera's scan band.
    #
    # The MAPPED obstacle is fatter than the real one (ESDF cell size plus the
    # planner's collision activation distance), so size still matters. Measured
    # live with both cameras, 120 s of planning each:
    #   0.11 x 0.31 x 0.33   no effect   52 clear routes at 81 waypoints
    #   0.12 x 0.35 x 0.35   DETOUR      43 consecutive routes at 121, 0 failed
    # and with no obstacle at all, 53 consecutive 81s. So this size diverts the
    # route every single cycle without ever blocking it.
    #
    # It used to block instead, which is why earlier notes describe the window
    # as unusably narrow. That was not the obstacle: the TARGET MARKERS were
    # being fused into the map as obstacles sitting on the goals (see
    # build_stage, and HANDOVER section 7). Every size measured before that fix
    # was really measuring the markers.
    "drag_me", [0.12, 0.35, 0.35], [0.30, 0.0, 0.175, 1, 0, 0, 0], (0.62, 0.20, 0.18)
)

# The cube slides along y on its own, so the test does not depend on dragging it
# by hand. Both ends of the sweep sit inside the band the wrist camera actually
# scans during the cycle -- measured as x 0.30..0.45, y -0.45..+0.30 -- because
# anything outside that is in a blind spot and never reaches the map.
CUBE_SWEEP = {
    # Detour / wait / resume along one pass was measured with an EARLIER cube
    # (0.50 m tall) at x=0.38, not with the DRAG_CUBE above -- the sweep has not
    # been re-measured since the slab replaced it, so treat those thresholds as
    # historical.
    #
    # Note x=0.38 differs from DRAG_CUBE's parked x=0.30: --move-cube therefore
    # carries the cube out of the nominal bounding box that planner_server uses
    # for its INSIDE-CUBE voxel count, so that number only means what it says
    # while the cube is parked (the default, and what the A/B test uses).
    "x": 0.38,
    "y_from": -0.05,
    "y_to": 0.55,
    "period_s": 40.0,
}

# A short sweep at startup so the map has content before the first plan.
# Joint-space poses (rad), kept high and away from the table.
SCAN_POSES = [
    HOME,
    [-0.55, -1.85,  1.35, -1.300, -1.57, 0.0],
    [ 0.00, -1.70,  1.20, -1.250, -1.57, 0.0],
    [ 0.55, -1.85,  1.35, -1.300, -1.57, 0.0],
    HOME,
]

# Cameras feeding the map, by name. Each says how its pose is obtained, and
# exactly one of the two keys must be present:
#
#   "link"  a frame in the robot's own URDF. The planner server derives the
#           pose by forward kinematics from the joint state it is sent, so
#           nothing has to ship camera poses over the socket.
#   "pose"  fixed in the world, [x, y, z, qw, qx, qy, qz]. This is the OPTICAL
#           frame, not the camera body -- see the axis note on "overhead".
#
# All of them are cuRobo OPTICAL frames: +Z forward along the view, +X image
# right, +Y image down. Isaac Sim's Camera wrapper wants ROS BODY axes instead
# (+X view, +Y left, +Z up), so the demo converts and then verifies the result
# against the prim's actual transform rather than trusting the quaternion.
CAMERAS = {
    # Wrist RealSense D435i, as rendered in Isaac Sim.
    "wrist": {
        "width": 640,
        "height": 480,
        "horizontal_fov_deg": 69.0,   # D435i depth FOV at 4:3
        "near": 0.15,
        "far": 3.0,
        "link": "camera_link",        # optical frame defined in the URDF
    },
    # Fixed camera on a gantry above the cell, looking straight down.
    #
    # A wrist camera fundamentally cannot support cross-cell avoidance: its
    # coverage is whatever the arm happens to sweep, it never sees above its
    # own altitude, and the measured table footprint over a full cycle is only
    # x 0.30..0.45, y -0.45..+0.30. This one sees the whole cell at once, so
    # obstacle height and placement stop having to be chosen to suit it.
    #
    # Orientation, derived rather than guessed. Looking straight down means the
    # optical +Z (view) is world -Z. Picking image-right (+X) = world +Y then
    # forces image-down (+Y) = world +X, which is right-handed:
    #     X x Y = (0,1,0) x (1,0,0) = (0,0,-1) = Z  ok
    # That basis is a 180 degree rotation about (1,1,0)/sqrt(2), i.e.
    # (w,x,y,z) = (0, sqrt(2)/2, sqrt(2)/2, 0). So image rows run along world
    # +X (away from the robot base) and columns along world +Y.
    #
    # Height 1.20 m gives a footprint at table level of
    #     along y: 2 * 1.20 * tan(34.5 deg) = 1.65 m  ->  y -0.82..+0.82
    #     along x: 2 * 1.20 * tan(27.3 deg) = 1.24 m  ->  x -0.27..+0.97
    # which covers the cell and still lands inside the MAPPER grid below
    # (x -0.55..1.25, y -0.90..0.90). The camera itself sits above the grid's
    # z ceiling of 1.05, which is fine -- it looks in from outside.
    "overhead": {
        "width": 640,
        "height": 480,
        "horizontal_fov_deg": 69.0,
        "near": 0.15,
        "far": 3.0,
        "pose": [0.35, 0.0, 1.20, 0.0, 0.70710678, 0.70710678, 0.0],
    },
}

# Volumetric map covering the cell in front of the robot.
MAPPER = {
    "voxel_size": 0.015,
    "esdf_voxel_size": 0.025,
    "extent": (1.8, 1.8, 1.4),
    "grid_center": (0.35, 0.0, 0.35),
    "depth_min": 0.15,
    "depth_max": 2.5,
    # Blind decay, applied to EVERY voxel on EVERY integrated frame -- so this
    # is per-frame, not a time constant. 1.0 disables it, which is what we want:
    # global decay erodes voxels the camera CANNOT currently see, which just
    # eats geometry (measured: 9456 -> 6946 voxels over 550 frames at 0.99; 0.3
    # wipes the map within a few frames). Clearing belongs to frustum decay.
    "decay_factor": 1.0,
    # Decay for voxels the camera is looking THROUGH -- the honest "I can see
    # that spot and it is empty now" signal, and the only thing that clears an
    # object's old position after it moves.
    #
    # This MUST be < 1.0. At 1.0 the map only ever grows, and a full run
    # measured the consequence: 75 412 voxels still climbing linearly (a
    # healthy map is 9000-12000), and self-mask artifacts that accumulate
    # until one sticks inside the arm permanently -- after which every plan
    # fails forever. Two cameras reach that point about twice as fast as one.
    # 0.85 was too aggressive: it wiped surfaces faster than they could be
    # re-observed.
    "frustum_decay_factor": 0.97,
    # How far from the robot's collision spheres a depth pixel must be to count
    # as scene rather than robot. A wrist camera looks straight at its own
    # gripper, so this needs to be generous.
    "self_mask_margin": 0.12,
    "esdf_every_n_frames": 10,      # recompute the distance field this often
    "minimum_tsdf_weight": 0.01,
}


SCENE = SceneSpec(
    obstacles=OBSTACLES,
    targets=TARGETS,
    home=HOME,
    scan_poses=SCAN_POSES,
    cameras=CAMERAS,
    mapper=MAPPER,
    # The planner is never told about this one; the cameras have to find it.
    unmapped=[DRAG_CUBE],
    # Report how many occupied voxels land inside the slab, and how high they
    # reach. The z extent is what separates "the cameras can see this" from
    # "the cameras can see the BOTTOM of this" -- a wrist camera alone stops
    # around 0.36 m on a 0.70 m object.
    watch=[WatchBox(DRAG_CUBE[0], DRAG_CUBE[1], DRAG_CUBE[2])],
    motions={DRAG_CUBE[0]: CUBE_SWEEP},
)
