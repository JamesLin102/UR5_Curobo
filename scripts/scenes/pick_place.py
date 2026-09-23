"""Shuttle a block between two pedestals, around an obstacle only the cameras see.

Three kinds of object, and the difference between them is the point:

  obstacles  the table and the two pedestals. The planner is TOLD about these,
             so it routes around them without any perception involved.
  unmapped   a slab between the pedestals. The planner is never told; the
             cameras have to find it.
  payload    the block. A real rigid body -- it falls, it is squeezed, it
             comes away when the gripper closes.

The block is picked from one pedestal and placed on the other, then picked back
off and returned, indefinitely. Each leg has to get past the slab.

The awkward part of grasping with a live map is that the thing you intend to
pick up is, to the map, an obstacle -- and the planner will not route a tool
into one. So the plan goes to a pose ABOVE the block, which the map agrees is
free, and only the last stretch down and back up is solved with IK and
interpolated. `pick` says how far that is; keep it short, because it is the
part of the motion nothing is checking.

Measured: the gripper stalls at +0.506 rad against the 45 mm block (commanded
+0.800) and the block tracks the tool to within 5 mm through a 0.126 m lift.

    python scripts/planner_server.py --robot ur5_robotiq --scene pick_place
    DISPLAY=:1 python scripts/isaacsim_ur5e_demo.py --robot ur5_robotiq --scene pick_place
"""

from .base import SceneSpec, WatchBox

# Rest pose. shoulder_lift and elbow are raised relative to the obvious
# [0, -2.2, 1.9, ...]: with the FT 300 and the Wrist Camera in the stack the
# tool sits 55 mm further out, and that pose started the gripper inside the
# obstacle below it. The wrist joints are untouched, so the tool points down.
#
# RETRACTED (2026-09-23) for this scene's slab. The elbow comes in 0.15 rad
# and wrist_1 gives it back, so the tool keeps pointing the same way and simply
# sits further back and higher: [421, 109, 491] -> [386, 109, 546] mm.
# Clearance to the slab went +11.2 mm (grazing it) -> +59.5 mm.
#
# 0.20 rad was tried first and is too much: pulled that far back the arm
# reaches the pedestals more extended, its upper arm ends up 15 mm off the
# mapped slab, and it strands itself -- 217 plans blocked out of 218. 0.15 is
# the most that leaves every pose clear without doing that.
HOME = [0.0, -1.8, 1.10, -1.233, -1.57, 0.0]

# A short sweep at startup so the map has content before the first plan.
# Joint-space poses (rad), kept high and away from the table. Retracted with
# HOME, and for a worse reason: two of these went 14 mm INSIDE the slab, so the
# startup scan swept the arm through the obstacle it was scanning for.
#
#     slab clearance    scan[1] -14.0 -> +21.7    scan[2] -14.2 -> +20.8
SCAN_POSES = [
    HOME,
    [-0.55, -1.85,  1.20, -1.150, -1.57, 0.0],
    [ 0.00, -1.70,  1.05, -1.100, -1.57, 0.0],
    [ 0.55, -1.85,  1.20, -1.150, -1.57, 0.0],
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
    # A wrist camera alone cannot support cross-cell avoidance: its coverage
    # is whatever the arm happens to sweep, it never sees above its own
    # altitude (~0.40 m), and its measured table footprint over a full cycle is
    # only x 0.30..0.45, y -0.45..+0.30. This one sees the whole cell at once.
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
    # gripper, so this needs to be generous. 0.12 held while the arm followed a
    # fixed route and stopped holding once it detoured: the arm mapped itself.
    "self_mask_margin": 0.18,
    "esdf_every_n_frames": 10,      # recompute the distance field this often
    "minimum_tsdf_weight": 0.01,
    # Do not fuse anything at or below this height. The table lives in
    # OBSTACLES, where the planner knows it exactly; mapping it again only
    # produces a fatter duplicate, and on a UR5 (CB3) the shoulder sits low
    # enough that the duplicate swallows the upper arm's own spheres. See
    # Planner.map_floor.
    "floor_z": 0.02,
}

# Raised so the block is well clear of the table and the grasp is unambiguous.
PEDESTAL_H = 0.10
PEDESTAL_TOP = PEDESTAL_H          # they stand on the table, whose top is z=0
BLOCK_SIZE = 0.045
# Moved out from +/-0.25 to +/-0.40 so the slab below has somewhere to stand.
# At +/-0.25 there was no position for it that both blocked the route and left
# the grasps alone: on the route it sat over the pedestals and every goal was
# refused, off the route it did nothing.
PICK_XY = ((0.45, -0.40), (0.45, 0.40))

OBSTACLES = [
    ("table", [2.00, 2.00, 0.10], [0.00, 0.0, -0.06, 1, 0, 0, 0], (0.45, 0.47, 0.50)),
] + [
    (f"pedestal_{i}", [0.14, 0.14, PEDESTAL_H],
     [x, y, PEDESTAL_H / 2, 1, 0, 0, 0], (0.35, 0.38, 0.42))
    for i, (x, y) in enumerate(PICK_XY)
]

# The block starts on pedestal 0.
BLOCK = ("block", [BLOCK_SIZE] * 3,
         [PICK_XY[0][0], PICK_XY[0][1], PEDESTAL_TOP + BLOCK_SIZE / 2, 1, 0, 0, 0],
         (0.90, 0.70, 0.10), 0.15)

# A visual prim, not a physics body: the arm passes through it rather than
# hitting it, and it can be dragged by hand in the viewport without fighting
# PhysX. 0.12 x 0.35 in plan, the size that was measured to divert a route
# every cycle without ever blocking it.
#
# Its position was chosen by sweeping pedestal y against slab x and measuring both things that
# matter: how far the route, planned WITHOUT the slab, runs into it, and how
# much room is left around the grasp once the cameras have put it in the map.
# They pull against each other -- with the pedestals at their old +/-0.25 no
# slab position satisfied both:
#
#     ped y   slab x |  route   pre-grasp  at block
#      0.25     0.30 |  +25.4      +42.9     +17.2   route misses it
#      0.25     0.42 |  -28.5       -1.0     -26.2   fouls the grasp
#      0.38     0.42 |  -14.0      +97.0     +43.0   both
#      0.40     0.46 |  -18.6     +116.4     +57.0   both, with room
#
# Then the height, at that position. 0.35 only reached the short approach
# legs; the long traverse between the pedestals cleared it by 116 mm either
# way, so the big visible motion was not avoiding anything:
#
#     height |  route   pre-grasp  at block
#       0.35 |  -18.6     +116.4     +57.0
#       0.42 |  -20.2      +89.9     +31.5
#       0.46 |  -40.3      +59.2     +31.0   <- this
#       0.50 |  -46.6      +37.1     +31.0   grasp room getting thin
SLAB = ("drag_me", [0.12, 0.35, 0.46],
        [0.46, 0.0, 0.23, 1, 0, 0, 0], (0.62, 0.20, 0.18))

DESCEND = 0.125

# Where grasp_frame has to end up, which is NOT the block's centre.
#
# grasp_frame sits at the middle of the finger pads WITH THE GRIPPER OPEN --
# 185.3 mm from tool0, measured off left_finger_tip.stl. The 2F-85's fingers
# swing rather than translate, so closing carries the pads 13.5 mm further
# out: the pad face goes from tool0 166.3..204.3 to 179.8..217.8. Aim at the
# block's centre and the pads therefore ARRIVE 13.5 mm low.
#
# That was enough to break every grasp. The pedestal is 140 mm across and the
# gripper only opens to 85, so the pads are inside its footprint the whole
# way down; aimed at the centre they ended 10 mm BELOW its top face, stalled
# against it at 0.04 rad of the 0.8 commanded, and shoved the block off
# instead of lifting it.
PAD_HALF = 0.019          # half the pad face, 166.3..204.3 mm
PAD_CLOSE_DROP = 0.0135   # how much further out the pads sit once closed
# 6 mm was not enough: at that height one finger caught the pedestal on the
# way in and stalled at 0.011 rad while the other closed to 0.724, so the
# block was carried pinched against a single pad. The pad face is 38 mm and
# the block only stands 45 mm proud of the pedestal, so buying clearance means
# letting the pad top rise past the block -- which is free space, and costs
# only overlap.
PAD_CLEARANCE = 0.012

# Lowest the closed pads may reach, plus the margin, is what fixes this.
GRASP_Z = PEDESTAL_TOP + PAD_HALF + PAD_CLOSE_DROP + PAD_CLEARANCE
# What actually has to hold is that the closed pads still overlap the block by
# a useful amount, not that they stay under its top face.
_pad_lo = GRASP_Z - PAD_CLOSE_DROP - PAD_HALF
_pad_hi = GRASP_Z - PAD_CLOSE_DROP + PAD_HALF
_overlap = min(_pad_hi, PEDESTAL_TOP + BLOCK_SIZE) - max(_pad_lo, PEDESTAL_TOP)
assert _overlap >= 0.025, f"closed pads overlap the block by only {_overlap*1000:.1f} mm"

# Tool down, DESCEND above the grasp height.
TARGETS = [
    [x, y, GRASP_Z + DESCEND, 0.0, 1.0, 0.0, 0.0]
    for x, y in PICK_XY
]

SCENE = SceneSpec(
    obstacles=OBSTACLES,
    targets=TARGETS,
    home=HOME,
    scan_poses=SCAN_POSES,
    cameras=CAMERAS,
    mapper=MAPPER,
    payload=[BLOCK],
    unmapped=[SLAB],
    pick={"descend_m": DESCEND, "lift_m": 0.15},
    watch=[WatchBox(SLAB[0], SLAB[1], SLAB[2])],
)
