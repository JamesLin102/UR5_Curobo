"""Pick a cube off the table from among tall cylinders, seen by the wrist camera only.

The RL scene (docs/grasp_rl_plan.md). Three kinds of object, as in pick_place:

  obstacles  the table. The planner is TOLD about it.
  unmapped   cylinders, radius 35 mm and 0.30 m tall -- taller than the wrist
             is at the pre-grasp -- that the planner is never told about in
             evaluation: the wrist camera has to find them. (In training the
             planner will be told; that is a mode, not a different scene.)
  payload    a 45 mm cube, the same block pick_place uses, straight on the
             table, in a colour the table does not have.

The only camera is the wrist D435i, as on the real arm. HOME is therefore not
a rest pose but the main VIEW: the camera 0.65 m over the middle of the
workspace, looking straight down. The scan adds oblique views from the sides
and from behind, which see the cylinders' sides and whatever one cylinder
hides from the view above.

This file holds one fixed layout, to look at and to test against. Episodes
will randomise the cube within CUBE_XY and the cylinders within CYL_XY
(a later milestone).

HOME and SCAN_POSES are solved, not guessed, by tools/grasp_scan_poses.py:
collision-aware IK for each camera VIEW against the table plus NO_GO -- the
volume any cylinder could occupy -- and then the joint-space blends between
consecutive poses (the path the scan actually takes) checked against NO_GO
too. Its report is pasted below them.

    python scripts/planner_server.py --scene grasp
    python tools/grasp_view.py
"""

import math

from rig import DEFAULT_ROBOT, ROBOTS
from scenes import SceneSpec

# --- the objects -----------------------------------------------------------

# The table top, which is where the robot's base stands (z = 0 is the base).
# 2 mm above the simulator's ground plane rather than on it: pick_place's
# table sits just below the ground, which is fine for depth, but in colour the
# ground's grid is what the camera sees, and one of its white lines ran
# straight through the cube in the view from HOME. Everything below is
# measured from this height.
TABLE_TOP = 0.002
TABLE_RGB = (0.45, 0.47, 0.50)
TABLE = ("table", [2.00, 2.00, 0.10], [0.00, 0.0, TABLE_TOP - 0.05, 1, 0, 0, 0], TABLE_RGB)

CUBE_SIZE = 0.045
CUBE_RGB = (0.95, 0.80, 0.05)      # bright yellow: an RGB threshold finds it (plan §4)
CUBE_MASS = 0.15

CYL_RADIUS = 0.035
CYL_HEIGHT = 0.30
CYL_RGB = (0.20, 0.35, 0.70)
MAX_CYLINDERS = 4

# Where things may stand, for the randomiser and for NO_GO. The cube's centre
# stays inside CUBE_XY; a cylinder's centre inside CYL_XY, which leaves room
# for one on every side of a cube at the edge.
CUBE_XY = ((0.30, 0.60), (-0.35, 0.35))
CYL_XY = ((0.25, 0.65), (-0.40, 0.40))

# Everything any cylinder could occupy, plus 20 mm over its top: no scan pose,
# and no blend between two of them, may put the arm in here. In evaluation the
# scan runs before the map has anything in it, so it has to be safe for every
# layout at once.
NO_GO_TOP = TABLE_TOP + CYL_HEIGHT + 0.02
NO_GO = ([CYL_XY[0][1] - CYL_XY[0][0] + 2 * CYL_RADIUS,
          CYL_XY[1][1] - CYL_XY[1][0] + 2 * CYL_RADIUS,
          NO_GO_TOP - TABLE_TOP],
         [(CYL_XY[0][0] + CYL_XY[0][1]) / 2, (CYL_XY[1][0] + CYL_XY[1][1]) / 2,
          (NO_GO_TOP + TABLE_TOP) / 2, 1, 0, 0, 0])

# The layout this file shows, and what makes it worth showing: of the two
# grips square to the cube's faces, exactly one works. Found by
# tools/grasp_reach.py's sampler (surface gaps >= 15 mm) and checked by its
# grip test, elbow-up limits on:
#
#     tool at +22 deg   works, 8 mm between the arm and the nearest cylinder
#     tool at -68 deg   no plan: the pre-grasp is boxed in
#
# The nearest cylinder is 38 mm from the cube. The first layout this file had
# (cylinders at 58 mm and more) had NO grip at all -- one face unplannable, the
# other driving the wrist 29 mm into a cylinder -- which is how a gap that
# looks generous turned out not to be.
CUBE_XY_YAW = (0.488, -0.032, 0.382)      # x, y, yaw (rad)
CYLINDERS_XY = [(0.572, -0.094), (0.464, -0.322), (0.364, -0.222), (0.480, -0.134)]


def _yaw_quat(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


CUBE = ("cube", [CUBE_SIZE] * 3,
        [CUBE_XY_YAW[0], CUBE_XY_YAW[1], TABLE_TOP + CUBE_SIZE / 2] + _yaw_quat(CUBE_XY_YAW[2]),
        CUBE_RGB, CUBE_MASS)

CYLINDERS = [
    (f"cyl_{i}", [2 * CYL_RADIUS, 2 * CYL_RADIUS, CYL_HEIGHT],
     [x, y, TABLE_TOP + CYL_HEIGHT / 2, 1, 0, 0, 0], CYL_RGB)
    for i, (x, y) in enumerate(CYLINDERS_XY)
]

# --- the camera ------------------------------------------------------------

# The wrist D435i, as pick_place has it, except `near`: the real D435's depth
# starts at about 0.28 m, so the simulated one must not see closer than that
# either, or perception would be tuned on depth the real camera cannot give.
CAMERAS = {
    "wrist": {
        "width": 640,
        "height": 480,
        "horizontal_fov_deg": 69.0,
        "near": 0.28,
        "far": 3.0,
        "link": "camera_link",
        "rgb": True,        # the cube's edges come from colour (plan §4)
    },
}

# Camera views the scan is made of: (name, eye, look-at point), world frame.
# The first is HOME. tools/grasp_scan_poses.py turns each into joint angles.
#
# Where the others can be is set by the arm, not by taste. The camera sits on
# one side of the wrist, so from HOME it can only be turned to look in from
# the +y side, from behind, or from the -y side; from in front of the
# workspace there was no solution at all. Searched around the look-at point
# (azimuth every 30 deg, elevation 45-65 deg, 0.60-0.70 m away), with IK seeded
# at HOME, keeping the joint travel from HOME and the blend's clearance to
# NO_GO:
#
#     az  90 el 65  0.60 m   96 deg  +110 mm    <- right
#     az 120 el 65  0.70 m  109 deg  +110 mm
#     az 180 el 65  0.70 m  165 deg   +98 mm    <- behind
#     az 180 el 65  0.60 m  182 deg   +13 mm
#     az 240 el 65  0.60 m  113 deg   +30 mm    <- left
#     az 270 el 65  0.60 m  106 deg   +15 mm       (under the 20 mm margin)
#     az 210-240, el 45-55  ...           +7..+15 mm   the elbow dips over NO_GO
#
# The -y side is the tight one: turning that way brings the elbow down over
# the cylinders' box, so its best view is steeper and nearer than the others.
_LOOK = (0.45, 0.00, 0.05)
VIEWS = [
    ("home",   (0.45, 0.00, 0.65), (0.45, 0.00, 0.00)),
    ("left",   (0.32, -0.22, 0.59), _LOOK),
    ("behind", (0.15, 0.00, 0.68), _LOOK),
    ("right",  (0.45, 0.25, 0.59), _LOOK),
]

# Solved by tools/grasp_scan_poses.py (2026-09-24). Its check, on these:
#
#     view    camera at              to the table    joint travel
#     home    [+0.450 +0.000 +0.650]  0.65 m, 1.4 mm/px
#     left    [+0.320 -0.220 +0.590]  0.65 m, 1.4 mm/px   115 deg
#     behind  [+0.150 -0.000 +0.680]  0.75 m, 1.6 mm/px    70 deg
#     right   [+0.450 +0.250 +0.590]  0.65 m, 1.4 mm/px   218 deg
#
#     blend              NO_GO     arm above z=0
#     home   -> left     +29 mm    +28 mm
#     left   -> behind   +21 mm    +27 mm
#     behind -> right    +21 mm    +27 mm
#     right  -> home    +110 mm    +29 mm
#
# Every camera is within 3.4 mm and 0.2 deg of its view; the margin asked
# for is 20 mm, and the -y side (left, behind) is where it is thinnest.
HOME = [-0.276, -1.5447, 0.6585, -0.6845, -1.5708, 0.2476]
SCAN_POSES = [
    HOME,
    [-1.0365, -1.7187, 0.9437, -0.5716, -1.1866, 0.5337],   # left
    [-1.1733, -2.1701, 0.8737, -0.4579, -1.1604, 0.9599],   # behind
    [0.5032, -1.3514, 0.6424, -0.6421, -1.9477, 1.0689],    # right
    HOME,
]

# --- the map (evaluation only) ---------------------------------------------

# pick_place's mapper, unchanged; see pick_place/scene.py for why each value
# is what it is. The grid covers the workspace and the arm's reach over it.
MAPPER = {
    "voxel_size": 0.015,
    "esdf_voxel_size": 0.025,
    "extent": (1.8, 1.8, 1.4),
    "grid_center": (0.35, 0.0, 0.35),
    "depth_min": 0.28,
    "depth_max": 2.5,
    "decay_factor": 1.0,
    "frustum_decay_factor": 0.97,
    "self_mask_margin": 0.18,
    "esdf_every_n_frames": 10,
    "minimum_tsdf_weight": 0.01,
    "floor_z": TABLE_TOP + 0.02,
}

# --- the grasp ---------------------------------------------------------------

# pick_place's rule, with the table top where the pedestal top was. The
# 12 mm clearance was measured on a pedestal; on the table it still has to be
# (docs/grasp_rl_plan.md §2).
PAD_HALF = ROBOTS[DEFAULT_ROBOT]["gripper"]["pad_half"]
PAD_CLOSE_DROP = ROBOTS[DEFAULT_ROBOT]["gripper"]["pad_close_drop"]
PAD_CLEARANCE = 0.012
GRASP_Z = TABLE_TOP + PAD_HALF + PAD_CLOSE_DROP + PAD_CLEARANCE
_pad_lo = GRASP_Z - PAD_CLOSE_DROP - PAD_HALF
_pad_hi = GRASP_Z - PAD_CLOSE_DROP + PAD_HALF
_overlap = min(_pad_hi, TABLE_TOP + CUBE_SIZE) - max(_pad_lo, TABLE_TOP)
assert _overlap >= 0.025, f"closed pads overlap the cube by only {_overlap*1000:.1f} mm"

DESCEND = 0.125

# A floor guard, for the planner only (SceneSpec.keep_out): a 0.10 m slab
# over the whole table with a hole round the base, so no plan puts the upper
# arm or the forearm down near the table. The elbow-up limits below keep the
# arm off the worst branch; this keeps it off low reaches the limits still
# allow, such as the upper arm near horizontal at shoulder height (0.09 m).
#
# Measured before choosing the height (tools/grasp_reach.py's planner, elbow
# up, cube over the whole of CUBE_XY, tool yaws -90..45): outside 0.2 m of the
# base's axis the lowest anything went on a planned path was a fingertip at the
# pre-grasp, 148 mm over the table. 100 mm leaves the planner 48 mm of that.
# Inside the hole are the shoulder and the root of the upper arm, which are
# always that low: the UR5's upper arm is offset 0.14 m sideways from the base
# axis, and its lowest spheres (bottom 29 mm over the table) reach 0.206 m out,
# on a circle the shoulder pan sweeps. A 0.20 m hole let every scan blend but
# one clip the guard by up to 6 mm, which would make every plan start "in
# collision"; 0.25 m clears that circle by 44 mm. The grip's straight moves are
# IK, not planned, and go through the guard by design.
GUARD_TOP = TABLE_TOP + 0.10
GUARD_HOLE = 0.25                  # half-width of the square hole round the base
_G = (-1.0, 1.0)                   # the table's extent


def _slab(name, xr, yr):
    return (name, [xr[1] - xr[0], yr[1] - yr[0], GUARD_TOP - TABLE_TOP],
            [(xr[0] + xr[1]) / 2, (yr[0] + yr[1]) / 2, (GUARD_TOP + TABLE_TOP) / 2, 1, 0, 0, 0],
            (0.9, 0.3, 0.2))


KEEP_OUT = [
    _slab("guard_front", (GUARD_HOLE, _G[1]), _G),
    _slab("guard_back", (_G[0], -GUARD_HOLE), _G),
    _slab("guard_left", (-GUARD_HOLE, GUARD_HOLE), (_G[0], -GUARD_HOLE)),
    _slab("guard_right", (-GUARD_HOLE, GUARD_HOLE), (GUARD_HOLE, _G[1])),
]

# Elbow up only, for the planner. The pre-grasp can be reached elbow DOWN
# (shoulder_lift ~ 0, elbow ~ -1.3), which is collision-free where it stands
# and puts the forearm 106 mm into the table once the grip is lowered: the
# straight moves are IK and interpolation, which nothing collision-checks.
# Measured by tools/grasp_reach.py on the empty table, before this: 2 of 924
# grips, the cube at (0.60, +0.28) with the tool at 60 deg among them. Every
# pose this scene uses is elbow up (HOME and the scans: shoulder_lift -1.35..
# -2.17, elbow +0.64..+0.94). cuRobo trims another 0.1 rad off each end
# (position_limit_clip), so what holds is shoulder_lift <= -0.3, elbow >= 0.1.
PLANNER_JOINT_LIMITS = {
    "shoulder_lift_joint": (-math.pi, -0.2),
    "elbow_joint": (0.0, math.pi),
}

# One target: tool down over the cube, turned with it, DESCEND above the grip.
TARGETS = [[CUBE_XY_YAW[0], CUBE_XY_YAW[1], GRASP_Z + DESCEND,
            0.0, math.cos(CUBE_XY_YAW[2] / 2), math.sin(CUBE_XY_YAW[2] / 2), 0.0]]

SCENE = SceneSpec(
    obstacles=[TABLE],
    targets=TARGETS,
    home=HOME,
    scan_poses=SCAN_POSES,
    cameras=CAMERAS,
    mapper=MAPPER,
    payload=[CUBE],
    unmapped=CYLINDERS,
    shapes={name: "cylinder" for name, *_ in CYLINDERS},
    pick={"descend_m": DESCEND, "lift_m": 0.15},
    planner_joint_limits=PLANNER_JOINT_LIMITS,
    keep_out=KEEP_OUT,
)
