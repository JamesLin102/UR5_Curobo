"""Shuttle a block between two pedestals, around an obstacle only the cameras see.

Three kinds of object, and the difference between them is the point:

  obstacles  the table and the two pedestals. The planner is TOLD about these,
             so it routes around them without any perception involved.
  unmapped   a slab between the pedestals. The planner is never told; the
             cameras have to find it, exactly as in demo_cube.
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
from .demo_cube import CAMERAS, DRAG_CUBE, HOME, MAPPER, SCAN_POSES

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

# demo_cube's slab, same size, its own position -- and that is not an
# oversight to be tidied away by importing it whole. The two scenes need it in
# different places because their pedestals are in different places, and
# importing it whole is exactly what broke this scene once.
#
# Chosen by sweeping pedestal y against slab x and measuring both things that
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
SLAB = ("drag_me", [DRAG_CUBE[1][0], DRAG_CUBE[1][1], 0.46],
        [0.46, 0.0, 0.23, 1, 0, 0, 0], DRAG_CUBE[3])

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
