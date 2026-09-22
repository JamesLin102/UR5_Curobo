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
from .demo_cube import CAMERAS, HOME, MAPPER, SCAN_POSES

# Raised so the block is well clear of the table and the grasp is unambiguous.
PEDESTAL_H = 0.10
PEDESTAL_TOP = PEDESTAL_H          # they stand on the table, whose top is z=0
BLOCK_SIZE = 0.045
PICK_XY = ((0.45, -0.25), (0.45, 0.25))

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

# The same slab demo_cube uses, in the same place: low and wide, entirely
# inside what the cameras can see, and measured there to divert the route
# rather than block it. It sits between the two pedestals.
SLAB = ("drag_me", [0.12, 0.35, 0.35], [0.30, 0.0, 0.175, 1, 0, 0, 0],
        (0.62, 0.20, 0.18))

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
