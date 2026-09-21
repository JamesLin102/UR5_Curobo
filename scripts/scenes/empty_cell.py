"""The same cell with nothing in it: the baseline that says the rig is healthy.

Identical to demo_cube except that there is no unmapped body, so every plan
should take the direct 81-waypoint route and none should fail. Run this when
plans start failing and you need to know whether the obstacle is responsible or
something else is: it was exactly this comparison that exposed the target
markers being fused into the map as obstacles sitting on the goals (HANDOVER
section 7).

It also exercises the optional half of the scene contract -- no `unmapped`, no
`watch`, no `motions` -- which is worth keeping working.

Expected, both cameras, 120 s of planning:
    101 101 121 then 81 forever, 0 failures
"""

from .base import SceneSpec
from .demo_cube import CAMERAS, HOME, MAPPER, OBSTACLES, SCAN_POSES, TARGETS

SCENE = SceneSpec(
    obstacles=OBSTACLES,
    targets=TARGETS,
    home=HOME,
    scan_poses=SCAN_POSES,
    cameras=CAMERAS,
    mapper=MAPPER,
    # Nothing for the cameras to discover: unmapped, watch and motions all
    # default to empty.
)
