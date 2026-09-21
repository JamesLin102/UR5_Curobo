"""demo_cube with nothing for the cameras to discover: the control run.

The cell is NOT empty and neither is the map. The table is still there and the
planner is still told about it, both cameras still run, and the map still fills
with ~18 000 voxels of table surface. What is missing is the `unmapped` body --
the one the planner is never told about and the cameras have to find. Every
other field is demo_cube's, unchanged.

So this answers exactly one question: with nothing to discover, is the rig
healthy? Every plan should take the direct 81-waypoint route and none should
fail. Run it when plans start failing and you need to know whether the obstacle
is responsible or something else is. That comparison is what exposed the target
markers being fused into the map as obstacles sitting on the goals -- shrinking
the obstacle to nothing still failed 193 plans out of 199, which ruled the
obstacle out in one run (HANDOVER section 7).

It also exercises the optional half of the scene contract -- no `unmapped`, no
`watch`, no `motions` -- which is worth keeping working.

Measured, both cameras, 120 s of planning:
    101 101 121 then 81 x 39, 0 failures, "0 tall []" in every map line
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
