"""demo_cube with nothing for the cameras to discover: the control run.

The planner is still told about the table and both cameras still run, but the
MAP now comes out empty: mapper["floor_z"] keeps the table out of it, and in
this scene the table is the only thing there is to see. 1510 frames fused, 0
voxels. That is the expected output here, not a symptom -- demo_cube, which
has an obstacle standing above the floor, maps it normally.

So this answers a narrower question than it used to: with nothing to
discover, is the rig healthy? Every plan should take the direct 81-waypoint
route and none should fail. Run it when plans start failing and you need to
know whether the obstacle is responsible or something else is. That
comparison is what exposed the target markers being fused into the map as
obstacles sitting on the goals -- shrinking the obstacle to nothing still
failed 193 plans out of 199, which ruled the obstacle out in one run
(HANDOVER section 7).

What it no longer exercises is the perception path itself. An empty map
cannot tell you the cameras are feeding the mapper; for that, use demo_cube
and watch the voxel count.

It also exercises the optional half of the scene contract -- no `unmapped`,
no `watch`, no `motions` -- which is worth keeping working.

Measured on UR5 + 2F-85, both cameras: 28 plans, 0 failures, 81 waypoints
each at 47-49 ms, "0 voxels" in every map line.
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
