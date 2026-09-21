"""What a scene has to provide, and what it may provide.

A scene is the one thing both processes must agree on: the planner server
builds its collision world from it, the Isaac Sim client builds the rendered
world from it, and they never exchange geometry at run time. Defining it once,
here, is what stops the two worlds drifting apart.

Deliberately free of cuRobo and Isaac Sim imports -- Isaac Sim 5.1 pins Warp
1.8.2 and cuRobo 0.8 needs Warp >= 1.13, so anything both processes import has
to stay neutral. Standard library only.

To add a scene, copy scenes/demo_cube.py, edit it, and run both processes with
``--scene <module name>``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# name, dims (full extents, m), pose [x, y, z, qw, qx, qy, qz], rgb 0..1
Body = Tuple[str, List[float], List[float], Tuple[float, float, float]]


@dataclass(frozen=True)
class WatchBox:
    """A volume to count occupied voxels in, reported on every ESDF refresh.

    Purely diagnostic, and the reason the planner server needs no knowledge of
    what any particular scene contains. Point one at an obstacle the planner is
    NOT told about and the count answers the only question that matters: has
    the map actually seen it, and how high does it reach?

    ``dims``/``pose`` use the same convention as Body, so a scene can usually
    build one straight from an object it already defines.
    """

    name: str
    dims: Sequence[float]
    pose: Sequence[float]

    def bounds(self):
        """(lo, hi) corner tuples in world coordinates."""
        lo = tuple(self.pose[i] - self.dims[i] / 2 for i in range(3))
        hi = tuple(self.pose[i] + self.dims[i] / 2 for i in range(3))
        return lo, hi


@dataclass(frozen=True)
class SceneSpec:
    """Everything the two processes need to agree on.

    Required:
        obstacles   bodies the planner IS told about, as exact cuboids. Keep
                    this to genuinely static structure (a table, a fixture).
                    Anything here is not a test of perception.
        targets     tool goal poses [x, y, z, qw, qx, qy, qz], visited in turn.
        home        joint-space rest pose (rad), in the robot's own joint order.
        scan_poses  joint poses swept once at startup so the map has content
                    before the first plan. Keep them clear of the table.
        cameras     name -> camera spec. Each needs width/height/
                    horizontal_fov_deg/near/far plus exactly one of:
                      "link"  a frame in the robot's URDF; the server derives
                              the pose by forward kinematics.
                      "pose"  fixed in world, [x,y,z,qw,qx,qy,qz], OPTICAL
                              frame (+Z along the view, +X right, +Y down).
        mapper      TSDF/ESDF settings; see demo_cube.py for what each one
                    does and what it costs to get wrong.

    Optional:
        unmapped    bodies that exist ONLY in the simulator. The planner is
                    never told about them, so avoiding them is proof that
                    perception is driving the plan. This is where a scene puts
                    the thing it wants the cameras to discover.
        watch       volumes to report voxel counts for (see WatchBox).
        motions     name -> dict describing how an unmapped body moves. The
                    client decides how to interpret its own entries; the server
                    ignores this field entirely.
    """

    obstacles: List[Body]
    targets: List[List[float]]
    home: List[float]
    scan_poses: List[List[float]]
    cameras: Dict[str, dict]
    mapper: dict
    unmapped: List[Body] = field(default_factory=list)
    watch: List[WatchBox] = field(default_factory=list)
    motions: Dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        for name, cam in self.cameras.items():
            has_link, has_pose = "link" in cam, "pose" in cam
            if has_link == has_pose:
                raise ValueError(
                    f"camera {name!r} must have exactly one of 'link' or "
                    f"'pose'; got link={has_link} pose={has_pose}"
                )
            for key in ("width", "height", "horizontal_fov_deg", "near", "far"):
                if key not in cam:
                    raise ValueError(f"camera {name!r} is missing {key!r}")
        for t in self.targets:
            if len(t) != 7:
                raise ValueError(f"target {t} is not [x,y,z,qw,qx,qy,qz]")

    def body(self, name: str) -> Optional[Body]:
        """Look up an obstacle or unmapped body by name."""
        for b in list(self.obstacles) + list(self.unmapped):
            if b[0] == name:
                return b
        return None
