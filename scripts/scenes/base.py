"""What a scene has to provide, and what it may provide.

A scene is the one thing both processes must agree on: the planner server
builds its collision world from it, the Isaac Sim client builds the rendered
world from it, and they never exchange geometry at run time. Defining it once,
here, is what stops the two worlds drifting apart.

Deliberately free of cuRobo and Isaac Sim imports -- Isaac Sim 5.1 pins Warp
1.8.2 and cuRobo 0.8 needs Warp >= 1.13, so anything both processes import has
to stay neutral. Standard library only.

To add a scene, make an example package scripts/<name>/ with a scene.py
(copy pick_place/scene.py), and run both processes with ``--scene <name>``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# name, dims (full extents, m), pose [x, y, z, qw, qx, qy, qz], rgb 0..1
Body = Tuple[str, List[float], List[float], Tuple[float, float, float]]

# What a Body's dims mean, by shape (SceneSpec.shapes; a body not named there
# is a cuboid):
#   "cuboid"    full extents along its own x, y, z
#   "cylinder"  [2r, 2r, height], its axis along its own z
SHAPES = ("cuboid", "cylinder")

# A Body plus a mass in kg. Unlike the other two kinds of object in a scene,
# this one is a real rigid body: it falls, it can be squeezed, and it comes
# away when the gripper closes on it.
Payload = Tuple[str, List[float], List[float], Tuple[float, float, float], float]


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
                    Optional "rgb": True renders colour as well as depth
                    (Isaac Lab backend), for perception that needs it.
        mapper      TSDF/ESDF settings; see pick_place/scene.py for what each one
                    does and what it costs to get wrong. Optional key
                    "floor_z": nothing at or below that height is fused, so a
                    surface the planner already knows exactly (the table) is
                    not also mapped as a fatter duplicate.

    Optional:
        unmapped    bodies that exist ONLY in the simulator. The planner is
                    never told about them, so avoiding them is proof that
                    perception is driving the plan. This is where a scene puts
                    the thing it wants the cameras to discover.
        payload     rigid bodies to be PICKED UP, with a mass. Distinct from
                    `unmapped`, which is visual-only and meant to be avoided:
                    a payload is meant to be reached, and the cameras will map
                    it as an obstacle like anything else. `pick` says how to
                    get the last few centimetres.
        pick        {"descend_m": ..., "lift_m": ...}. The planner routes to a
                    pose ABOVE the payload, which is a place the map agrees is
                    free; the final descent onto it and the lift back off are
                    open-loop, because a thing you intend to grasp is exactly
                    a thing the map calls an obstacle. Keep them short.
        watch       volumes to report voxel counts for (see WatchBox).
        shapes      {unmapped body name: "cylinder"}; a body not named here is a
                    cuboid, so a scene with only boxes leaves this out. A
                    cylinder's dims are [2r, 2r, height], axis along its z.
        solid       names of unmapped bodies that are also COLLIDERS: kinematic
                    (they do not move when hit), so the arm is really stopped
                    by them and the contact is reported (Isaac Lab backend).
                    Unmapped bodies not named here are visual only, as the
                    draggable slab in pick_place is.
        keep_out    bodies the PLANNER is told about and nothing else: never
                    spawned in the simulator, never seen by a camera. Volumes
                    the arm must not be planned into although nothing is
                    there -- a floor guard over the table, say. The straight
                    moves of a grasp are IK, not planned, so they may go in.
        planner_joint_limits
                    {joint name: (lower, upper)} in rad, narrowing the URDF's
                    limits for the PLANNER only (plans and the IK of the
                    vertical moves); the simulator keeps the URDF's. For a scene
                    where one IK branch is safe to plan to and not to descend
                    from -- see grasp/scene.py.
    """

    obstacles: List[Body]
    targets: List[List[float]]
    home: List[float]
    scan_poses: List[List[float]]
    cameras: Dict[str, dict]
    mapper: dict
    unmapped: List[Body] = field(default_factory=list)
    payload: List[Payload] = field(default_factory=list)
    pick: dict = field(default_factory=dict)
    watch: List[WatchBox] = field(default_factory=list)
    shapes: Dict[str, str] = field(default_factory=dict)
    planner_joint_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    keep_out: List[Body] = field(default_factory=list)
    solid: List[str] = field(default_factory=list)

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
        if self.payload and not self.pick:
            raise ValueError("a scene with a payload needs pick={'descend_m':..., "
                             "'lift_m':...}; the planner cannot route the last "
                             "few centimetres onto something it maps as an obstacle")
        for name, dims, pose, _, mass in self.payload:
            if mass <= 0:
                raise ValueError(f"payload {name!r} needs a positive mass, got {mass}")
        unmapped = {b[0]: b[1] for b in self.unmapped}
        for name, shape in self.shapes.items():
            if shape not in SHAPES:
                raise ValueError(f"shape of {name!r} is {shape!r}; one of {SHAPES}")
            if name not in unmapped:
                raise ValueError(f"shapes names {name!r}, which is not an unmapped body "
                                 f"(only those can be shaped, for now)")
            if shape == "cylinder" and abs(unmapped[name][0] - unmapped[name][1]) > 1e-9:
                raise ValueError(f"cylinder {name!r} has dims {unmapped[name]}; "
                                 f"want [2r, 2r, height]")

        for name in self.solid:
            if name not in unmapped:
                raise ValueError(f"solid names {name!r}, which is not an unmapped body")
        for joint, (lo, hi) in self.planner_joint_limits.items():
            if not lo < hi:
                raise ValueError(f"planner_joint_limits[{joint!r}] = ({lo}, {hi}) is empty")

    def shape(self, name: str) -> str:
        """A body's shape: "cylinder" if `shapes` says so, else "cuboid"."""
        return self.shapes.get(name, "cuboid")

    def body(self, name: str) -> Optional[Body]:
        """Look up an obstacle, unmapped body or payload by name."""
        for b in list(self.obstacles) + list(self.unmapped) + \
                [p[:4] for p in self.payload]:
            if b[0] == name:
                return b
        return None
