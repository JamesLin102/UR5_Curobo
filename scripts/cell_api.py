"""The contract every simulator backend offers: one cell, driven by primitives.

Two backends implement it -- sim_env.SimEnv on Isaac Sim, and lab.cell.CellView
on Isaac Lab -- and everything that only wants to DRIVE a cell (the demo loop,
the pick-and-place task, the checks) is written against this file and nothing
else. Standard library and numpy only, so both processes can import it.

Two layers:

  CellLike    the blocking primitives: move_to, move_tool_z, grip, idle, ...
              Each steps the simulation itself and returns when it is done.
  ops         the same primitives as DATA (MoveTo, MoveZ, Grip, Idle, Scan).
              A program is a list of them. run_ops_blocking() plays one on a
              CellLike; the Isaac Lab backend also interprets them one physics
              tick at a time across many environments at once, which a
              blocking call cannot do.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

import scenes
from rig import DEFAULT_ROBOT


# --- configuration and results -------------------------------------------------

# Observation.holding: the gripper was last closed, and a payload is within
# this distance of the tool frame. A heuristic, not a contact query: good
# enough to tell "carrying it" from "closed on air".
HOLDING_RADIUS = 0.05

# How long the primitives take, in sim steps. Both backends run the same
# numbers, so a leg costs the same on either.
SCAN_STEPS = 70            # per scan pose...
SCAN_BLEND_STEPS = 50      # ...of which blending onto it takes this many
MOVE_Z_STEPS = 70          # move_tool_z's default interpolation
GRIP_EXTRA_STEPS = 6       # added to the stroke's own open/close time
SETTLE_MAX_STEPS = 240     # wait at most this long for the fingers to stop,
SETTLE_QUIET_STEPS = 12    # "stopped" meaning this many steps in a row
SETTLE_TOL = 2.0e-4        # each moving less than this, rad


@dataclass
class EnvCfg:
    scene: str = scenes.DEFAULT
    robot: str = DEFAULT_ROBOT
    mapping: bool = True        # False: no cameras, the planner sees the static scene
    overhead: bool = True       # the fixed camera, on top of the wrist one
    map_every: int = 6          # fuse a frame every N sim steps while moving
    depth_lag: int = 2          # sim steps the depth annotator trails the physics by
    verbose: bool = True


@dataclass
class ResetOptions:
    block_on: int = 0                       # which target the payload starts under
    slab_pose: Optional[List[float]] = None  # [x, y, z] for the unmapped body; None = scene's
    clear_map: bool = True                  # empty the server's map first
    scan: bool = True                       # sweep scene.scan_poses to seed the map
    settle_steps: int = 60


@dataclass
class MoveResult:
    ok: bool
    reason: Optional[str] = None     # why not, in the server's words
    solve_ms: float = 0.0
    waypoints: int = 0
    clearance: Optional[str] = None  # closest approach to unmapped bodies, as reported
    sim_steps: int = 0


@dataclass
class GripResult:
    settled: bool                    # False: still creeping when the budget ran out
    steps: int
    target: float                    # commanded leader angle, rad
    error: float                     # worst driven joint's distance from it
    spread: Dict[str, float] = field(default_factory=dict)  # every joint, as fraction closed


@dataclass
class Observation:
    q: np.ndarray                    # arm joints, in the planner's order
    gripper: float                   # leader joint, rad (0 open)
    tool_pose: np.ndarray            # tool frame [x, y, z, qw, qx, qy, qz]
    objects: Dict[str, np.ndarray]   # payload poses, same layout -- ground truth
    holding: bool
    sim_time: float
    depth: Optional[Dict[str, np.ndarray]] = None


class CellLike(Protocol):
    """One cell as a blocking API. Poses are in the scene's own coordinates."""

    scene: Any                        # scenes.SceneSpec
    payload: Dict[str, Any]           # name -> backend handle
    bodies: Dict[str, Any]            # unmapped bodies, name -> backend handle
    cams: Dict[str, Any]              # camera name -> backend handle; empty = no mapping

    @property
    def running(self) -> bool: ...
    @property
    def gripper_closed(self) -> bool: ...

    def reset(self, options: Optional[ResetOptions] = None) -> Observation: ...
    def move_to(self, pose: Sequence[float]) -> MoveResult: ...
    def move_tool_z(self, dz: float, n_steps: int = MOVE_Z_STEPS) -> MoveResult: ...
    def grip(self, close: bool) -> GripResult: ...
    def idle(self, n: int) -> None: ...
    def scan(self) -> int: ...
    def observe(self, images: bool = False) -> Observation: ...
    def command_arm(self, q: Sequence[float]) -> None: ...
    def close(self) -> None: ...


# --- programs ------------------------------------------------------------------
#
# Every op may carry a `label`, so a caller can find its result afterwards
# (ProgramResult.labelled). The two that can fail also carry what a failure
# means for the program they are in:
#   why         the reason reported if it fails ("no plan", ...)
#   fail_idle   steps to hold after a failure, before giving up
#   fatal       stop the program on failure (True), or carry on and report the
#               failure once the program ends (False)


@dataclass
class MoveTo:
    pose: Sequence[float]
    why: str = "no plan"
    fail_idle: int = 0
    fatal: bool = True
    label: str = ""


@dataclass
class MoveZ:
    dz: float
    n_steps: int = MOVE_Z_STEPS
    why: str = "no IK"
    fail_idle: int = 0
    fatal: bool = True
    label: str = ""


@dataclass
class Grip:
    close: bool
    label: str = ""


@dataclass
class Idle:
    n: int
    label: str = ""


@dataclass
class Scan:
    label: str = ""


Op = Any  # one of the dataclasses above


@dataclass
class ProgramResult:
    ok: bool
    why: Optional[str] = None
    results: List[Tuple[Op, Any]] = field(default_factory=list)  # (op, primitive's result)

    def first(self, kind):
        """The result of the first op of `kind` that ran, or None."""
        return next((r for op, r in self.results if isinstance(op, kind)), None)

    def labelled(self, label):
        """The result of the first op carrying `label` that ran, or None."""
        return next((r for op, r in self.results if op.label == label), None)


def fail_reason(op, result):
    """The reason a failed op reports: its `why`, plus the server's words for a plan."""
    if isinstance(op, MoveTo):
        return f"{op.why}: {result.reason}"
    return op.why


def run_ops_blocking(cell: CellLike, ops: Sequence[Op]) -> ProgramResult:
    """Play a program on a blocking cell. The reference semantics for ops."""
    out = ProgramResult(ok=True)
    for op in ops:
        if isinstance(op, MoveTo):
            r = cell.move_to(op.pose)
        elif isinstance(op, MoveZ):
            r = cell.move_tool_z(op.dz, n_steps=op.n_steps)
        elif isinstance(op, Grip):
            r = cell.grip(close=op.close)
        elif isinstance(op, Idle):
            r = cell.idle(op.n)
        elif isinstance(op, Scan):
            r = cell.scan()
        else:
            raise TypeError(f"not an op: {op!r}")
        out.results.append((op, r))
        if isinstance(op, (MoveTo, MoveZ)) and not r.ok:
            if out.ok:
                out.ok, out.why = False, fail_reason(op, r)
            if op.fail_idle:
                cell.idle(op.fail_idle)
            if op.fatal:
                break
    return out
