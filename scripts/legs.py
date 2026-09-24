"""One leg -- above, down, grip, up -- as cell_api ops. Any task that grips uses it.

The planner routes to a pose ABOVE the grip, which the map agrees is free;
the last stretch down and back up is IK and interpolation, because the thing
to be gripped is, to the map, an obstacle (see scenes.base.SceneSpec.pick).

    action = [x, y, z, yaw]   z is the tool frame's height at the grip; yaw
                              turns the tool about world Z, tool pointing down.

Standard library only, so both backends and every task can import it.
"""

import math

from cell_api import Grip, Idle, MoveTo, MoveZ

GRIP_LABEL = "leg_grip"


def tool_pose(x, y, z, yaw):
    """[x, y, z, qw, qx, qy, qz] with the tool pointing down, turned by yaw.

    Tool-down is (0, 1, 0, 0), a half turn about X -- what the scene's own
    targets use. Turning that about world Z by yaw gives
    (0, cos(yaw/2), sin(yaw/2), 0).
    """
    return [float(x), float(y), float(z),
            0.0, math.cos(yaw / 2.0), math.sin(yaw / 2.0), 0.0]


def leg_ops(pose, close, open_first, descend, lift):
    """One leg as a program: above, down, grip, up.

    open_first: a pick needs an open hand. After a pick that closed on nothing
    the gripper is still shut, so it opens where the arm stands, first.
    """
    above = list(pose)
    above[2] += descend
    ops = [Grip(close=False)] if open_first else []
    return ops + [
        MoveTo(above, why="no plan", fail_idle=30),
        Idle(20),
        MoveZ(-descend, why="no IK going down"),
        Idle(15),
        Grip(close=close, label=GRIP_LABEL),
        MoveZ(lift, why="no IK going up", fatal=False),
        Idle(20),
    ]


def leg_info(result):
    """What a leg reports, from its ProgramResult: the planner's numbers, the grip."""
    info = {}
    moved = result.first(MoveTo)
    if moved is not None:
        info.update(solve_ms=moved.solve_ms, waypoints=moved.waypoints,
                    clearance=moved.clearance)
    grip = result.labelled(GRIP_LABEL)
    if grip is not None:
        info["grip_settled"] = grip.settled
    return info
