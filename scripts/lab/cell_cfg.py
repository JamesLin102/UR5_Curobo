"""CellCfg: every runtime knob of an Isaac Lab cell, as one configclass.

The hardware and the world are NOT in here: those come from rig.ROBOTS and the
scene module, by name, exactly as on the Isaac Sim side. What is here is how
this backend builds and drives them, so it can be overridden from the command
line (lab/app.py) or through Isaac Lab's Hydra config overrides.
"""

from isaaclab.utils import configclass

import scenes
from rig import DEFAULT_ROBOT, HOST, PORT


@configclass
class CellCfg:
    robot: str = DEFAULT_ROBOT          # key into rig.ROBOTS
    scene: str = scenes.DEFAULT         # module under scripts/scenes/
    mapping: bool = True                # False: no cameras, the planner sees the static scene
    overhead: bool = True               # also build the scene's FIXED ("pose") cameras
    map_every: int = 6                  # fuse a frame every N sim steps while moving
    # Sim steps the depth image trails the physics by. Isaac Lab renders
    # synchronously, so it is not the Isaac Sim side's 2: measured 0 with
    # tools/lab_measure_depth_lag.py, for both cameras, for a teleported
    # payload and for a 0.3 rad arm jump alike -- the frame read after the
    # step that moved something already shows it.
    depth_lag: int = 0
    verbose: bool = True

    # Where plans come from. "single": one planner_server.py over the socket,
    # which serves one map, so one environment. "none": no planner at all --
    # the cell builds and steps, and every plan or IK request fails. For tools
    # that only look at the stage.
    planner_mode: str = "single"
    host: str = HOST
    port: int = PORT

    camera_class: str = "camera"        # "camera" | "tiled" (TiledCamera: one render product)
    markers: str = "overlay"            # goal markers: "overlay" | "usd" | "off"

    # How the URDF becomes USD. False merges only the massless frames into
    # their parent, which is what the Isaac Sim side's importer does; True
    # also folds links that have mass into theirs (Isaac Lab's default).
    merge_inertial: bool = False
    force_usd: bool = False             # regenerate the USD even if the cache looks current
    usd_dir: str = "~/.cache/ur5_curobo/lab_usd"
