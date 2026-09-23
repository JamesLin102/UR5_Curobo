# UR5 + Robotiq FT 300 + Wrist Camera + 2F-85 + RealSense D435i

Everything here comes from
[eugene900805/mir_ur5_humble](https://github.com/eugene900805/mir_ur5_humble),
which carries a UR5 on a MiR100. The MiR chassis is the only thing removed:
`source/ur5_robotiq.urdf.xacro` re-parents the arm to `world` at the origin and
then instantiates the same wrist stack, gripper and camera macros with the same
mounting transforms as `mir_description/urdf/include/mir_100_v1.urdf.xacro`.

Nothing in the kinematic chain was re-derived, re-measured or adjusted by
hand. That is the point of vendoring it: the previous model in this repo was
assembled link by link from photographs and datasheets, and its geometry was
wrong.

## Layout

One folder per device. Every mesh path in `ur5_robotiq.urdf` is relative to
the URDF itself, so the directory can be moved as a whole.

```
ur5_robotiq.urdf        the one URDF in use (generated)
source/
  ur5_robotiq.raw.urdf  xacro's output, package:// paths; the builder's input
  ur5_robotiq.urdf.xacro  how the raw URDF was made; its includes need the
                          upstream packages and are not runnable as-is
meshes/
  ur5/                  UR5 (CB3) links            visual/*.dae  collision/*.stl
  ft300/                FT 300 + mounting plate    visual/
  wrist_camera/         Robotiq Wrist Camera       visual/
  robotiq_2f85/         2F-85 gripper              visual/*.dae  collision/*.stl
  d435i/                D435i body and bracket     visual/  collision/bracket_hull.stl
```

The FT 300 and the Wrist Camera have no collision mesh: upstream collides
them as cylinders and boxes. The bracket's collision mesh is its convex hull,
written by `tools/build_ur5_robotiq_urdf.py` (the raw mesh is not watertight,
so cuRobo's sphere fitter cannot find its inside).

Mesh files are byte-identical to upstream; only their paths and names changed.
`tools/build_ur5_robotiq_urdf.py` (`MESH_DIRS`) maps each upstream path to
its place here:

| Upstream package path | Here |
|---|---|
| `ur_description/meshes/ur5/{visual,collision}/` | `meshes/ur5/{visual,collision}/` |
| `mir_description/meshes/robotiq_wrist/robotiq_ft300*.stl` | `meshes/ft300/visual/ft300*.stl` |
| `mir_description/meshes/robotiq_wrist/robotiq_wrist_camera.stl` | `meshes/wrist_camera/visual/wrist_camera.stl` |
| `mir_description/meshes/{visual,collision}/D435i_mounted.STL` | `meshes/d435i/visual/bracket.stl` (the two were identical) |
| `realsense2_description/meshes/d435.dae` | `meshes/d435i/visual/d435.dae` |
| `robotiq_description/meshes/{visual,collision}/2f_85/` | `meshes/robotiq_2f85/{visual,collision}/` |

Not kept: `2f_140/` (only the 2F-85 is used) and `ur_to_robotiq_adapter`
(the Wrist Camera takes the adapter's place in this stack).

## Licences

* `meshes/ur5/LICENSE` -- BSD-3-Clause, Universal Robots A/S.
* `meshes/robotiq_2f85/LICENSE` -- the `ros2_robotiq_gripper` licence.
* `meshes/ft300/LICENSE.ros-industrial` for the FT 300 meshes, and
  `meshes/ft300/LICENSE.mir_robot`, the `mir_robot` licence, which also covers
  the D435i bracket in `meshes/d435i/visual/bracket.stl`.
* The Robotiq Wrist Camera mesh is tessellated from Robotiq's own CAD download
  and is NOT covered by the BSD notice; see `meshes/wrist_camera/NOTICE.md` for
  its source. The project owner has confirmed it may be redistributed with
  this repository (2026-09-23).
* `meshes/d435i/visual/d435.dae` -- Apache-2.0, Intel (`realsense2_description`).
  Upstream's licence file was not vendored with it.
