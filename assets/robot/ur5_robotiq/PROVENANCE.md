# UR5 + Robotiq FT 300 + Wrist Camera + 2F-85 + RealSense D435i

Everything here comes from
[eugene900805/mir_ur5_humble](https://github.com/eugene900805/mir_ur5_humble),
which carries a UR5 on a MiR100. The MiR chassis is the only thing removed:
`ur5_robotiq.urdf.xacro` re-parents the arm to `world` at the origin and then
instantiates the same wrist stack, gripper and camera macros with the same
mounting transforms as `mir_description/urdf/include/mir_100_v1.urdf.xacro`.

Nothing in the kinematic chain was re-derived, re-measured or adjusted by
hand. That is the point of vendoring it: the previous model in this repo was
assembled link by link from photographs and datasheets, and its geometry was
wrong.

## Layout

Each subdirectory is named after the ROS package the meshes came from, so a
`package://<pkg>/...` reference in the upstream URDF maps to `<pkg>/...` here
with nothing else to check.

| Directory | Upstream package | Holds |
|---|---|---|
| `ur_description/` | `Universal_Robots_ROS2_Description` | UR5 (CB3) link meshes |
| `mir_description/` | `mir_robot/mir_description` | FT 300, Wrist Camera, D435i bracket |
| `robotiq_description/` | `ros2_robotiq_gripper/robotiq_description` | 2F-85 links |
| `realsense2_description/` | `realsense2_description` | D435i body |

`robotiq_description/meshes/*/2f_140/` was dropped; only the 2F-85 is used.

## Regenerating the URDF

`ur5_robotiq.urdf.xacro` needs the upstream packages on disk, so it is kept for
reference rather than run on every build. `tools/build_ur5_robotiq_urdf.py`
works from the already-expanded `ur5_robotiq.raw.urdf` and does only
mechanical, reviewable edits -- see that file's docstring.

## Licences

* `ur_description/LICENSE` -- BSD-3-Clause, Universal Robots A/S.
* `robotiq_description/LICENSE` -- the `ros2_robotiq_gripper` licence.
* `mir_description/LICENSE` -- the `mir_robot` licence, and
  `mir_description/meshes/robotiq_wrist/LICENSE.ros-industrial` for the FT 300
  meshes specifically.
* The Robotiq Wrist Camera mesh is tessellated from Robotiq's own CAD download
  and is NOT covered by the BSD notice; see
  `mir_description/meshes/robotiq_wrist/README.md` for its terms.
* `realsense2_description/` -- Apache-2.0, Intel.
