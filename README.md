# UR5_curobo — cuRobo 0.8.x (cuRoboV2)

Env: `conda activate curobo_isaaclab`  (nvidia-curobo 0.8.0.post1.dev42)

cuRobo 0.8.x ships no UR5e config — only `ur10e`. These configs were converted
from the 0.7.7 ones with `tools/convert_v1_robot_yaml.py` and verified to plan.

```
configs/ur5e.yml                  converted from curobo-0.7.7 ur5e.yml
configs/ur5e_robotiq_2f_140.yml   converted from curobo-0.7.7 ur5e_robotiq_2f_140.yml
assets/robot/ur_description/      ur5e URDFs (from 0.7.7) + ur5e meshes (from 0.8.0)
assets/robot/kinova/...           robotiq_2f_140 meshes (from 0.7.7, absent in 0.8.0)
tools/convert_v1_robot_yaml.py    v1 -> v2 robot yaml schema converter
tools/check_robot_cfg.py          smoke test: FK -> planner build -> plan_pose
tools/bench_mapper.py             Mapper TSDF integrate + ESDF timing
```

The Isaac Sim demo and its planner service live in `scripts/`; see HANDOVER.md
for why they are two processes and how the cameras are set up.

Verify:

    python tools/check_robot_cfg.py ur5e
    python tools/check_robot_cfg.py ur5e_robotiq_2f_140 ur5e_robotiq_2f_140.urdf

Point cuRobo at these files with `ContentPath(robot_config_absolute_path=...,
robot_urdf_absolute_path=..., robot_asset_absolute_path=...)` — see
`tools/check_robot_cfg.py`.

## v1 -> v2 schema changes applied

| v1 (0.7.x)                              | v2 (0.8.x)                  |
|-----------------------------------------|-----------------------------|
| `ee_link` + `link_names`                 | `tool_frames: [...]`        |
| `cspace.retract_config`                  | `cspace.default_joint_position` |
| `usd_path`/`usd_robot_root`/`isaac_usd_path`/`usd_flip_joints`/`usd_flip_joint_limits` | dropped (not accepted) |
| —                                        | `format_version: 2.0`       |

Collision spheres, `self_collision_ignore`, `self_collision_buffer`, `cspace`
limits and `mesh_link_names` carry over unchanged.

## Not carried over

`ur10e.yml` has a `dynamics:` block pointing at a neural inverse-dynamics
checkpoint (`inverse_dynamics/ur10e.pt`). No UR5e equivalent exists, and the
checkpoint is not shipped in the repo. `load_dynamics` defaults to `False`, so
planning works without it; the torque-limit-aware part of B-spline trajopt does
not apply until such a model exists for the UR5e.
