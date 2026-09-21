# UR5_curobo — camera-driven obstacle avoidance

A UR5e with a Robotiq 2F-85 and two depth cameras, planning with **cuRobo 0.8**
inside **Isaac Sim 5.1**. Depth is fused into a live TSDF/ESDF and handed back
to the motion planner every few frames.

The point of the demo is what the planner is *not* told. The obstacle between
the two targets exists only in the simulator; the planner's world contains a
table and nothing else. The arm routes around it anyway, because the cameras
put it in the map.

| | trajectory |
|---|---|
| `--no-mapping` | 81 waypoints — straight through the obstacle |
| mapping on | 121 waypoints — around it, 43 cycles running, 0 failures |

Confirmed three ways, including forward kinematics over the no-map trajectory
showing the arm's own collision spheres passing 10 mm inside the obstacle that
cuRobo considered collision-free. [HANDOVER.md](HANDOVER.md) §7–8 has the
numbers.

---

## Requirements

```
conda activate curobo_isaaclab
```

nvidia-curobo 0.8.0.post1.dev42 · torch 2.7.0+cu128 · Isaac Sim 5.1.0.0 · a
CUDA GPU (developed on an RTX 5080).

**It runs as two processes, and it has to.** Isaac Sim 5.1 pins Warp 1.8.2;
cuRobo 0.8 needs Warp >= 1.13; no version satisfies both. So cuRobo lives in
one process and Isaac Sim in another, talking over a local socket.
`scripts/isaacsim_ur5e_demo.py` **must never import cuRobo** — HANDOVER §2 has
the version matrix and what breaks.

---

## Running it

Two terminals. The server must be listening before the sim starts.

```bash
python scripts/planner_server.py --robot ur5e_2f85 --scene demo_cube
```

```bash
DISPLAY=:1 python scripts/isaacsim_ur5e_demo.py --robot ur5e_2f85 --scene demo_cube
```

Both sides need the **same `--scene`**. They never exchange geometry, so the
client checks on connect and stops rather than planning against a different
world.

Closing the Isaac Sim window does **not** stop the planner server. If a restart
behaves strangely, check that first:

```bash
ss -ltnp | grep 5599
```

### Flags

| flag | side | what it does |
|---|---|---|
| `--scene NAME` | both | which scene to load; must match |
| `--robot NAME` | both | `ur5e_2f85` (has the cameras) or `ur5e` (arm only) |
| `--no-mapping` | both | plan against the static world only — the A/B baseline |
| `--no-overhead` | demo | wrist camera alone, for A/B against the fixed one |
| `--static` | demo | hold at HOME and just look through the cameras |
| `--move-body` | demo | drive the unmapped body along its scene motion |
| `--map-every N` | demo | fuse a frame every N sim steps (default 6) |
| `--depth-lag N` | demo | pair depth with the pose N steps back (default 2) |
| `--no-cuda-graph` | server | build the planner without CUDA graphs |

The mapping scenes need `--robot ur5e_2f85`: `camera_link` is defined in that
URDF, and plain `ur5e` has no camera to map with.

---

## Scenes

| scene | what it is |
|---|---|
| `demo_cube` | the shipped demo — one slab the cameras have to discover |
| `baseline` | the same cell with nothing to discover — the control run |

`baseline` is not an empty world. The table, both cameras and ~18 000 voxels of
mapped table surface are all still there; what is missing is the body the
planner is never told about. Run it when plans start failing and you need to
know whether the obstacle is responsible. That comparison is what caught the
target markers being fused into the map as obstacles sitting on the goals.

### Adding one

Copy `scripts/scenes/demo_cube.py`, edit it, start both processes with
`--scene <your module>`. `scripts/scenes/base.py` is the contract:

```python
SCENE = SceneSpec(
    obstacles=[...],   # cuboids the planner IS told about (table, fixtures)
    targets=[...],     # tool goals [x, y, z, qw, qx, qy, qz]
    home=[...],        # joint rest pose (rad)
    scan_poses=[...],  # swept once at startup to seed the map
    cameras={...},     # each needs "link" (FK) or "pose" (fixed in world)
    mapper={...},      # TSDF/ESDF settings
    unmapped=[...],    # bodies ONLY the simulator knows — what the cameras
                       # have to discover. This is what the demo is about.
    watch=[...],       # volumes to report voxel counts for
    motions={...},     # how an unmapped body moves
)
```

Camera poses are cuRobo **optical** frames (+Z along the view, +X right, +Y
down). The demo derives the ROS body quaternion Isaac Sim wants and then checks
it against the prim's real transform, printing `dot = +1.000 (AGREE)`; it
raises rather than continuing if that disagrees. Getting this wrong silently is
the most expensive mistake available here — HANDOVER §5 explains why.

Anything added to a scene purely to be *looked at* must not be scene geometry:
the cameras will fuse it, and a marker sitting on a goal makes every plan fail.
Target markers are drawn as a viewport overlay for exactly this reason.

---

## Layout

```
scripts/
  scenes/base.py            SceneSpec + WatchBox — the contract
  scenes/demo_cube.py       the shipped scene
  scenes/baseline.py        same cell, nothing to discover
  rig.py                    what is NOT the scene: robots, dt, host/port
  planner_server.py         cuRobo 0.8: planning + mapping service
  isaacsim_ur5e_demo.py     Isaac Sim client. Must not import cuRobo.
  proto.py                  length-prefixed framing (depth frames are 1.2 MB)

configs/                    cuRobo v2 robot configs
assets/robot/               URDFs and meshes
tools/                      model generation, checks and harnesses
```

`tools/` splits in two. **One-time model generation**, rerun only if the robot
changes — `build_ur5e_2f85_urdf.py`, `build_ur5e_2f85_config.py`,
`clip_joint_limits.py`, `convert_2f85_meshes.py`, `convert_v1_robot_yaml.py`.
**Checks and harnesses**, rerun whenever you change something —
`check_robot_cfg.py` (FK → planner build → plan_pose), `ab_solution_spread.py`
(failure rate and joint wander, `--scene` aware), `bench_mapper.py`
(integrate/ESDF timing).

```bash
python tools/check_robot_cfg.py ur5e
python tools/check_robot_cfg.py ur5e_robotiq_2f_85 ur5e_robotiq_2f_85.urdf
```

---

## Robot configs

cuRobo 0.8 ships **no UR5e config** — only `ur10e` — and no UR5e URDF, though
the meshes are still there. `configs/` was converted from the 0.7.7 originals
with `tools/convert_v1_robot_yaml.py` and verified to plan.

| file | |
|---|---|
| `ur5e.yml` | arm only, `tool0` |
| `ur5e_robotiq_2f_85.yml` | arm + gripper + wrist camera; what the demo uses |
| `ur5e_robotiq_2f_140.yml` | converted from 0.7.7, not wired into the demo |

Point cuRobo at them with `ContentPath(robot_config_absolute_path=...,
robot_urdf_absolute_path=..., robot_asset_absolute_path=...)` — see
`tools/check_robot_cfg.py`.

The 2F-85 model is spliced rather than shipped: the arm from `ur5e.urdf`, the
gripper chain from cuRobo 0.7.7's Kinova description, plus a D435i. Its joints
are **fixed** — collision geometry for planning, not an actuated mechanism,
matching cuRobo's own 2F-85 model. Joint limits are deliberately clipped to
±180°; HANDOVER §6 has the measurement behind that.

### v1 -> v2 schema changes applied

| v1 (0.7.x) | v2 (0.8.x) |
|---|---|
| `ee_link` + `link_names` | `tool_frames: [...]` |
| `cspace.retract_config` | `cspace.default_joint_position` |
| `usd_path` / `usd_robot_root` / `isaac_usd_path` / `usd_flip_joints` / `usd_flip_joint_limits` | dropped (not accepted) |
| — | `format_version: 2.0` |

Collision spheres, `self_collision_ignore`, `self_collision_buffer`, `cspace`
limits and `mesh_link_names` carry over unchanged.

**Not carried over:** `ur10e.yml` has a `dynamics:` block pointing at a neural
inverse-dynamics checkpoint. No UR5e equivalent exists and the checkpoint is
not in the repo. `load_dynamics` defaults to `False`, so planning works without
it; the torque-limit-aware part of B-spline trajopt does not apply until such a
model exists.

---

## Where the rest is

[HANDOVER.md](HANDOVER.md) is the measured record — every number in it came
from a run on this machine, with the command to reproduce it. Go there for the
Warp constraint (§2), the upstream cuRobo bugs worked around in-tree (§4), the
unit and convention traps that cost the most time (§5), what mapping does and
does not do (§7), the fixed overhead camera (§8), and the reference numbers to
regress against (§10).
