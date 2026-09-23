# UR5_curobo — camera-driven obstacle avoidance

A UR5 (CB3) carrying a Robotiq FT 300, a Robotiq Wrist Camera, a 2F-85 gripper
and a RealSense D435i, planning with **cuRobo 0.8** inside **Isaac Sim 5.1**.
Depth is fused into a live TSDF/ESDF and handed back to the motion planner
every few frames.

The robot description is **vendored, not assembled here**: it is
[eugene900805/mir_ur5_humble](https://github.com/eugene900805/mir_ur5_humble)
with the MiR100 chassis removed and nothing else changed. The model it replaced
was built link by link from photographs and datasheets, and its geometry was
wrong in ways nobody could see. See
[assets/robot/ur5_robotiq/PROVENANCE.md](assets/robot/ur5_robotiq/PROVENANCE.md).

The point of the demo is what the planner is *not* told. The obstacle between
the two targets exists only in the simulator; the planner's world contains a
table and nothing else. The arm routes around it anyway, because the cameras
put it in the map.

**Measured as the closest the plan comes to that obstacle**, not as the
trajectory's length. Length is a bad proxy and it misled this repo for a while:
a detour can come back the same number of waypoints, and does. The planner
server reports the real thing on every plan — reporting only, nothing there
reaches the planner:

| `demo_cube`, 120 s a row | closest approach to the obstacle |
|---|---|
| `--no-mapping` | 60 plans, **all −20 mm** — straight through it |
| mapping on | 49 plans, 0 failures: one −20 (before the map exists), 35 between −2 and 0, 14 between +3 and +11 |

The first plan of a mapped run is always the unmapped one, because the map does
not exist yet. That single number flipping from −20 to positive, and staying
there, is the demonstration.

Honest about the margin: with the cameras on it goes from 20 mm *through* the
slab to skimming its surface, not clearing it by a comfortable distance. The
mapped top sits at z = 0.34 against the real 0.35 — voxel sampling — and the
planner clears what it was shown.

**The obstacle had to be moved for this arm**, and that is not a regression.
It diverted the UR5e from x = 0.30 and did nothing at all to the UR5: the
unmapped route cleared it by +46.9 mm and +65.5 mm, and every row of the A/B
came out at 81 waypoints. The CB3 shoulder sits 73 mm lower than the e-Series
one, so the same two targets are reached in a different posture. Counting the
route's own collision spheres by x band says where it actually goes:

```
x 0.20..0.30:   500 spheres      x 0.42..0.48:  6116
x 0.30..0.38:  1250              x 0.48..0.55:  4677
x 0.38..0.42:  1939
```

The arm crosses between the targets around x = 0.45, not x = 0.30. Moved
there, its ORIGINAL height penetrates the route by 20.2 mm. Making it taller
where it stood was the obvious move and the wrong one — see HANDOVER §11.

---

## Requirements

```
conda activate curobo_isaaclab
```

nvidia-curobo 0.8.0.post1.dev42 · torch 2.7.0+cu128 · Isaac Sim 5.1.0.0 · a
CUDA GPU (developed on an RTX 5080).

**cuRobo is pinned to commit `8e734f3`** (`rig.CUROBO_COMMIT`). It is an
editable checkout of `main`, not a release, so the checkout sits on a local
branch `ur5-curobo-pin` with no upstream — `git pull` refuses — and
`planner_server` will not start on any other commit, or with local edits to
cuRobo's tracked files. Moving the pin means re-measuring HANDOVER §10 and
re-checking the workarounds in §4 first.

**It runs as two processes, and it has to.** Isaac Sim 5.1 pins Warp 1.8.2;
cuRobo 0.8 needs Warp >= 1.13; no version satisfies both. So cuRobo lives in
one process and Isaac Sim in another, talking over a local socket.
`scripts/isaacsim_ur5e_demo.py` **must never import cuRobo** — HANDOVER §2 has
the version matrix and what breaks.

---

## Running it

Two terminals. The server must be listening before the sim starts.

```bash
python scripts/planner_server.py --robot ur5_robotiq --scene demo_cube
```

```bash
DISPLAY=:1 python scripts/isaacsim_ur5e_demo.py --robot ur5_robotiq --scene demo_cube
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
| `--robot NAME` | both | `ur5_robotiq`, the only robot (default) |
| `--no-mapping` | both | plan against the static world only — the A/B baseline |
| `--no-overhead` | demo | wrist camera alone, for A/B against the fixed one |
| `--static` | demo | hold at HOME and just look through the cameras |
| `--move-body` | demo | drive the unmapped body along its scene motion |
| `--map-every N` | demo | fuse a frame every N sim steps (default 6) |
| `--depth-lag N` | demo | pair depth with the pose N steps back (default 2) |
| `--no-cuda-graph` | server | build the planner without CUDA graphs |
| `--allow-curobo-drift` | server | start on a cuRobo other than the pinned commit (warns) |

---

## Scenes

| scene | what it is | measured on the UR5 |
|---|---|---|
| `demo_cube` | the shipped demo — one slab the cameras have to discover | 36–49 plans, 0 failures, 51 ms a plan; −20 mm into the slab unmapped, −2..+11 mapped |
| `pick_place` | a block shuttled between two pedestals, past that slab | 19 plans, 0 failures; grasp settles in 35–43 steps, block lands on [0.450, ±0.400] |
| `baseline` | the same cell with nothing to discover — the control run | 33 plans, 0 failures, **0 voxels** |

The two slabs are the same size and in **different places**, and that is
deliberate rather than an oversight waiting to be tidied away: `demo_cube`
needs its obstacle on the route, `pick_place` needs its own not to sit on the
grasps. Importing one into the other was tried and made every goal in
`pick_place` unreachable. `pick_place` also had to move its pedestals out to
±0.40 to leave anywhere for a slab to stand.

`baseline` answers one question: with nothing to discover, is the rig healthy?
Run it when plans start failing and you need to know whether the obstacle is
responsible. That comparison is what caught the target markers being fused into
the map as obstacles sitting on the goals.

Its map is now **empty**, and that is expected rather than a symptom: the table
is the only thing in that cell and `mapper["floor_z"]` keeps it out of the map.
Which also means baseline no longer exercises perception at all — for that, run
`demo_cube` and watch the voxel count.

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
    mapper={...},      # TSDF/ESDF settings, incl. "floor_z" (see below)
    unmapped=[...],    # bodies ONLY the simulator knows — what the cameras
                       # have to discover. This is what the demo is about.
    payload=[...],     # rigid bodies to be PICKED UP, with a mass
    pick={...},        # {"descend_m", "lift_m"} for the last few centimetres
    watch=[...],       # volumes to report voxel counts for
    motions={...},     # how an unmapped body moves
)
```

**`mapper["self_mask_margin"]` is 0.18, not 0.12.** 0.12 is enough while the
arm shuttles along a fixed route and not enough once it starts detouring: the
mask began missing, the arm mapped itself, and the demo went from 0 failures
to 74.

**`mapper["floor_z"]`: do not map what the planner already knows exactly.**
Depth below that height is dropped before fusion. The table is a cuboid in
`obstacles`, so mapping it too only produces a second, fatter copy of it — 2.5
cm ESDF voxels plus the planner's collision activation distance. On a UR5 (CB3)
that copy is fatal: the shoulder sits at z = 89 mm, 73 mm lower than the
e-Series arm cuRobo's config was tuned on, and the upper arm's own collision
spheres end up permanently inside it. Measured before the fix: the
collision-aware IK refused **both** goals of `demo_cube` and of `baseline`, on
every attempt, while the map correctly reported nothing inside the robot. The
real table is still in the static scene and still checked.

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
  scenes/pick_place.py      pick a block off one pedestal, place it on the other
  rig.py                    what is NOT the scene: robots, dt, host/port
  planner_server.py         cuRobo 0.8: planning + mapping service
  isaacsim_ur5e_demo.py     Isaac Sim client. Must not import cuRobo.
  proto.py                  length-prefixed framing (depth frames are 1.2 MB)

configs/                    cuRobo v2 robot configs
assets/robot/ur5_robotiq/   the vendored description — see its PROVENANCE.md
  ur5_robotiq.urdf          the one URDF in use; mesh paths relative to it
  source/                   raw xacro output + the xacro, for regenerating
  meshes/ur5/               UR5 (CB3) links        visual/ + collision/
  meshes/ft300/             Robotiq FT 300         visual/
  meshes/wrist_camera/      Robotiq Wrist Camera   visual/
  meshes/robotiq_2f85/      2F-85 gripper          visual/ + collision/
  meshes/d435i/             D435i body + bracket   visual/ + collision/ (hull)
tools/                      model generation, checks and harnesses
```

`tools/` splits in two. **Model generation**, rerun only if the robot changes:

```bash
python tools/build_ur5_robotiq_urdf.py     # xacro output -> the project URDF
python tools/build_ur5_robotiq_config.py   # collision spheres -> cuRobo yml
```

The URDF builder also maps every upstream `package://` mesh to its folder
under `meshes/`, and stops if one has no entry. Both do only mechanical, reviewable edits, and both print what they did. The
config builder scores every link with cuRobo's own coverage metric and **fails
the build** if any link falls below its bar — one unguarded run had put a
single 5 mm sphere on wrist_3, covering 0.4% of it.

**Checks and harnesses**, rerun whenever you change something —
`check_robot_cfg.py` (FK → planner build → plan_pose), `show_collision_spheres.py`
(the robot and its spheres in a browser, seconds rather than minutes),
`ab_solution_spread.py` (failure rate and joint wander, `--scene` aware),
`bench_mapper.py` (integrate/ESDF timing).

```bash
python tools/check_robot_cfg.py
python tools/show_collision_spheres.py ur5_robotiq
```

---

## Robot configs

cuRobo 0.8 ships **no UR5 or UR5e config** — only `ur10e`.
`configs/ur5_robotiq.yml` is generated from the vendored URDF: the whole stack,
`grasp_frame` + `camera_link`. Its cspace weights and limits are cuRobo 0.7.7's
hand-tuned UR5e values, now written into the config builder (`TUNED_CSPACE`);
the bare UR5e option and its `configs/ur5e.yml` were removed on 2026-09-23.

**Rebuilding the config refits the collision spheres, and the fit is not
deterministic.** A rerun on 2026-09-23 came out with different spheres and
missed the builder's own bar on two links (`base_link_inertia` 0.8999,
`robotiq_wrist_camera_link` 0.8982, bar 0.90). The checked-in yml is the
measured one; only rebuild it when the robot changes, and re-measure after.

Point cuRobo at them with `ContentPath(robot_config_absolute_path=...,
robot_urdf_absolute_path=..., robot_asset_absolute_path=...)` — see
`tools/check_robot_cfg.py`.

**The UR5e's spheres are not transferable to the UR5.** The CB3 link meshes are
a different shape — by up to 28 mm on wrist_1 — so cuRobo's hand-tuned ur5e
spheres would land in the wrong places, silently, since nothing checks that a
sphere is on its link. What does carry across is its *style*: few big spheres
that protrude rather than many small ones that follow the surface, 27 for the
whole arm, one 100 mm ball for the shoulder, and **none at all on the base**.

That last one is not an oversight in NVIDIA's config. The base is bolted to the
table, so any sphere on it sits inside the table cuboid for every
configuration — measured at 17.1 mm of penetration — and the planner then calls
the robot in collision whatever it is asked. Eight base spheres were tried here
and did exactly that: every plan failed, with no map loaded at all.

Two links have their spheres clamped to their own geometry, `shoulder_link` and
`upper_arm_link`, because a sphere hanging 14–20 mm below the metal is 14–20 mm
closer to the table, and on a CB3 there is not 14 mm to spare. The generator
explains both, with the numbers.

Joint limits are clipped to ±180°, folded into the URDF builder — HANDOVER §6
has the measurement behind that.

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

## Known quirks

Three things that are true of the current build, none of them a blocker, all of
them expensive to rediscover.

**Isaac Sim 5.1 ignores the URDF importer's drive settings.**
`default_drive_strength` and `default_position_drive_damping` do nothing: every
joint arrives at stiffness 625 and **damping 0**, whatever they are set to —
read the `DriveAPI` back after import at 1e6 and at 1e5 and the values are
identical. An undamped position drive rings around a moving target, which is
what made slow vertical moves judder and, less obviously, what shoved the block
off centre during a grasp. `tune_arm_drives()` and `tune_gripper_drives()` in
the demo set the gains after import, which is the only place that works.

**The gripper does not return to a true zero.** With the softer drives it rests
at 0.028 rad rather than 0, so the real gripper opens about 3 mm narrower than
the planner's locked-open model. That direction is safe — the planner believes
the gripper is wider than it is — but it is a discrepancy, not a rounding.

**Some collision spheres exist only for the cameras.** The base's spheres are
in the config and are *removed from the planner's copy* by `planner_server`,
because both halves are needed and they conflict: the segmenter masks the
robot out of the depth using those spheres, so a base without them is a base
the cameras MAP — 13 700 voxels of it, sitting permanently where the shoulder
and forearm have to pass — while a planner that checks them calls the robot in
collision in every configuration, since the base is bolted to the table. The
self-hit diagnostic cannot see this on its own: it counts voxels inside the
robot's spheres, so the one link without spheres is invisible to it.

**`settle_gripper` occasionally reports `TIMED OUT, still moving`.** Its idea of
"stopped" is stricter than it needs to be, so a grasp that is holding perfectly
well sometimes runs out its 240 steps with the joints still creeping by
microradians. The joint angles it prints are the same as a clean run's, and the
block travels either way. It is a noisy log line, not a failed grasp.

---

## Where the rest is

[HANDOVER.md](HANDOVER.md) is the measured record — every number in it came
from a run on this machine, with the command to reproduce it. Go there for the
Warp constraint (§2), the upstream cuRobo bugs worked around in-tree (§4), the
unit and convention traps that cost the most time (§5), what mapping does and
does not do (§7), the fixed overhead camera (§8), and the reference numbers to
regress against (§10).
