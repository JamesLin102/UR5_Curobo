# Handover — UR5e + cuRobo 0.8 + Isaac Sim

Written 2026-09-21. Everything below was measured on this machine, not taken
from documentation. Where a number appears, it came from a run whose command is
given so you can reproduce it.

---

## 1. What this is

A UR5e (optionally with a Robotiq 2F-85 gripper and a wrist-mounted RealSense
D435i) driven by **cuRobo 0.8.0** inside **Isaac Sim 5.1**, with live volumetric
mapping feeding obstacle data back to the motion planner.

Environment: `conda activate curobo_isaaclab`
(nvidia-curobo 0.8.0.post1.dev42, torch 2.7.0+cu128, Isaac Sim 5.1.0.0, RTX 5080)

A second env, `curobo077_isaaclab`, holds cuRobo 0.7.7 for the older projects.
**Do not mix them.** 0.8.0 is a full rewrite; no v1 API survives.

---

## 2. The hard constraint that shapes everything: Warp

Isaac Sim 5.1 bundles **Warp 1.8.2**. cuRobo 0.8 needs **Warp >= 1.13**
(it calls `wp.func(..., module=...)`, which 1.8.2 lacks). Warp restructured its
package at 1.13, so:

| Warp version | Isaac Sim | cuRobo 0.8 |
|---|---|---|
| 1.8.2 | works | `TypeError: func() got an unexpected keyword argument 'module'` |
| 1.12.1 | works | fails |
| 1.13 – 1.15 | `AttributeError: module 'warp.types' has no attribute 'array'` | works |

**There is no version that satisfies both.** I checked the wheels individually.
Importing cuRobo first to win `sys.modules` does not help — it just moves the
breakage to Isaac Sim's entire extension stack (`core.api`, `core.prims`,
`simulation_manager`, `replicator`, sensors all fail to start).

### Consequence: two processes

```
scripts/planner_server.py     imports cuRobo,  gets Warp 1.15   (port 5599)
        ^  JSON + raw float32 depth over a local socket (scripts/proto.py)
        v
scripts/isaacsim_ur5e_demo.py imports Isaac Sim only, gets Warp 1.8.2
```

`scripts/isaacsim_ur5e_demo.py` **must never import cuRobo.** If you add a
cuRobo import there, everything dies in a confusing way.

If NVIDIA ships an Isaac Sim built on Warp >= 1.13, this split can collapse back
into one process; the planning logic is deliberately isolated in
`planner_server.build()` and the plan/map handlers to make that easy.

---

## 3. Running it

Two terminals. The server must be listening before the sim starts.

```bash
/home/eencku/anaconda3/envs/curobo_isaaclab/bin/python \
  /media/eencku/2TBDATA/yusian-ubuntu/UR5_curobo/scripts/planner_server.py --robot ur5e_2f85
```

```bash
DISPLAY=:1 /home/eencku/anaconda3/envs/curobo_isaaclab/bin/python \
  /media/eencku/2TBDATA/yusian-ubuntu/UR5_curobo/scripts/isaacsim_ur5e_demo.py --robot ur5e_2f85
```

Useful flags: `--scene NAME` (**both sides, must match**), `--no-mapping` (both
sides, for A/B), `--no-overhead` (demo only: wrist camera alone, for A/B
against the fixed one), `--static` (demo only: hold the arm at HOME and just
look through the cameras), `--move-body` (demo only: drive the unmapped body
along its scene motion), `--no-cuda-graph` (server).

The server names its cameras at startup, so a mismatch is visible immediately:

```
[planner] warp 1.15.0  robot ur5e_2f85  scene demo_cube
[planner] cameras: wrist (camera_link), overhead (fixed)
```

**If the sim connects but behaves strangely, check for a stale server first:**

```bash
ss -ltnp | grep 5599
```

A previous server holding the port makes the new one exit with
`OSError: [Errno 98]` while the sim happily connects to the *old* code. This
cost me a debugging cycle. Kill with `fuser -k 5599/tcp`.

---

## 4. Upstream bugs found in cuRobo 0.8 main (all worked around in-tree)

1. **`RobotSegmenter` default `ops_dtype=torch.bfloat16` is unusable.** Its own
   tensor check accepts only float16/float32, so the default always raises.
   `RobotSegmenter.from_robot_file()` does not expose the parameter — construct
   `RobotSegmenter(...)` directly with `ops_dtype=torch.float32`.

2. **`Mapper.integrate()` dereferences `rgb_image` unconditionally** even though
   colour is nominally optional — pass a dummy RGB tensor or it raises
   `AttributeError: 'NoneType' object has no attribute 'shape'`.

3. **`update_world()` applies one refresh late.** Measured: push grid-with-box,
   plan → unaffected; push grid-without-box, plan → the box appears. Not the
   root cause of anything here, but be aware.

4. **`RobotSegmenter` freezes its projection rays on the first frame.**
   `get_pointcloud_from_depth()` calls `update_camera_projection()` only while
   `_projection_rays is None`, so the rays are built once from whichever camera
   arrives first and never rebuilt. Sharing one segmenter across two cameras
   therefore self-masks the second one with the first one's intrinsics —
   silently, if the resolutions happen to match. `planner_server.Mapping` keeps
   **one segmenter per camera** for this reason.

5. Isaac Sim side, not cuRobo: **its COLLADA importer segfaults** on the Robotiq
   2F-85 `.dae` meshes (`libomniverse_asset_converter` → `tinyxml2`, exit 139,
   no Python traceback). trimesh reads the same files fine, so
   `tools/convert_2f85_meshes.py` re-exports them as `.obj`.

---

## 5. Two unit/convention traps that cost the most time

### `depth_to_meter` defaults to 0.001

`CameraObservation.depth_to_meter` assumes **millimetres**, because that is what
RealSense hardware reports. Isaac Sim's `distance_to_image_plane` is already in
**metres**. Without `depth_to_meter=1.0` the whole point cloud collapses to ~1 mm
from the lens, lands inside the robot's own collision spheres, and the
self-mask removes **100% of the image** — the map stays permanently empty while
every log line looks healthy.

**When you move to real hardware, take this back out.** The default is correct
for a real D435i.

### Isaac Sim's `Camera` takes ROS body axes, not raw USD

cuRobo's mapper kernels use the optical convention (+Z forward, +X right,
+Y down — see `wp_raycast_pose_refine.py`, the ray is `((u-cx)/fx, (v-cy)/fy, 1)`).
USD cameras look down their own −Z. But `isaacsim.sensors.camera.Camera` takes
orientation in **ROS body axes** (view = +X, +Y left, +Z up), so the
"obvious" 180°-about-X correction is a **no-op on the view axis** and leaves the
camera staring sideways down its own wrist.

The correct quaternion is `(w,x,y,z) = (0.5, 0.5, -0.5, 0.5)`, plus a −90° yaw
baked into `camera_optical_joint` in the URDF so image-horizontal runs along the
camera body's long edge.

`attach_wrist_camera()` verifies all of this at runtime and prints it:

```
view in camera_link axes:      [0, -0, 1]    (want [0 0 1])
image-right vs body long edge: dot = -1.000  (want -1.000)
```

The roll check is **signed** on purpose. An earlier version compared `|dot|`,
which passes both +90° and −90° — i.e. it happily accepted an upside-down image.

---

## 6. Robot model

cuRobo 0.8 ships **no UR5e config** (only `ur10e`), and no UR5e URDF either,
though the ur5e meshes are still there. Also note the cuRobo **0.7.7 checkout at
`/home/eencku/curobo-0.7.7` has 224 mesh files that are Git-LFS pointer stubs**
(cloned without `git lfs pull`, and git-lfs is not installed). Planning never
noticed because collision uses spheres from the YAML, not meshes. Fetch
individual files through GitHub's `/raw/` path, which resolves LFS:

```bash
curl -sL -o out.dae "https://github.com/NVlabs/curobo/raw/v0.7.7/src/curobo/content/assets/robot/kinova/kortex_description/grippers/robotiq_2f_85/meshes/visual/robotiq_arg2f_85_outer_finger.dae"
```

### Regenerating the model

```bash
python tools/build_ur5e_2f85_urdf.py      # splice arm + 2F-85 + wrist camera
python tools/clip_joint_limits.py assets/robot/ur_description/ur5e_robotiq_2f_85.urdf --deg 180
python tools/build_ur5e_2f85_config.py    # collision spheres + cuRobo yml
python tools/check_robot_cfg.py ur5e_robotiq_2f_85 ur5e_robotiq_2f_85.urdf
```

Order matters: the config builder reads the URDF. `.orig` backups of both URDFs
sit beside them.

### Joint limits are deliberately clipped to ±180°

Stock UR5e URDF gives five of six joints ±360°. With 720° of range the planner
picks IK branches that wind a wrist right round — legal, collision-free, ugly,
and it eventually strands the arm somewhere the next target is unreachable.

Measured over 60 cycles: **60/60 cycles touched a wound-up pose** before,
**0/60** after, with **no cost** — same 0 failures, same 0.000 mm pose error,
same median trajectory length. `shoulder_pan` peak went 310.6° → 158.6°.

`configs/*.yml` also carry `cspace_distance_weight: [1,1,1,1.5,1.5,1.5]` to
penalise wrist motion. `tools/ab_solution_spread.py` is the harness that
measured this; re-run it if you change limits.

### Gripper fingers are rigid

The 2F-85 joints are fixed, matching cuRobo's own Kinova 2F-85 model. The real
4-bar linkage needs a loop-closure joint URDF cannot express and PhysX handles
badly. So: collision geometry yes, actuation no. `grasp_frame` sits at
0.130324 m above the gripper base (summed along the finger chain).

---

## 7. Mapping: what works and what does not

### Verified working

`tools/` and the scratch probes established, with evidence:

- Camera geometry, orientation and roll — verified numerically and by eye.
- Self-masking — `0 on the robot` across entire runs.
- Map builds and is stable — with both cameras it rises for roughly 1000 fused
  frames and then plateaus at **26 000–31 000 voxels** for the rest of the run
  (~9000–12000 with the wrist alone). Bounded by frustum decay, not growing.
- **Depth-fused ESDF genuinely blocks planning.** Minimal repro: synthetic
  depth placing a wall across the route → `plan_pose` FAILED, while the same
  planner with no map returned n=121. The pipeline is sound end to end.
- **A/B on the live sim finally separates** (see below).
- **Avoidance produces a stable detour**, not just a refusal. Longest run
  measured: 71 plans, **0 failures**, 69 of them on a detour route (58 + 9 at
  121 waypoints, 2 at 161), over 3610 fused frames. `0 on the robot` in every
  one of its 361 map reports.

### The current demo result: a stable detour

`scene_def.DRAG_CUBE` is an obstacle that exists **only in Isaac Sim** — the
planner is never told about it. Same scene, only difference is what the map
knows. Both cameras, 120 s of planning per row:

| | trajectory (waypoints) |
|---|---|
| `--no-mapping` | `101 121 81` then **81 forever** — drives straight through |
| no obstacle, mapping on | `101 101 121` then **81 × 53** — clean baseline |
| mapping on | `101 101` then **121 × 43, zero failures** — diverts every cycle |

**This is the proof that live mapping drives avoidance**, and it now produces a
*detour* rather than a refusal: 43 consecutive cycles took the 121-waypoint
route instead of the 81-waypoint one, and not one plan failed.

It used to block instead — see the marker bug below, which was the real cause.

### Visualisation markers must never reach the depth image

The two `TARGETS` markers are `VisualCuboid`s showing where the tool is being
sent. They render, so the cameras fused them into the map — as obstacles
sitting exactly on the goals. The planner was then being asked to move the tool
inside an obstacle, and simply failed.

Measured with the cube shrunk to 0.02 m, i.e. **no real obstacle in the scene
at all**:

| | plans failed | longest failure streak | voxels above the table |
|---|---|---|---|
| markers as geometry | **193 / 199** | 82 | 30 — `x 0.44..0.45, y ±0.33, z 0.24..0.28` |
| markers as `guide` | **0 / 56** | 0 | 0 — but invisible to you as well |
| markers as overlay | **0 / 33** | 0 | 0 — and visible in the viewport |

That voxel extent is the markers themselves (`0.45, ±0.32, 0.25`, 0.04 cube).
With the fix the same scene plans cleanly 53 times in a row at 81 waypoints.

The fix went through two rounds, and the first one was half a fix.

`UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)` — USD's own
term for geometry that is an authoring aid rather than part of the scene — does
keep them out of the depth image. But guides are hidden in the **viewport**
too, so the map became correct and there was nothing left for a person to look
at. Correct map, unusable demo.

They are now drawn as a **viewport overlay** instead, via
`isaacsim.util.debug_draw` (`draw_targets()` in the demo): a point plus a small
axis cross at each goal. An overlay is produced by a separate pass that render
products do not sample, so it cannot reach a depth annotator by construction
rather than by configuration — visible to you, invisible to the cameras.

The lesson generalises: a goal marker should not be scene geometry at all.
Anything added purely to be looked at belongs in an overlay.

**This bug predates the overhead camera and poisoned every earlier
obstacle-size measurement**, including the table in the next section: those
runs attributed to the obstacle failures that were partly the markers. A wrist
camera only catches the markers occasionally, so it looked intermittent; a
fixed camera watches them continuously, which turned it into near-total
failure and finally made it visible. Anything added to the scene purely to be
looked at needs the same treatment.

### Obstacle size

The mapped obstacle is fatter than the real one (ESDF cell size + collision
activation distance), so size still decides the outcome. Re-measured with both
cameras after the marker fix, 120 s of planning each:

| size | live result |
|---|---|
| 0.11 × 0.31 × 0.33 | no effect — 52 clear routes at 81 waypoints |
| **0.12 × 0.35 × 0.35** | **detour — 43 routes at 121, 0 failures** |

Earlier notes described this window as unusably narrow and said no size gave a
reliable detour rather than a block. That was wrong, and the marker bug above
is why: those runs were measuring the markers, not the obstacle. The wrist-only
figures for `0.10 × 0.26 × 0.30` (no effect) and the `0.50 m` tall post (no
effect — top invisible) were taken under the same conditions and have not been
re-measured; the tall-post row is in any case obsolete now that the fixed
camera can see above 0.40 m (section 8).

The `~0.40 m` mapping ceiling that shaped this obstacle applied to the wrist
camera alone and no longer binds — but this obstacle has not been re-tuned to
take advantage of it, because low-and-wide already gives a stable detour.

### Mapper settings, and why

`scene_def.MAPPER` — all of these were wrong at some point and cost a run each:

- `decay_factor: 1.0` — this is applied to **every voxel every frame**, not a
  time constant. 0.99 eroded the map 9456 → 6946 voxels over 550 frames; 0.3
  wiped it within a few frames. Blind decay eats geometry the camera cannot
  see, which is never what you want. Clearing belongs to frustum decay.
- `frustum_decay_factor: 0.97` — decay for voxels the camera looks *through*,
  i.e. the honest "I can see that spot and it is empty now" signal. This is
  what clears an object's old position after it moves, and **it must stay below
  1.0**. It was briefly 1.0; a full run measured what that costs: the map grew
  monotonically to 75 412 voxels (healthy is 9 000–12 000) and self-mask
  artifacts accumulated until one lodged inside the arm permanently, after
  which every plan failed and never recovered. At 0.97 the same run stays
  between 16 000 and 21 300 voxels and recovers instead of deadlocking. Over a longer
  run it settles at 26 000–31 000 and stays there. 0.85 was too aggressive: it
  wiped surfaces faster than they could be re-observed.
- `self_mask_margin: 0.12` — 0.05 is too tight for a wrist camera; leaked
  gripper pixels get fused and then the robot's own start state reads as in
  collision, after which every plan fails.

### Depth/pose synchronisation

Isaac Sim's depth annotator trails the physics by a step or two. Pairing a frame
with the *current* joint state misaligns the self-mask and the arm smears its own
image into the map as phantom obstacles — the logs showed 0 self-hits during the
slow scan, then hundreds once trajectory playback started.

`--depth-lag` (default 2) compensates by pairing depth with the pose from N
steps back. An earlier fix gated fusion on "arm is stationary", which works but
**starves the map** (420 voxels instead of 6500) — do not go back to that.

On real hardware the D435i has proper timestamps; align on those instead.

---

## 8. The fixed overhead camera

A single wrist camera fundamentally cannot support cross-cell avoidance: its
coverage is whatever the arm happens to sweep, it cannot see above its own
altitude, and the measured footprint on the table during a full cycle is only
`x 0.30..0.45, y -0.45..+0.30`. Everything outside that is a blind spot. The
demo obstacle had to be shaped and placed to fit inside that band — backwards
from how a real cell should work.

`scene_def.CAMERAS["overhead"]` adds a fixed camera 1.20 m above the cell
looking straight down. Its footprint at table level is `y -0.82..+0.82`,
`x -0.27..+0.97`, which covers the workspace and lands inside the mapper grid.

### Measured: the altitude ceiling is gone

A 0.70 m post (`0.12 x 0.35 x 0.70` at `x=0.30`), scan sweep only, no planning
(`--static`), counting occupied voxels inside the post's own bounding box:

| | voxels in map | in the post's bbox | **z reached** |
|---|---|---|---|
| `--no-overhead` | 7 765 | 362 | **0.01 – 0.36** |
| both cameras | 17 708 | 473 | **0.01 – 0.69** |

The wrist-only map stops dead at 0.36 m on a 0.70 m object — the ~0.40 m
ceiling, exactly as section 7 predicted. With the fixed camera the map reaches
0.69 m, i.e. the whole post. Tall-voxel extent went `z 0.16..0.39` →
`z 0.16..0.70`. Both runs reported `0 on the robot`.

Reproduce with `--static` on each side and compare the `INSIDE-CUBE z..` field.

### How it is wired

Each camera is integrated with its **own** `integrate()` call rather than
batched: a fixed camera never moves, so batching would drag it into the
depth-lag bookkeeping it does not need. Verified that two poses fused through
separate calls land in one map exactly where the geometry predicts, with
`num_cameras=1`.

`scene_def` stores every camera as a cuRobo **optical** frame (+Z along the
view). The demo derives the ROS body quaternion Isaac Sim wants, rather than
storing a second set of magic numbers, and then checks the result against the
prim's real transform — `dot = +1.000 (AGREE)` in the startup log. It raises
rather than continuing if that check fails.

### Still open

Nothing blocking. The obstacle is still the 0.35 m low slab chosen when the
wrist camera's 0.40 m ceiling forced that shape; it gives a stable detour
(section 7), so it has not been changed. Now that height is free, a taller or
off-axis obstacle would be a more natural demo — that is a choice, not a fix.

In rough priority order:

1. Actuate the gripper (simplified 1-DOF fingers; the real 4-bar needs a loop
   closure URDF cannot express).
2. Re-check the detour-vs-block window once coverage is better — with a fixed
   camera the mapped obstacle should match reality more closely.
3. Consider pinning the cuRobo checkout. `/home/eencku/curobo` is an **editable
   install sitting on `main`, 42 commits past the v0.8.0 tag** — a `git pull`
   silently changes behaviour. Current commit was `8e734f3`.

---

## 9. Files

```
scripts/
  scenes/
    base.py                 SceneSpec + WatchBox: what a scene must and may
                            define. The contract both processes build from.
    demo_cube.py            the shipped scene. Copy this to make your own.
    baseline.py             demo_cube minus the body the cameras have to
                            find. The control run: if this fails, the
                            problem is not the obstacle.
    __init__.py             load(name) -> SceneSpec, available()
  rig.py                    what is NOT the scene: robots, SIM_DT, host/port
  planner_server.py         cuRobo 0.8: planning + mapping service
  isaacsim_ur5e_demo.py     Isaac Sim client. Must not import cuRobo.
                            Builds the cameras; derives the ROS body pose of
                            each fixed one from its optical pose and checks it.
  proto.py                  length-prefixed framing (depth frames are 1.2 MB)

tools/  -- one-time model generation (rerun only if the robot changes):
  build_ur5e_2f85_urdf.py   splice UR5e + 2F-85 + wrist camera
  build_ur5e_2f85_config.py collision spheres + cuRobo yml
  clip_joint_limits.py      ±360° -> ±180°, keeps .orig backups
  convert_2f85_meshes.py    .dae -> .obj (Isaac Sim's COLLADA importer crashes)
  convert_v1_robot_yaml.py  cuRobo v1 -> v2 robot yaml schema

tools/  -- checks and harnesses (rerun whenever you change something):
  check_robot_cfg.py        smoke test: FK -> planner build -> plan_pose
  ab_solution_spread.py     headless: failure rate + joint wander, --scene aware
  bench_mapper.py           mapper integrate/ESDF timing, synthetic input
```

### Adding a scene

Copy `scripts/scenes/demo_cube.py`, edit it, and start **both** processes with
`--scene <your module name>`. `scenes/base.py` documents every field; the short
version is that `obstacles` are the cuboids the planner is told about,
`unmapped` are the bodies only the simulator knows — the ones the cameras have
to discover — and `watch` names volumes to report voxel counts for, which is
how you tell whether the map actually found something.

The two processes never exchange geometry, so they must load the same scene.
The client sends its scene name when it connects and stops before building the
stage if the server is running a different one, rather than silently planning
against a different world.

When plans start failing, run `--scene baseline` before touching the obstacle.
It keeps the table, the cameras and the map (~18 000 voxels of table surface)
and removes only the thing the cameras are supposed to discover, so a failure
there is never the obstacle's fault. That is what caught the target markers.

Git repo since the baseline commit. The overhead camera went in on the
`overhead-camera` branch.

---

## 10. Measured reference numbers

Useful as regression baselines.

| | |
|---|---|
| Planning (arm only) | 23–31 ms, 121 waypoints |
| Planning (2F-85, static scene) | 28 ms median, 61 ms max |
| Planning (2F-85 + live voxel map) | 43 ms median, 99 ms max |
| Mapper integrate (640×480, incl. self-mask + filter) | 1.61 ms/frame |
| Mapper ESDF + `update_world` | 1.3 ms (compute_esdf alone 0.1 ms) |
| Map memory | 22 MB (both cameras) |
| Map size, both cameras | 26 000–31 000 voxels after ~1000 frames, then flat |
| Trajectory tracking error in sim | median 0.09°, 99th pct 0.44° (0.08–0.27 over 71 plans) |
| Arm's travel corridor | z 0.4–0.6 m |
| Wrist camera table footprint over a full cycle | x 0.30–0.45, y −0.45–+0.30 |
| Overhead camera footprint at table level | x −0.27–+0.97, y −0.82–+0.82 |
| Full demo, both cameras, 3610 frames | 71 plans, 0 failures, 69 detours, 0 self-hits |

First call of anything Warp-backed includes JIT compilation — ESDF's first call
is ~450 ms, then 1 ms. Do not benchmark cold.
