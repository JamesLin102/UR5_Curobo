# Handover — UR5 + cuRobo 0.8 + Isaac Sim

Written 2026-09-21, extended 2026-09-22 when the robot changed. Everything
below was measured on this machine, not taken from documentation. Where a
number appears, it came from a run whose command is given so you can reproduce
it.

**The robot changed on 2026-09-22.** It was a UR5e with a hand-assembled wrist
stack; it is now a UR5 (CB3) with a vendored one. Sections written before that
still say "UR5e" where the reasoning is about the arm in general; §11 is the
conversion and everything it broke, §12 is putting the demo back on its feet
afterwards, and those two are the place to start if something in an older
section does not match what you see.

**Only `pick_place` is left (2026-09-23).** The `demo_cube` and `baseline`
scenes were removed, along with the client's plain two-target loop and
`--move-body`; their shared settings (cameras, mapper, HOME, scan poses) now
live in `scripts/scenes/pick_place.py`, unchanged. Everything below that
measures `demo_cube` or `baseline` is kept as the record it is -- the
reasoning behind the mapper settings, the camera and the slab came from those
runs -- but those scenes can no longer be run from this tree; `git show
0f13431:scripts/scenes/demo_cube.py` has them.

---

## 1. What this is

A UR5 (CB3) carrying a Robotiq FT 300, a Robotiq Wrist Camera, a 2F-85 gripper
and a wrist-mounted RealSense D435i, driven by **cuRobo 0.8.0** inside **Isaac
Sim 5.1**, with live volumetric mapping feeding obstacle data back to the
motion planner. The bare UR5e (`--robot ur5e`) that used to be kept as the
no-gripper option was removed on 2026-09-23, with its meshes and config.

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
scripts/isaacsim_client.py    imports Isaac Sim only, gets Warp 1.8.2
```

`scripts/isaacsim_client.py` **must never import cuRobo.** If you add a
cuRobo import there, everything dies in a confusing way.

If NVIDIA ships an Isaac Sim built on Warp >= 1.13, this split can collapse back
into one process; the planning logic is deliberately isolated in
`planner_server.build()` and the plan/map handlers to make that easy.

---

## 3. Running it

Two terminals. The server must be listening before the sim starts.

```bash
/home/eencku/anaconda3/envs/curobo_isaaclab/bin/python \
  /media/eencku/2TBDATA/yusian-ubuntu/UR5_curobo/scripts/planner_server.py --robot ur5_robotiq
```

```bash
DISPLAY=:1 /home/eencku/anaconda3/envs/curobo_isaaclab/bin/python \
  /media/eencku/2TBDATA/yusian-ubuntu/UR5_curobo/scripts/isaacsim_client.py --robot ur5_robotiq
```

Useful flags: `--scene NAME` (**both sides, must match**), `--no-mapping` (both
sides, for A/B), `--no-overhead` (demo only: wrist camera alone, for A/B
against the fixed one), `--static` (demo only: hold the arm at HOME and just
look through the cameras), `--no-cuda-graph` (server).

The server names its cameras at startup, so a mismatch is visible immediately:

```
[planner] warp 1.15.0  robot ur5_robotiq  scene pick_place
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

5. Isaac Sim side, not cuRobo: **its URDF importer does not sanitise mesh
   FILENAMES into USD prim names.** A hyphen is not legal in a prim name, so
   `robotiq_ft300-G-062-COUPLING_G-50-4M6-1D6_20181119.STL` produces a null
   prim and the **entire** import fails with `RuntimeError: Used null prim` —
   naming no file, so nothing points at the cause. It *does* sanitise link and
   joint names and says so in the log (`The path base_link-base_link_inertia is
   not a valid usd path, modifying to ...`), which is exactly what makes the
   omission easy to miss. The FT 300 coupling plate is therefore stored as
   `ft300_coupling.STL`. Isolated by bisection: swapping that one mesh for a
   box imports fine, renaming it imports fine, converting it to `.obj` does
   **not** — the format was never the problem.

6. Isaac Sim side, not cuRobo: **its COLLADA importer segfaulted** on the
   Robotiq 2F-85 `.dae` meshes from cuRobo's Kinova description
   (`libomniverse_asset_converter` → `tinyxml2`, exit 139, no Python
   traceback), and a converter re-exported them as `.obj`. Both are gone with
   the old model: the vendored `robotiq_description` `.dae` meshes import
   without complaint.

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

**Superseded on 2026-09-22.** The splice-it-yourself pipeline described here is
gone, along with the models it produced; see §11. What replaces it:

```bash
python tools/build_ur5_robotiq_urdf.py     # xacro output -> the project URDF
python tools/build_ur5_robotiq_config.py   # collision spheres -> cuRobo yml
python tools/check_robot_cfg.py
```

Order still matters: the config builder reads the URDF. The joint-limit clip is
now folded into the URDF builder rather than being a separate step run by hand,
because being a separate step is how it got lost during the conversion.

### Joint limits are deliberately clipped to ±180°

Stock UR5e URDF gives five of six joints ±360°. With 720° of range the planner
picks IK branches that wind a wrist right round — legal, collision-free, ugly,
and it eventually strands the arm somewhere the next target is unreachable.

Measured over 60 cycles: **60/60 cycles touched a wound-up pose** before,
**0/60** after, with **no cost** — same 0 failures, same 0.000 mm pose error,
same median trajectory length. `shoulder_pan` peak went 310.6° → 158.6°.

`configs/*.yml` also carry `cspace_distance_weight: [1,1,1,1.5,1.5,1.5]` to
penalise wrist motion. `tools/ab_solution_spread.py` was the harness that
measured this; it was removed on 2026-09-23 (it had stopped working on the
UR5 config, and the question it answered is settled). `git show
e1b5fed:tools/ab_solution_spread.py` has it if the limits ever change.

### Gripper fingers are articulated, but not planned

The 2F-85 is two mirrored 4-bar linkages, and a 4-bar needs a loop closure
URDF cannot express. The approximation every Robotiq ROS package uses is one
driving joint plus `<mimic>` followers — but **the URDF here carries no
`<mimic>` tags**, because PhysX will not have them:

```
Usd Physics: the revolute joint at .../left_inner_knuckle_joint needs a
finite limit set to be used by the mimic joint feature
```

every one of them having `lower="0" upper="0.8757"`. The failure is not
contained: it takes the whole articulation with it, and the fingers visibly
come apart in the viewport. Stripping the tags takes the error count 1 → 0 and
the articulation builds.

The coupling therefore lives in `rig.ROBOTS[...]["gripper"]["joints"]` — joint to
multiplier, ±1 — and three consumers derive from it: the URDF builder takes the
**sign of each joint's limits** from it, the config builder locks all six, and
the simulator commands all six. Change a multiplier and all three follow.

cuRobo 0.7.7's Kinova model — which these links are copied from — made every
gripper joint `fixed`, but left the origins identical to the ros-industrial
`robotiq_2f_85_gripper_visualization` macro. Checked joint by joint, so
restoring the articulation moves nothing at angle 0.

**The planner stays 6 DOF.** The robot config carries
`lock_joints: {left_outer_knuckle_joint: 0.0}`, exactly as `franka.yml` does
with its fingers. Locking is not the same as fixing: cuRobo still places the
finger links and their collision spheres at that angle, so it knows the
gripper's real shape — measured, open to closed moves a sphere 48.5 mm while
every arm sphere moves 0.000 mm.

It is locked **open** on purpose. That is the widest the gripper ever is, so a
route that clears with it open stays clear while it closes. The reverse does
not hold, and a plan made at one lock value is not valid at another.

Three things that bite:

- **The returned trajectory is wider than the cspace.** A 6-DOF plan comes back
  **12** columns here — the six locked joints appended — and only
  `interpolated_trajectory.joint_names` says so. It was 7 when a single joint
  was locked, which is exactly why `handle_plan` selects columns by name:
  anything hard-coded to the width would now be silently off.
- **Signed limits.** A joint with multiplier −1 travels negative, so
  `lower="0"` pins it at zero. Driven explicitly, the four positive joints
  tracked a close command perfectly while both `inner_finger` joints sat at
  −0.000. Upstream's URDF has the same `lower="0"` with `multiplier="-1"`, and
  gets away with it only because a mimic constraint computes the follower
  rather than commanding it through its own limit.
- **Budget the stroke from the velocity limit.** The joints are limited to
  2.0 rad/s, so 0.8 rad needs 0.4 s — 24 steps at 60 Hz. Ten-step ramps left
  the gripper stranded at +0.334 rad when the next plan started. Sized from
  the limit it returns to +0.001..+0.017 rad every cycle.

### The 4-bar is closed in USD, not in the URDF

*(Still true, and still how it works. The link names below are the old model's
-- `inner_finger` is now `finger_tip_link` -- and the anchors are now derived
from the URDF's joint origins rather than from mesh surfaces. The pin alone
also turned out not to be enough on the new model: see §11 fault 9.)*

URDF is a tree, so the loop that makes a 4-bar a 4-bar cannot be written in
one: the inner knuckle hangs off the base as its own branch with nothing tying
it to the finger. Driven open-loop it fights whatever it touches, and measured
while gripping a 45 mm block the branches stalled 0.22 rad apart within one
side — the linkage visibly comes apart at the moment it grips, which is when
anyone looking at it notices.

**USD is not a tree.** `close_gripper_linkage()` adds a `UsdPhysics.
RevoluteJoint` after import, pinning each inner knuckle to its inner finger,
and the mechanism sets the knuckle's angle the way it does on the real gripper.
The knuckles are then not driven at all; only the load-bearing chain is.

The anchors are derived, not guessed: at angle 0 the two links are in their
correct relative pose, so their closest surface points — 4.66 mm apart — are
where the pin goes. Both sides produce identical local coordinates, which is
the check that the derivation is right.

| gripping a 45 mm block | open-loop | pinned |
|---|---|---|
| outer vs inner knuckle, same side | 0.142 rad | **0.059** |
| left vs right asymmetry | 0.043 rad | **0.012** |
| block slip over a 0.125 m lift | 5 mm | **1 mm** |

HANDOVER warned that PhysX handles this mechanism badly. On this one it does
not: no jitter, no constraint blow-up, and the grasp got *better*. The residual
0.059 rad is the pin being an ideal hinge where the real linkage has
millimetre-scale slop in link lengths.

Three things were tried first and are recorded so nobody repeats them:
commanding the followers from the leader's **measured** angle made the
left/right spread **worse** (0.146 → 0.297 rad, because a follower commanded
to the leader's position still cannot get there — contact stops it, not the
command); taking the knuckles out of collision fixed the left/right symmetry
but left 0.22 rad within one side; and `set_joint_positions` to place them
kinematically resets the whole articulation, leaving the arm stuck at its spawn
pose even when only two indices are written.

**Checking the planner is not checking the simulator.** cuRobo resolves the
coupling itself and never goes near PhysX, so throughout the broken-linkage
episode it reported a perfectly correct gripper — spheres moving 48.5 mm over
the stroke — while the fingers were coming apart on screen. The importer
logging that it saw the `<mimic>` tags is not the physics engine agreeing to
enforce them. Verify both sides: PhysX joint angles read back after a
commanded move, and cuRobo's spheres.

The articulation has 12 DOF where the planner has 6, so the demo addresses the
two sets by joint index rather than reindexing whole arrays.

### Grasping

`scenes/pick_place.py` shuttles a 150 g block between two pedestals. The scene
contract gained `payload` — a rigid body with mass, distinct from `unmapped`,
which is visual-only and meant to be avoided.

**The thing you intend to grasp is, to the map, an obstacle**, and `plan_pose`
will not route a tool into one. So the plan goes to a pose ABOVE the block and
the last stretch is solved by IK and interpolated. That needs a **second IK
solver built with no collision checker** — the planner's own is collision-aware
and refuses the descent every time, which showed up as the arm shuttling to the
pre-grasp pose and stopping, over and over, with `no IK for z-0.125 m` the only
clue in the log.

The trade is explicit: nothing checks the approach. It is short, vertical, and
seeded from directly above, which is what makes that acceptable and would not
make it acceptable for a long move.

A deliberately wrong alternative, measured before it was discarded:
approximating the descent as shoulder_lift and elbow moving oppositely gives
`dz/dq = -0.11 m/rad` and `dx/dq = -0.40` — mostly horizontal, and the opposite
sign to the guess it replaced.

### Historical: why the fingers used to be rigid

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

#### Re-measured on the UR5 (2026-09-22): the detour is gone

Same scene, same three rows, 120 s of planning each:

| | trajectory (waypoints) |
|---|---|
| `--no-mapping` | `101` then **81 × 86**, 0 failures |
| no obstacle, mapping on | `101` then **81 × 41**, 0 failures |
| mapping on | `101` then **81 × 41**, 0 failures |

All three identical. The obstacle is not missing from the map — the server
reports 723 voxels with 239 of them inside `drag_me`, spanning z 0.01..0.34,
which is the slab. Mapping is working and changing nothing.

The reason is the arm, and it is measurable. FK over the *unmapped*
trajectory, which on the UR5e passed 10 mm inside the slab:

```
target[0]  101 waypoints, closest approach +46.9 mm  (right finger, x 0.423,
           63 mm past the slab's +x face, level with its top at z 0.348)
target[1]   81 waypoints, closest approach +65.5 mm  (upper arm, x 0.122,
           118 mm past the slab's -x face)
```

The route goes over and around the slab rather than through it. A CB3 shoulder
sits at z = 89 mm against the e-Series' 162.5, so the same two targets are
reached in a different posture.

**`demo_cube`'s obstacle therefore had to be re-tuned**, and it was — see §12.
Short version: the height was never the lever, the position was. The arm
crosses between the targets around x = 0.45, not x = 0.30, and moved there the
slab's original 0.35 m height runs into the route by 20.2 mm.

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

Nothing blocking.

A startup transient appeared when the wrist stack went on and was then fixed
by raising `HOME`, which is worth recording because the two looked unrelated.
The FT 300 and Wrist Camera push the tool 55 mm further out, which put
`grasp_frame` at z=0.350 — exactly the top face of the demo slab — so the
gripper started *inside* it, and the wrist camera spent the opening scan
looking at its own fingers from a few centimetres away. Measured: 35
consecutive plans failed with `6 inside the robot [left_inner_finger:1]` before
frustum decay cleared them.

Raising `HOME` to `[0, -1.8, 1.25, ...]` (clearance −11 mm → +140 mm, tool
z 0.350 → 0.550) removed it entirely: 68 plans, **0 failures**, and `0 on the
robot` in all 351 map reports. Worth knowing that `self_mask_margin` was the
obvious suspect and would have been the wrong fix — the camera being too close
to the fingers was the cause, not the margin being too tight. The obstacle is still the 0.35 m low slab chosen when the
wrist camera's 0.40 m ceiling forced that shape; it gives a stable detour
(section 7), so it has not been changed. Now that height is free, a taller or
off-axis obstacle would be a more natural demo — that is a choice, not a fix.

In rough priority order:

1. Actuate the gripper (simplified 1-DOF fingers; the real 4-bar needs a loop
   closure URDF cannot express).
2. Re-check the detour-vs-block window once coverage is better — with a fixed
   camera the mapped obstacle should match reality more closely.
3. ~~Consider pinning the cuRobo checkout.~~ Done 2026-09-23. `/home/eencku/curobo`
   is an editable install on `main`, 42 commits past v0.8.0, and upstream has
   already moved (`78fd485` by then). It now sits on a local branch
   `ur5-curobo-pin` at `8e734f3` with no upstream, so `git pull` refuses, and
   `planner_server` checks HEAD against `rig.CUROBO_COMMIT` on startup and
   exits on a mismatch (`--allow-curobo-drift` to override). Do not tag the
   pin: cuRobo takes its version from `git describe`, and a non-version tag
   there makes `import curobo` raise.

---

## 9. Files

```
scripts/
  scenes/
    base.py                 SceneSpec + WatchBox: what a scene must and may
                            define. The contract both processes build from.
    pick_place.py           the scene, and the only one. Copy this to make
                            your own. (demo_cube and baseline were removed
                            on 2026-09-23.)
    __init__.py             load(name) -> SceneSpec, available()
  rig.py                    what is NOT the scene: robots, SIM_DT, host/port
  planner_server.py         cuRobo 0.8: planning + mapping service
  planner_client.py         socket client for planner_server; imports
                            neither Isaac Sim nor cuRobo
  sim_env.py                SimEnv: the Isaac Sim side as a library -- stage,
                            cameras, gripper, reset/move_to/move_tool_z/grip/
                            observe. Must not import cuRobo. Builds the
                            cameras; derives the ROS body pose of each fixed
                            one from its optical pose and checks it.
  isaacsim_client.py        the pick-and-place demo loop on top of SimEnv.
  pick_place_env.py         PickPlaceEnv(gymnasium.Env): one step = one leg,
                            action [x, y, z, yaw] where the gripper acts.
  proto.py                  length-prefixed framing (depth frames are 1.2 MB)

tools/  -- model generation (rerun only if the robot changes):
  build_ur5_robotiq_urdf.py    xacro output -> the project URDF, mechanical
                               edits only, each one printed
  build_ur5_robotiq_config.py  collision spheres -> cuRobo yml, every link
                               scored and the build FAILS below its bar

tools/  -- checks and harnesses (rerun whenever you change something):
  check_robot_cfg.py        smoke test: FK -> planner build -> plan_pose
  show_collision_spheres.py robot + spheres in a browser, seconds not minutes

configs/ur5_robotiq.yml     generated by build_ur5_robotiq_config.py

assets/robot/ur5_robotiq/   one folder per device, since 2026-09-23 -- see
                            its PROVENANCE.md for the old -> new mapping
  ur5_robotiq.urdf          mesh paths relative to this file
  source/                   raw xacro output (the builder's input) + xacro
  meshes/{ur5,ft300,wrist_camera,robotiq_2f85,d435i}/{visual,collision}/
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

Useful as regression baselines. **Everything in this table was measured on the
UR5e model**, before the 2026-09-22 conversion; §11 has the UR5's numbers where
they have been re-measured, and says which have not.

| | |
|---|---|
| Planning (arm only) | 23–31 ms, 121 waypoints |
| Planning (2F-85, static scene) | 28 ms median, 61 ms max |
| Planning (2F-85 + live voxel map) | 43 ms median, 99 ms max |
| Mapper integrate (640×480, incl. self-mask + filter) | 1.61 ms/frame |
| Mapper ESDF + `update_world` | 1.3 ms (compute_esdf alone 0.1 ms) |
| Map memory | 22 MB (both cameras) |
| Map size, both cameras | 26 000–31 000 voxels after ~1000 frames, then flat |
| Trajectory tracking error in sim | median 0.09°, 99th pct 0.44° (0.08–0.27 over 71 plans); 0.09–0.19° with the gripper actuating |
| Arm's travel corridor | z 0.4–0.6 m |
| `HOME` tool height / obstacle clearance | z 0.550 m / +140 mm |
| Wrist camera table footprint over a full cycle | x 0.30–0.45, y −0.45–+0.30 |
| Overhead camera footprint at table level | x −0.27–+0.97, y −0.82–+0.82 |
| Full demo, both cameras, 3610 frames | 71 plans, 0 failures, 69 detours, 0 self-hits |

UR5 equivalents, after §11 and §12:

| | |
|---|---|
| Planning (2F-85 + live voxel map) | 47–56 ms, 81 waypoints |
| `demo_cube` | 36–49 plans, 0 failures; −20 mm into the slab unmapped, −2..+11 mapped |
| `pick_place` | 19 plans, 0 failures; grasp settles in 35–43 steps; block on [0.450, ±0.400] |
| `baseline` | 33 plans, 0 failures, 0 voxels (the table is out of the map) |
| Map size, both cameras | 340–700 voxels — the obstacle, and nothing else |
| `HOME` tool height / obstacle clearance | z 0.546 m / +59.5 mm (pick_place), +168.8 (demo_cube) |

First call of anything Warp-backed includes JIT compilation — ESDF's first call
is ~450 ms, then 1 ms. Do not benchmark cold.

---

## 11. The UR5 conversion (2026-09-22), and the nine faults it exposed

The hardware is a UR5 (CB3), not a UR5e. The model in the repo was a UR5e with
a wrist stack assembled here link by link from photographs and datasheets, and
it was wrong in ways nobody could see by looking.

It was replaced wholesale by [eugene900805/mir_ur5_humble] with the MiR100
chassis removed and nothing else changed:
`assets/robot/ur5_robotiq/source/ur5_robotiq.urdf.xacro` re-parents the arm to the
origin and instantiates the same `ur_robot` / `robotiq_wrist_stack` /
`robotiq_gripper` / `sensor_d435i` macros with the mounting transforms copied
from `mir_100_v1.urdf.xacro`. `PROVENANCE.md` beside it has the layout and the
licences.

Nine things then had to be fixed before the demo ran again. Every one was
measured first; the two that were *guessed* at first (the 4-bar pin, the
waypoint count) were both wrong.

### The model itself

**1. UR5 vs UR5e geometry.** The CB3 shoulder sits at z = 89 mm, the e-Series
one at 162.5. Link meshes differ by up to 28 mm (wrist_1). Nothing about the
e-Series model transfers except the *style* of its collision spheres.

**2. The FT 300 coupling was mounted back to front** — upstream, and in
ROS-Industrial's own FT 300 macro. Measured off the STL, the part is stepped:

```
z  0.0 -  4.0   dia 31.5   locating spigot
z  4.0 -  6.5   dia 75     collar, the FT 300's own diameter
z  7.0 - 13.0   dia 63     body, the UR flange's own diameter
```

so dia 63 is the robot face and the spigot locates the SENSOR. Drawn as
shipped it overhangs the wrist by 6 mm and buries itself 6.3 mm inside the
sensor. Turned around, the diameters match at both ends and the stack-up
closes: body 0..6.5, sensor mesh starts at 6.7. Only the drawn geometry moved;
the 41.5 mm FT 300 offset is ROS-Industrial's figure and was not touched.

**3. Collision spheres.** cuRobo's tuned ur5e spheres cannot be reused, but its
approach can: few big spheres that protrude, 27 for the whole arm, none on the
base. Every link now has a stated budget, is fitted best-of-eight, and is
scored with cuRobo's own coverage metric; below its bar the build fails. One
unguarded run had put a single 5 mm sphere on wrist_3, covering 0.4% of it.

Two links clamp their spheres to their own geometry, `shoulder_link` and
`upper_arm_link` — see fault 6.

### Isaac Sim

**4. wrist_1 would not hold still**: 770 mrad of drift at HOME while every
other joint stayed inside 5 mrad, and it read as a loose joint on screen. This
description has 26 links that are pure coordinate frames with no mass and no
geometry, and the importer gives each "a small isotropic inertia"; twenty of
those hanging off the wrist wreck the articulation solver. `merge_fixed_joints`
drops it to 4.4 mrad. The 4-bar pin was the first suspicion and was ruled out
by measurement: 663 mrad with it removed.

Merging discards the merged frames' prims, so `camera_link` no longer exists to
hang the D435i on. `frame_prim()` walks the URDF to the nearest link that does
have a prim and puts an Xform back, which costs PhysX nothing.

**5. Every plan failed**, because eight `base_link_inertia` spheres sit inside
the table cuboid for every configuration — 17.1 mm of penetration, at HOME and
at both goals. NVIDIA leaves the base bare and that is not an oversight.

**6. The goals were then refused by the MAPPED table rather than the real one.**
The planner is told the table exactly; the cameras fusing it too only make a
fatter copy, and on a CB3 the upper arm's spheres live permanently inside that
copy. `mapper["floor_z"]` drops depth below a height before fusion. The map
goes from 16 000 voxels to the 300 that are the obstacle.

**7. With that fixed, plans succeeded at 440 ms** — because the demo had been
run with `--no-cuda-graph` to get that far. With graphs on, everything failed
from the first call. Building the approach IK from the planner's own
`ik_solver_config` corrupts the planner:

```
second solver from the shared config : plan FAIL, FAIL, FAIL
second solver from a deep copy       : plan ok,   ok,   ok
no second solver at all              : plan ok,   ok,   ok
```

Reproduced on cuRobo's shipped ur5e config as well, so it is not this robot. A
`copy.deepcopy` fixes it, the two solvers still behave differently — a pose
buried in the table is refused by the collision-aware one and solved by the
free one — and plans are back to 50 ms. **This is an upstream trap and belongs
with the ones in §4.**

### The gripper

**8. It never closed on the block.** `grasp_frame` is the middle of the pad
face WITH THE GRIPPER OPEN, and the 2F-85's fingers swing rather than
translate, so closing carries the pads 13.5 mm further out: the pad face goes
from tool0 166.3..204.3 to 179.8..217.8. Aimed at the block's centre they
therefore ARRIVE 13.5 mm low, into a pedestal 140 mm across where the gripper
only opens to 85. They stalled against it at 0.04 rad of the 0.8 commanded.
`pick_place` now aims from the constraint that matters: the closed pads clear
the pedestal by 12 mm and still overlap the block by 33.

**9. The linkage came apart, and vertical moves juddered — one fault.**
Isaac Sim 5.x **ignores** the importer's `default_drive_strength` and
`default_position_drive_damping`. Every joint arrives at stiffness 625 and
damping **0**, whatever is asked for; verified by reading the `DriveAPI` back
after import at 1e6 and at 1e5 with identical results. An undamped position
drive rings around a moving target.

Measured over one 12.5 cm descent, as velocity sign flips across the six arm
joints and the velocity ripple on wrist_1:

```
damping    0 -> 138 flips, ripple 0.48, lag  8 mrad
          20 ->  19 flips, ripple 0.13, lag  9 mrad
          50 ->   0 flips, ripple 0.19, lag 22 mrad
         150 ->   0 flips, ripple 0.20, lag 66 mrad
```

20 is the knee. It also fixed the grasp, which did not look like the same
fault: the two sides used to close to 0.34 and 0.48 rad — a block shoved off
centre by the buzzing, not by anything in the gripper. They now agree to 0.01,
the three joints within a side to 0.011, and the close settles in 35 steps
instead of 240.

The gripper's own drives are separately softened to 100/10, and the two joints
the 4-bar pin owns have their drives zeroed entirely. Measured as the spread
across the six joints while gripping a 45 mm block:

```
1e6 -> 0.201     1e4 -> 0.191     1e2 -> 0.064
1e5 -> 0.197     1e3 -> 0.099
```

Solver iterations (64, 255) moved this by 0.000; driving two joints instead of
four by 0.005. Removing the pin moved it to 0.992 — the inner knuckle left
behind at zero while everything else closes. The pin is load-bearing.

### What the measurements cost

Two harnesses were written and one of them was useless. A free-air rig that
closed the gripper on nothing scored **every** configuration at 0.003 and could
not see fault 9 at all; only a rig that parked a block between the pads
reproduced it. Neither is committed — they are scratch — but the lesson is
worth keeping: a gripper harness that does not grip anything measures nothing.

### Where it stands

All three scenes run on the UR5.

```
demo_cube   81 waypoints, 51 ms a plan, 683 voxels of obstacle
pick_place  grasp settles in 35 steps, block lands on [0.450, +/-0.250]
baseline    28 plans, 0 failures, 0 voxels (the table is out of the map)
```

Known and not fixed: the gripper rests at 0.028 rad rather than 0, so the real
gripper opens ~3 mm narrower than the planner's locked-open model (safe
direction); and `settle_gripper` reports `TIMED OUT` on a grasp that is
holding perfectly well, because its idea of "stopped" is stricter than it
needs to be.

The 81-vs-121-waypoint A/B was repeated on the UR5 and did not hold; §12 is
what came of that, including why waypoint count was the wrong thing to be
measuring in the first place.

[eugene900805/mir_ur5_humble]: https://github.com/eugene900805/mir_ur5_humble

---

## 12. Making the demo demonstrate something again (2026-09-23)

§11 left `demo_cube` in a state where the cameras mapped the obstacle
correctly and the route ignored it, because on a CB3 the route was never going
to hit it. Fixing that turned up two more faults and one mistake of my own.

### Height was never the lever; position was

The obvious move is to make the slab taller until it reaches the route. It
works and it is wrong. Counting the route's own collision spheres by x band:

```
x 0.20..0.30:   500 spheres      x 0.42..0.48:  6116
x 0.30..0.38:  1250              x 0.48..0.55:  4677
x 0.38..0.42:  1939
```

The arm crosses between the targets around x = 0.45. From x = 0.30 it took
0.48 m of height to touch the route at all, and at that height the corridor
left over was 16 mm wide: the arm diverted, parked itself against the slab and
could not plan out again, sticking permanently after about 14 cycles. At
x = 0.42 the ORIGINAL 0.35 m height penetrates the route by 20.2 mm and leaves
the goals 145 mm clear.

### Waypoint count is a bad proxy, and it cost most of the search

The 81-vs-121 result that this repo was built on measures trajectory length.
A detour can come back the same length, and with the slab in its new place
every plan is 81 waypoints whether the cameras see it or not. `handle_plan`
now reports the closest the plan comes to each body the planner was never told
about — reporting only, nothing there reaches the planner:

```
demo_cube   --no-mapping   60 plans, all -20 mm
            mapping on     49 plans, 0 failures: one -20 (no map yet),
                           35 between -2 and 0, 14 between +3 and +11
```

It goes from 20 mm through the slab to skimming it, not to clearing it: the
mapped top sits at z = 0.34 against the real 0.35, and the planner clears what
it was shown.

### Removing the base's spheres broke self-masking

The base spheres had to go so the planner would stop calling the robot in
collision (§11 fault 5). RobotSegmenter masks the cameras with those same
spheres, so the cameras then mapped the robot's own base: 13 700 of the map's
14 000 voxels sat between z = 0.02 and 0.15, which is the base (0..0.024) and
the shoulder (0.024..0.157), permanently occupying the space the arm has to
pass through. It took the demo from 0 failures to 74.

The self-hit check could not see it. It counts occupied voxels inside the
robot's collision spheres, so the one link whose spheres were removed is
invisible to it: it reported "0 on the robot" throughout.

Both halves are needed and they conflict, so they are separated. The spheres
live in the config for the segmenter; `planner_server.build()` drops them from
the planner's copy (`rig.ROBOTS[...]["arm"]["mask_only_links"]`, once
`SEGMENTER_ONLY`). Removing either half puts the demo back
in a state that looks like a different bug.

### self_mask_margin 0.12 -> 0.18

0.12 was enough while the arm shuttled along a fixed route and not enough once
it started detouring — the mask began missing, and that leak is what the 74
failures above were. 0.18 took it back to 6, and the base-sphere fix took the
rest.

### pick_place needed its own layout, not demo_cube's

Its slab was de-duplicated into demo_cube's, which moved it onto this scene's
pedestals and made every goal unreachable. Reverted: the size is shared, the
position is not.

Then it needed a layout of its own, because with the pedestals at their
original +/-0.25 there is no slab position that both blocks the route and
leaves the grasps alone. Swept, in mm:

```
ped y   slab x |  route   pre-grasp  at block
 0.25     0.30 |  +25.4      +42.9     +17.2   route misses it
 0.25     0.42 |  -28.5       -1.0     -26.2   fouls the grasp
 0.38     0.42 |  -14.0      +97.0     +43.0   both
 0.40     0.46 |  -18.6     +116.4     +57.0   both, with room
```

So the pedestals moved out to +/-0.40. The height then had to go to 0.46,
because at 0.35 the only legs the slab reached were the short approaches --
the long traverse between pedestals, which is the big visible motion, cleared
it by 116 mm whether the cameras saw it or not. Measured end to end:

```
mapping off   -40 -41 -20 +59 -20 +112 -20 +59 -20 +112 ...
mapping on    -40  +4  -4 +59  +6 +112  -3 +59  +6 +112 ...
```

The +59 and +112 legs are identical either way. That is the honest part: they
were never near it.

### Retracting a rest pose makes the working poses MORE extended

With the taller slab, `HOME` cleared it by 11 mm -- visible on screen -- and
two of the scan poses were 14 mm INSIDE it, so the startup sweep ran the arm
through the obstacle it was scanning for. That one was not visible at all.

The fix is elbow in, wrist_1 back out, so each pose keeps its orientation and
simply sits further back and higher. 0.20 rad was the first attempt and broke
planning completely -- 217 of 218 plans blocked, goals reachable throughout --
because an arm that starts further back reaches the pedestals more extended,
and its upper arm ends up 15 mm off the mapped slab and strands itself. 0.15
is the most that clears every pose without that:

```
                  demo_cube   pick_place
HOME     before     +121.2       +11.2
         after      +168.8       +59.5
scan[1]  before      +93.5       -14.0
         after      +126.1       +21.7
scan[2]  before      +98.5       -14.2
         after      +130.8       +20.8
```

### Where it stands

```
demo_cube   36-49 plans, 0 failures, 51 ms a plan
pick_place  19 plans, 0 failures, block shuttling between [0.450, +/-0.400]
baseline    33 plans, 0 failures, 0 voxels
```

Unexplained and worked around rather than understood: with the tall slab at
x = 0.30, the map grew a band of ~10 000 voxels spanning x +0.24..+1.36,
y -0.90..+1.00 at z 0.02..0.10 -- far wider than the robot or the table --
that the floor filter demonstrably was not letting through (it cut
224 676 of 226 310 overhead pixels). Raising `floor_z` to 0.12 removed it. It
has not reappeared with the slab in its final position and `floor_z` back at
0.02, so it is recorded here rather than chased.
