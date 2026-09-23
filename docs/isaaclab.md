# The Isaac Lab backend

The same cell as `scripts/sim_env.py` — the same robot, scene, planner server and
demo loop — built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and
registered as a gymnasium task:

```python
env = gymnasium.make("Isaac-PickPlace-Ur5Robotiq-v0", cfg=cfg)   # a DirectRLEnv
```

**Status: phase 1, one environment.** It reproduces the Isaac Sim backend's
results (below) and is built so that phase 2, many environments at once, is an
extension rather than a rewrite. See [Phase 2](#phase-2-many-environments).

Both backends live side by side. Nothing about the Isaac Sim one changed except
that the pieces both need moved into shared modules.

## Still two processes

Isaac Lab 2.3 runs on Isaac Sim 5.1 and its Warp 1.8.2, and cuRobo 0.8 needs
Warp ≥ 1.13, so cuRobo stays in `planner_server.py` and the Isaac Lab process
reaches it over the same socket. Nothing under `scripts/lab/` imports cuRobo, and
`tools/check_lab_modularity.py` fails if anything starts to.

## Install

Isaac Lab, from source, into the **same** environment as everything else. It was
tested at commit `b4c3210247` (`VERSION` 2.3.2, extension `isaaclab` 0.54.4):

```bash
git clone https://github.com/isaac-sim/IsaacLab ~/IsaacLab
cd ~/IsaacLab
git checkout b4c321024792976150ca55fddb26fa34480d974e
./isaaclab.sh --install rsl_rl
python -m pip install --no-deps "packaging==23.0"
cd -
```

- `--install rsl_rl` installs Isaac Lab's packages plus one RL library, which the
  check uses to prove the environment plugs into Isaac Lab's RL wrappers.
  `--install none` is enough to run the demo.
- The last line restores the `packaging` version Isaac Sim pins, as after the
  cuRobo install. Then check that `python scripts/planner_server.py` still starts.
- The development machine already had this Isaac Lab installed (editable, in the
  same conda env as cuRobo), so these are Isaac Lab's standard steps at that
  commit rather than a sequence re-run from scratch here.

## Run

```bash
python scripts/planner_server.py                 # as before
```

```bash
python scripts/lab/isaaclab_client.py --device cpu
```

The flags are `isaacsim_client.py`'s (`--no-mapping`, `--no-overhead`,
`--static`, `--scene`, `--robot`, `--map-every`, `--depth-lag`) plus Isaac Lab's
own (`--headless`, `--device`, …) and a few for this backend: `--markers`,
`--camera-class camera|tiled`, `--merge-inertial`, `--force-usd`, `--planner none`.
`--enable_cameras` is implied whenever mapping is on. The slab to drag is
`/World/envs/env_0/drag_me`.

**Use `--device cpu` for one cell.** GPU PhysX costs about 12 ms per step for a
single articulation, CPU PhysX 0.8 ms — the GPU only pays for itself once there
are many environments to step at once.

## Use it from code

As any Isaac Lab task: start the app first, then make the environment from its
registered id. One step is one pick or place per environment, exactly as
`PickPlaceEnv` on the Isaac Sim side.

```python
import sys; sys.path.insert(0, "scripts")
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, device="cpu").app       # before anything else from Isaac

import gymnasium as gym, torch
import lab                                               # registers the ids
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

cfg = parse_env_cfg("Isaac-PickPlace-Ur5Robotiq-v0", device="cpu", num_envs=1)
cfg.cell.mapping = False                                 # the server needs --no-mapping to match
env = gym.make("Isaac-PickPlace-Ur5Robotiq-v0", cfg=cfg)
obs, extras = env.reset(seed=0, options={"block_on": 0})
obs, reward, terminated, truncated, extras = env.step(torch.tensor([[x, y, z, yaw]]))
extras["leg"][0]                                         # what the leg did: plan, clearance, grip, ...
```

The primitives underneath, the same API as `sim_env.SimEnv`:

```python
cell = env.unwrapped.cell.view(0)       # one environment as a cell_api.CellLike
cell.move_to(target); cell.move_tool_z(-0.125); cell.grip(close=True)
obs = cell.observe()
```

`cfg.cell` (`lab/cell_cfg.py`) holds every runtime setting: robot, scene,
mapping, cameras, markers, depth lag, planner mode.

## Check

```bash
python tools/lab_inspect_robot.py --headless --mapping   # the robot, cameras and gripper, no planner
python tools/check_pick_place_lab.py --device cpu        # the whole loop, both modes
python tools/check_pick_place_lab.py --device cpu --both-backends   # A/B, side by side
python tools/check_lab_modularity.py                     # import rules, registry, every scene
python tools/test_leg_ops.py                             # shared task logic vs the Isaac Sim env
```

`check_pick_place_lab.py` builds the environment the way Isaac Lab's own scripts
do (`parse_env_cfg` on the gym id, then `gym.make`), runs the same oracle and
pass criteria as `check_pick_place.py`, and checks that Isaac Lab's rsl_rl
wrapper accepts the environment.

## A/B against the Isaac Sim backend

`tools/check_pick_place_lab.py --device cpu --both-backends`, 2026-09-23: the
same oracle, two seeds per mode, closest approach to the slab on each leg as the
planner server reports it, and how far each place left the block from its rest
pose on the goal pedestal.

| | Isaac Sim (`sim_env`) | Isaac Lab (`lab/`) |
|---|---|---|
| mapping off, slab clearance | −26, −20, −16, −20 mm | −26, −20, −16, −20 mm |
| mapping on, slab clearance | +4, +10, +2, +2 mm | +5, +9, +2, +6 mm |
| placement error | 1.9–2.0 mm | 2.2–2.7 mm |
| plan failures | 0 | 0 |
| one seed (pick + place), mapping off | 1.8–2.0 s | 0.8–1.4 s |
| one seed, mapping on (reset rescans) | 17–23 s | 11–12 s |

The Isaac Sim side's last mapped leg read −3 mm before both backends started
applying the URDF's colours (`sim_usd.apply_urdf_colors`), +2 mm after; nothing
else moved.

On `--device cuda:0` the Isaac Lab side passes as well (mapping on: +5, −4, +2,
+5 mm; placement 1.2–2.9 mm), but takes 13–33 s a seed: see the note on GPU
PhysX under [Run](#run).

With mapping off the two backends drive the same routes, to the millimetre:
the planner is given the same start states. With mapping on the routes depend on
what each map holds, and the Isaac Lab side's pairs each depth frame with the
pose it was rendered at on every step (see below).

## What differs, and why

Each of these was measured, not assumed:

- **Drive type.** Isaac Sim's URDF importer makes *acceleration* drives, and
  every gain in `rig.ROBOTS` was tuned on those. Isaac Lab's converter defaults
  to *force* drives. With the same numbers as force drives, the 2F-85's 14–40 g
  links are orders of magnitude stiffer and overpower the 4-bar's pin: one finger
  stalled at 0.1 rad while the other pushed the block 24 mm across, and every
  place landed 13 mm off. `rig.ROBOTS[...]["drive_type"]` now names it and both
  backends apply it.
- **Gain units.** The rig's gains are USD's, per degree. Isaac Lab actuators take
  SI, per radian, so `lab/robots.py` multiplies by 180/π (without it the arm would
  be 57× softer). `lab_inspect_robot.py` reads the gains back from PhysX.
- **Fixed-joint merging.** Isaac Lab's converter pins URDF importer 2.4.31 and
  also merges fixed links that have mass; Isaac Sim's own 2.4.30 merges only
  massless frames. The Lab side does what the Isaac Sim side does by default (21
  bodies either way); `--merge-inertial` gives Isaac Lab's behaviour.
- **Depth lag: 0, not 2.** Isaac Lab renders synchronously. Measured by
  `tools/lab_measure_depth_lag.py`: a teleported block and a 0.3 rad arm jump both
  show in the depth read right after the step that caused them, in both cameras.
- **Camera poses.** Under Fabric, `Camera.data.pos_w` is not updated for a camera
  riding on a robot link, although the image is rendered from the right place.
  The start-up check therefore tests what the mapper consumes instead: it
  back-projects each depth image through the pose the frame will be paired with
  (forward kinematics for the wrist camera, the scene's pose for the fixed one)
  and requires the points to land on the scene's surfaces (median 0.0 mm).
- **Cameras are not in `scene.sensors`.** Registered there, `scene.reset()` calls
  `Camera.reset()`, whose pose re-read goes through an `XformPrimView` that, on
  the GPU pipeline, copies the camera's (stale) USD world transform into Fabric
  on its first read. From then on a camera riding a robot link renders with a
  wrong offset: on `cuda:0` the wrist depth sat 34–45 mm off forward kinematics
  for the whole first episode, the map filled with phantom obstacles, and a
  place goal was refused. The cell updates its cameras itself instead, and
  re-runs the alignment check after the first reset. CPU was never affected
  (the view reads USD directly there).
- **Goal markers.** Isaac Lab's `VisualizationMarkers` turned out to be
  invisible to the depth cameras (0 pixels changed, against 2019 for a plain
  sphere of the same size), so `--markers usd` is safe; it checks this at start
  and falls back if not. The default stays the viewport overlay, as on the
  Isaac Sim side.
- **Joint history for depth pairing** is kept on every step. The Isaac Sim side
  only records it while moving or scanning, so its first fused frame of a move
  can be paired with a pose from before the last descend.
- **Solver iterations** (64/16, now `rig.ROBOTS[...]["solver_iterations"]`) are
  set before the simulation starts. Holding HOME, the worst joint drifts
  2.2 mrad here (4.4 mrad was the Isaac Sim side's figure).

## Layout

```
scripts/
  cell_api.py          the contract: EnvCfg, results, CellLike, ops, run_ops_blocking
  pick_place_task.py   the task, simulator-free: legs as ops, Scorer (rewards, obs)
  demo_loop.py         the demo loop, for any CellLike
  sim_usd.py           USD fixes both backends make (4-bar pin, pads, URDF colours, frames, overlay)
  urdf_frames.py       frame arithmetic read off the URDF
  lab/
    robots.py          rig.ROBOTS[key] -> ArticulationCfg, any robot
    scene_cfg.py       SceneSpec -> assets under /World/envs/env_*, cameras by kind
    cell.py            LabCell (N environments) and CellView (one, as a CellLike)
    programs.py        ops interpreted one physics tick at a time, all envs together
    planner_pool.py    the planner per environment (one server today)
    markers.py         goal markers the depth cameras cannot see
    cell_cfg.py        CellCfg: every runtime knob
    app.py             the shared command line
    isaaclab_client.py the demo
    tasks/             __init__ (registry), base.py (CellEnv), pick_place.py
```

Dependencies only point one way: `lab/` → the shared modules → `rig`, `scenes`.
The shared modules import neither backend, and neither backend imports the other
(`tools/check_lab_modularity.py`).

## Extending it

Everything plugs into the registries the Isaac Sim side already had:

| to add | do this | picked up by |
|---|---|---|
| a robot | an entry in `rig.ROBOTS` (URDF, cuRobo config, joints, gains, drive type, gripper) | both backends; a gym id per task, `Isaac-{Task}-{Robot}-v0` |
| a gripper mechanism | a handler in `sim_usd.LINKAGES`, named by the rig's `gripper.linkage` | both backends |
| a scene | `scripts/scenes/<name>.py` defining a `SceneSpec` | both backends, the server, `--scene`; optional fields may be left out |
| a camera kind | a builder in `lab/scene_cfg.CAMERA_BUILDERS` | every scene that uses the kind |
| a task | `lab/tasks/<name>.py` with a `CellEnvCfg` / `CellEnv` subclass, and a line in `lab/tasks/TASKS` | the gym registry, for every robot |
| a planner transport | a `PlannerPool` subclass in `lab/planner_pool.py`, chosen by `CellCfg.planner_mode` | the cell, unchanged |

A task supplies `programs(actions)` (one list of `cell_api` ops per environment),
`reset_cells()`, and the usual `_get_dones` / `_get_rewards` / `_get_observations`.
The cell runs every environment's program in lockstep.

## Phase 2: many environments

Phase 1 was built so this is an extension:

- Every per-cell prim is already under `/World/envs/env_*`, and every pose the
  planner or a task sees is env-local (`scene.env_origins`).
- The cell's state is already `(num_envs, …)` arrays, the primitives take
  `env_ids`, and `programs.py` already advances every environment's program one
  tick at a time and batches the planner requests made on the same tick.
- USD edits (the pins, the pad material) are applied to every environment's
  robot after cloning.

What is left:

1. **The planner server.** One server holds one map. The plan: `--num-envs N` on
   the server; cuRobo 0.8's `BatchMotionPlanner` (`multi_env=True`,
   `max_batch_size=N`; a private module in 0.8,
   `curobo/_src/motion/motion_planner_batch.py`) with each environment's ESDF
   loaded into its own collision world (`load_collision_model(world_i,
   env_idx=i)`), and one `Mapping` per environment. A backward-compatible protocol: an optional `"env"` field on
   every op (default 0), `plan_batch` / `ik_batch` ops, `num_envs` in the
   handshake. At N = 1 it keeps today's `MotionPlanner` path, so today's numbers
   stay valid. The batch planner does not retry, so plan failures need
   re-measuring. A `BatchPool` in `planner_pool.py` is the client side.
   (For bring-up, one server per environment on `PORT + i` works without planner
   changes, but does not scale on 16 GB.)
2. **Replicated physics and the pins.** With `replicate_physics=True`, whether the
   PhysX replicator carries the `ExcludeFromArticulation` pin joints must be
   tested (spread ≤ 0.01 in every environment, closing on nothing). If not, keep
   `replicate_physics=False`: slower to start, correct by construction.
3. **Cameras.** 2 cameras × 640×480 per environment is the real cost. Switch to
   `--camera-class tiled` (one render product; `distance_to_image_plane` is
   supported), render only on the steps that fuse, and lower the resolution in
   the scene if needed.
4. **Environment spacing** ≥ 3.5 m, so no cell's cameras map its neighbour
   (4.0 m today).

Lockstep means one env step takes as long as the slowest environment's leg; the
others hold still meanwhile. That keeps the vector-env contract rsl_rl, skrl and
Stable-Baselines3 assume.
