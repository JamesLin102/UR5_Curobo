# The Isaac Lab backend

The same cell as `scripts/sim/sim_env.py` — the same robot, scene, planner server and
demo loop — built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and
registered as a gymnasium task:

```python
env = gymnasium.make("Isaac-PickPlace-Ur5Robotiq-v0", cfg=cfg)   # a DirectRLEnv
```

**Status: many environments.** One cell reproduces the Isaac Sim backend's
results (below); N cells run side by side in lockstep, each with its own
cameras, map and planner server, and pass the same checks in every cell. See
[Many environments](#many-environments).

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
python scripts/pick_place/isaaclab_client.py --device cpu
```

The flags are `isaacsim_client.py`'s (`--no-mapping`, `--no-overhead`,
`--static`, `--scene`, `--robot`, `--map-every`, `--depth-lag`) plus Isaac Lab's
own (`--headless`, `--device`, …) and a few for this backend: `--markers`,
`--camera-class camera|tiled`, `--merge-inertial`, `--force-usd`, `--planner none`.
`--enable_cameras` is implied whenever mapping is on. The slab to drag is
`/World/envs/env_0/drag_me`.

**Use `--device cpu`.** GPU PhysX costs about 12 ms a step for one articulation
and about 20 ms for 8 or 32, while CPU PhysX is 0.8 ms for one: with this
robot's 64/16 solver iterations the GPU's fixed cost per step dominates, and
at 32 environments CPU is still about 3× faster (measured below).

Several cells at once: one planner server per cell, then `--num_envs`:

```bash
python scripts/planner_servers.py --num 4
```

```bash
python scripts/pick_place/isaaclab_client.py --device cpu --num_envs 4 --camera-class tiled
```

Every cell shuttles its own block. One env step is one leg in every cell; the
oracle picks each cell's next leg from its goal. With `--no-mapping` the
planner's world is the static scene for everybody, so cells can share servers:
`planner_servers.py --num 2 --no-mapping` and `--num-servers 2`.

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
python tools/check_pick_place_lab.py --device cpu --num-envs 4 --camera-class tiled
python tools/lab_inspect_robot.py --headless --device cpu --num_envs 4 --replicate-physics --mapping
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
| mapping on, slab clearance | +4, +10, +2, −3 or +2 mm | +5, +9, +2, +6 mm |
| placement error | 1.9–2.0 mm | 2.2–2.7 mm |
| plan failures | 0 | 0 |
| one seed (pick + place), mapping off | 1.8–2.0 s | 0.8–1.4 s |
| one seed, mapping on (reset rescans) | 17–23 s | 11–12 s |

The Isaac Sim side's last mapped leg varies from run to run between −3 and
+2 mm; every other number above has been identical in every run.

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
  legs.py              one grip as ops: above, down, grip, up
  sim_usd.py           USD fixes both backends make (4-bar pin, pads, URDF colours, frames, overlay)
  urdf_frames.py       frame arithmetic read off the URDF
  planner_servers.py   N planner_server.py processes on consecutive ports
  scenes/              the SceneSpec contract, and the registry that finds each example's scene
  lab/
    robots.py          rig.ROBOTS[key] -> ArticulationCfg, any robot
    scene_cfg.py       SceneSpec -> assets under /World/envs/env_*, cameras by kind
    cell.py            LabCell (N environments) and CellView (one, as a CellLike)
    programs.py        ops interpreted one physics tick at a time, all envs together
    planner_pool.py    the planner per environment: a server each, or shared
    markers.py         goal markers the depth cameras cannot see
    cell_cfg.py        CellCfg: every runtime knob
    app.py             the shared command line
    tasks/             __init__ (registry of every example's task), base.py (CellEnv)
  sim/                 the Isaac Sim backend (sim_env.SimEnv), frozen
  pick_place/          the example: scene, task (simulator-free), lab_env (the
                       Isaac Lab task), isaaclab_client (the demo), demo_loop,
                       and the Isaac Sim env and client
```

Dependencies only point one way: an example → the backends → the shared
modules. The shared modules import no backend and no example, neither backend
imports the other or any example, and no example imports another. Inside an
example, files named `lab_*` / `isaaclab_*` are the Isaac Lab side,
`isaacsim_*` the Isaac Sim side, and everything else must be simulator-free
(`tools/check_lab_modularity.py` checks all of it).

## Extending it

Everything plugs into the registries the Isaac Sim side already had:

| to add | do this | picked up by |
|---|---|---|
| a robot | an entry in `rig.ROBOTS` (URDF, cuRobo config, joints, gains, drive type, gripper) | both backends; a gym id per task, `Isaac-{Task}-{Robot}-v0` |
| a gripper mechanism | a handler in `sim_usd.LINKAGES`, named by the rig's `gripper.linkage` | both backends |
| a scene | an example folder `scripts/<name>/` with a `scene.py` defining a `SceneSpec` | both backends, the server, `--scene <name>`; optional fields may be left out |
| a camera kind | a builder in `lab/scene_cfg.CAMERA_BUILDERS` | every scene that uses the kind |
| a task | the example's `lab_env.py` with a `CellEnvCfg` / `CellEnv` subclass, and a line in `lab/tasks/TASKS` | the gym registry, for every robot |
| a planner transport | a `PlannerPool` subclass in `lab/planner_pool.py`, chosen by `CellCfg.planner_mode` | the cell, unchanged |

A task supplies `programs(actions)` (one list of `cell_api` ops per environment),
`reset_cells()`, and the usual `_get_dones` / `_get_rewards` / `_get_observations`.
The cell runs every environment's program in lockstep.

## Many environments

What makes N cells N copies of one:

- Every per-cell prim is under `/World/envs/env_*`, and every pose the planner or
  a task sees is env-local (`scene.env_origins`). The cells stand 4 m apart,
  outside each other's mapper grid and camera range.
- The cell's state is `(num_envs, …)` arrays and the primitives take `env_ids`.
  `programs.py` advances every environment's program one tick at a time,
  and sends the planner requests made on the same tick to their servers in
  parallel.
- **One planner server per cell** (`scripts/planner_servers.py`, ports
  `PORT + i`), because one server holds one map and a cell's map must hold only
  what its own cameras saw. Planning is exactly the single-cell server's. With
  mapping off the world is the static scene for all, and `CellCfg.num_servers`
  lets cells share: env `e` uses server `e % num_servers`.
- USD edits (the pins, the pad material, the URDF colours) go onto every cell's
  robot after cloning. With `--replicate-physics` the PhysX replicator carries
  the pins too: closing on nothing, the gripper spread is 0.000 in every cell.
- The camera alignment check runs on every cell, at start-up and after the first
  reset.

Measured 2026-09-24, `tools/check_pick_place_lab.py`, two seeds, every cell
passing the single-cell criteria (every leg plans, every pick holds, every
place delivers; mapped routes ≥ −10 mm from the slab):

| cells | mapping | device, cameras | servers | one seed (all cells) | slab clearance, worst | placement |
|---|---|---|---|---|---|---|
| 1 | on | cpu, Camera | 1 | 11–12 s | +2 mm | 2.2–2.6 mm |
| 2 | on | cpu, Camera | 2 | 19–21 s | −4 mm | 2.3–2.6 mm |
| 4 | on | cpu, Camera | 4 | 35–37 s | −4 mm | 2.2–2.6 mm |
| 4 | on | cpu, TiledCamera | 4 | 20–22 s | −4 mm | 2.2–2.6 mm |
| 4 | on | cuda:0, TiledCamera | 4 | 42–43 s | −4 mm | 1.2–2.8 mm |
| 1 | off | cpu | 1 | 0.8–1.4 s | — | 2.3–2.7 mm |
| 8 | off | cpu | 2 | 3.5–5.2 s | — | 2.2–2.8 mm |
| 8 | off | cuda:0 | 2 | 19–21 s | — | 1.1–2.9 mm |
| 32 | off | cpu | 2 | 7–8 s | — | 2.2–2.7 mm |
| 32 | off | cuda:0 | 2 | 22–23 s | — | 1.1–2.9 mm |
| 4 | off | cpu, `--replicate-physics` | 4 | 2–4 s | — | 2.3–2.7 mm |

So with mapping off, 32 cells do a seed in about 7 s where one cell takes about
1 s: 4–5× the throughput. With mapping on the cameras are the cost —
TiledCamera, one render product for all of them, nearly halves it — and four
cells get through about twice what one does.

Limits and what would lift them:

- **Servers.** A planner server takes about 2 GB of RAM and 0.55 GB of GPU
  memory, so with mapping on (one each) this 31 GB machine runs about 8 cells.
  Beyond that, one server planning for every cell at once: cuRobo 0.8's
  `BatchMotionPlanner` (`multi_env=True`, `max_batch_size=N`; a private module,
  `curobo/_src/motion/motion_planner_batch.py`) with each cell's ESDF in its own
  collision world (`load_collision_model(world_i, env_idx=i)`) and a map per
  cell. It does not retry, so plan failures would need re-measuring. It would be
  one more `PlannerPool` in `lab/planner_pool.py`; nothing above that changes.
- **GPU PhysX** has a fixed cost of about 20 ms a step at these solver
  iterations, however many cells. CPU wins at every size measured; the crossover
  is somewhere past 32.
- **Rendering.** Every step renders every camera, though only every
  `map_every`-th step fuses. Rendering only on fusing steps (headless) and
  lower-resolution cameras are the next savings.

Lockstep means one env step takes as long as the slowest cell's leg; the
others hold still meanwhile. That keeps the vector-env contract rsl_rl, skrl and
Stable-Baselines3 assume (the rsl_rl wrapper takes the 32-cell environment:
observations `(32, 25)`).
