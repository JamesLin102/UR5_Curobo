# UR5_curobo

**Camera-guided pick and place for a UR5 in Isaac Sim, planned by cuRobo.**
The planner is never told where the obstacle is — the cameras find it.

<p align="center">
  <img src="docs/media/pick_place.gif" width="720" alt="pick_place: two round trips past the slab">
  <br>
  <sub>Two round trips, 2× speed. <a href="docs/media/pick_place.mp4">Full quality, real time (MP4)</a></sub>
</p>

A UR5 (CB3) with a Robotiq FT 300, Wrist Camera, 2F-85 gripper and a RealSense
D435i shuttles a block between two pedestals. Between them stands a slab that
exists **only in the simulator**: the planner's world is the table and the
pedestals, nothing else. A wrist camera and a fixed overhead camera fuse depth
into a live TSDF/ESDF with cuRobo 0.8's mapper, and the planner routes around
whatever the map holds.

Closest approach to the slab on the legs that cross it, measured by the planner
server on every plan:

| | plans | closest approach to the slab |
|---|---|---|
| mapping off | 0 failures | **−16 to −26 mm** — straight through it |
| mapping on | 0 failures | **−4 to +11 mm** — around it, once the map exists |

The block lands on its target pedestal to within 1–3 mm either way.

The same cell is also a library (`SimEnv`: reset, move, grip, observe) and a
gymnasium environment (`PickPlaceEnv`) for learning where to grip, with cuRobo
doing every motion in between. It also runs on **Isaac Lab**, as a registered
`DirectRLEnv` task, with the same planner and the same results — see
[Isaac Lab backend](#isaac-lab-backend-experimental).

---

## Requirements

- Ubuntu 22.04, conda
- An NVIDIA GPU and driver **≥ 580.65.06** (cuRobo 0.8's minimum). Developed on
  an RTX 5080 (16 GB). No system CUDA toolkit is needed: cuRobo JIT-compiles
  its kernels at runtime.
- A desktop session for the viewer, or `--headless`

It runs as **two processes, by design**: Isaac Sim 5.1 needs Warp 1.8.2,
cuRobo 0.8 needs Warp ≥ 1.13, and no version satisfies both. cuRobo lives in
`planner_server.py`, Isaac Sim in `isaacsim_client.py`, and they talk over a
local socket.

## Install

**1. The environment** — Python 3.11, torch 2.7.0+cu128, Isaac Sim 5.1:

```bash
conda env create -f environment.yml
conda activate ur5_curobo
```

**2. cuRobo, from source, pinned.** Everything here was measured on commit
`8e734f3`, and `planner_server.py` refuses to start on any other:

```bash
git clone https://github.com/NVlabs/curobo ~/curobo
cd ~/curobo
git switch -c ur5-curobo-pin 8e734f3ced1df898990bcd92de40abce475907db
python -m pip install --upgrade-strategy only-if-needed -e ".[cu13]"
python -m pip install --no-deps "packaging==23.0"
cd -
```

- `[cu13]`, not `[cu12]`, whatever `nvidia-smi` says: Isaac Sim already
  brings the CUDA 13 pip packages, and `[cu12]` would downgrade them.
- The last line puts back the `packaging` version Isaac Sim pins, which the
  cuRobo install upgrades.
- Leave `websockets` at the version pip installs, even though Isaac Sim's
  metadata asks for 12.0; downgrading it breaks cuRobo's robot builder.
- Keep the pin on a branch, not a tag: cuRobo reads its version from
  `git describe`, and a non-version tag makes `import curobo` fail.

**3. First launch.** Isaac Sim asks you to accept its EULA the first time it
starts (or set `OMNI_KIT_ACCEPT_EULA=YES` once you have read it). The first
launch on a machine builds shader and extension caches and can take well over
ten minutes; later ones take seconds.

**4. Check it works:**

```bash
python tools/check_robot_cfg.py      # cuRobo only: FK, planner build, one plan (~1 min)
python tools/check_pick_place.py     # the whole loop, headless, both modes
```

`check_pick_place.py` starts its own planner server, runs two oracle episodes
with mapping off and on, and checks that every pick holds the block, every
place delivers it, the mapped routes clear the slab and the unmapped ones do
not. It ends with `ALL PASSED` (76 s on the development machine, after the
first launch).

## Run the demo

Two terminals, from the repository root. The server must be listening before
the sim starts.

```bash
python scripts/planner_server.py
```

```bash
python scripts/isaacsim_client.py
```

The arm scans the cell, then carries the block back and forth. Drag the slab
(`/World/drag_me`) in the viewport and the next plans go around its new
position. Closing the window does **not** stop the server; stop it with
Ctrl-C (`ss -ltnp | grep 5599` shows a stale one).

| flag | side | |
|---|---|---|
| `--no-mapping` | both | plan against the static world only — the A/B control |
| `--no-overhead` | client | wrist camera alone |
| `--headless` | client | no window; renders only if the cameras need it |
| `--static` | client | hold at HOME and just look through the cameras |
| `--scene NAME` | both | another scene under `scripts/scenes/`; must match |
| `--allow-curobo-drift` | server | start on a cuRobo other than the pinned commit |

## Use it from code

```python
import sim_env                              # scripts/ on sys.path
sim_env.launch(headless=True)               # before anything else from Isaac Sim
env = sim_env.SimEnv(sim_env.EnvCfg(mapping=False))
env.reset(sim_env.ResetOptions(block_on=0))
env.move_to(target)                         # planned by cuRobo
env.move_tool_z(-0.125)                     # straight down, by IK
env.grip(close=True)
obs = env.observe()                         # joints, tool pose, block pose, holding
```

As a gymnasium environment, one step is one pick or place, and the action is
where the gripper acts:

```python
from pick_place_env import PickPlaceEnv
env = PickPlaceEnv(headless=True)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step([x, y, z, yaw])
```

An oracle delivers in 2 steps; an episode takes about 2 s headless with mapping
off and about 20 s with it on (each reset rescans). The planner server must be
running, with `--no-mapping` to match `EnvCfg(mapping=False)`.

## Isaac Lab backend (experimental)

The cell rebuilt on Isaac Lab 2.3 (`scripts/lab/`), next to the Isaac Sim one
and sharing its robot, scenes, planner server and demo loop. It registers
`Isaac-PickPlace-Ur5Robotiq-v0` — one id per task per robot in `rig.ROBOTS` —
so `gymnasium.make()`, Isaac Lab's `parse_env_cfg()` and its RL wrappers all
take it. One environment today, built to become many.

```bash
python scripts/planner_server.py                      # as before
python scripts/lab/isaaclab_client.py --device cpu    # the demo, on Isaac Lab
python tools/check_pick_place_lab.py --device cpu --both-backends   # A/B, headless
```

It needs Isaac Lab installed from source into the same environment. Install
steps, the A/B numbers, what differs between the backends and why, how to add
robots, scenes and tasks, and the plan for many environments are in
[docs/isaaclab.md](docs/isaaclab.md).

## Layout

```
scripts/
  planner_server.py     cuRobo: planning + live mapping, on a local socket
  planner_client.py     its client; needs neither Isaac Sim nor cuRobo
  sim_env.py            Isaac Sim side as a library: SimEnv
  isaacsim_client.py    the demo
  pick_place_env.py     PickPlaceEnv (gymnasium)
  lab/                  the Isaac Lab backend: robots, scene, cell, gym tasks
  cell_api.py           what every backend offers: config, results, primitives
  pick_place_task.py    the task itself: legs, rewards, observations
  demo_loop.py          the demo loop, for either backend
  sim_usd.py, urdf_frames.py   USD and URDF helpers both backends use
  rig.py                robots (arm + gripper settings), timing, port, cuRobo pin
  scenes/               base.py = the scene contract, pick_place.py = the scene
configs/                cuRobo robot config (generated)
assets/robot/           the robot description, one folder per device
tools/                  checks, a collision-sphere viewer, model builders
```

A new scene is a copy of `scripts/scenes/pick_place.py`; a new robot is an
entry in `rig.ROBOTS` plus its URDF and cuRobo config. Both backends pick either
up without changes; docs/isaaclab.md lists the rest of the extension points.

## Limitations

- **The margin is thin.** Mapped routes skim the slab (−4 to +11 mm) rather
  than clear it comfortably. Fine for a simulation demo, not a safety margin
  for hardware.
- **Simulation only.** The overhead camera exists only in the simulator, and
  `PickPlaceEnv` observes the block's true pose. A real cell needs a second
  calibrated camera (or a lower obstacle) and object perception.
- The gripper log sometimes says `TIMED OUT, still moving` on a grasp that is
  holding fine: its idea of "stopped" is stricter than it needs to be.

## More

- Why each setting is what it is — drive gains, mapper settings, the scene's
  layout, the upstream cuRobo bugs worked around — is written next to the
  setting, with the measurement behind it: `scripts/rig.py`,
  `scripts/scenes/pick_place.py`, `scripts/planner_server.py`,
  `scripts/sim_env.py`.
- [assets/robot/ur5_robotiq/PROVENANCE.md](assets/robot/ur5_robotiq/PROVENANCE.md)
  — where the robot description comes from. It is vendored from
  [eugene900805/mir_ur5_humble](https://github.com/eugene900805/mir_ur5_humble)
  with the MiR100 chassis removed and nothing else changed.

## License

MIT — see [LICENSE](LICENSE). The vendored robot description keeps its own
licences, listed in PROVENANCE.md.
