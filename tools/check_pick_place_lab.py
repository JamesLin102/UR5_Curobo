"""End-to-end check of the Isaac Lab backend, and its A/B against the Isaac Sim one.

tools/check_pick_place.py, with the cell built by Isaac Lab: one command,
its own planner server, headless, an oracle gripping at the true pedestals.
The environment is created the way Isaac Lab's own scripts create one --
isaaclab_tasks' parse_env_cfg() on the registered gym id, then gym.make() --
so passing also proves the task is registered and constructible as any Isaac
Lab task is.

    python tools/check_pick_place_lab.py                    # both modes
    python tools/check_pick_place_lab.py --mapping off      # the fast one
    python tools/check_pick_place_lab.py --both-backends    # + the Isaac Sim check, side by side
    python tools/check_pick_place_lab.py --device cpu       # CPU PhysX

Checked per mode over two seeds, with the same pass criteria as the Isaac Sim
check (see there): every leg plans, the pick holds the block, the place
delivers it; mapped routes stay >= -10 mm from the slab and unmapped ones do
not. Then, once, that Isaac Lab's rsl_rl wrapper accepts the environment.

For comparison, what the Isaac Sim backend measured when this was written:
    mapping off   -26 / -20 / -16 / -20 mm     mapping on   +4 / +10 / +2 / +2 mm

Exit status 0 means every check passed. rig.PORT must be free.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import check_pick_place as base  # noqa: E402  (server plumbing, same bound)
from rig import DEFAULT_ROBOT, HOST, PORT  # noqa: E402

REFERENCE_MM = {False: [-26, -20, -16, -20], True: [4, 10, 2, 2]}


def log(msg):
    print(f"[check-lab] {msg}", flush=True)


# --- child: one mode, inside Isaac Lab ------------------------------------------


def run_mode(args, mapping):
    """Run the oracle episodes. Returns a list of failure strings."""
    from isaaclab.app import AppLauncher

    AppLauncher(dict(headless=True, enable_cameras=mapping, device=args.device))

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import lab  # noqa: F401  (registers the ids)
    from lab.tasks import task_id

    tid = task_id("PickPlace", args.robot)
    cfg = parse_env_cfg(tid, device=args.device, num_envs=1)
    cfg.cell.mapping = mapping
    cfg.cell.verbose = False
    if args.depth_lag is not None:
        cfg.cell.depth_lag = args.depth_lag
    cfg.task.max_legs = 4
    env = gym.make(tid, cfg=cfg)
    u = env.unwrapped
    targets = u.scene_spec.targets
    grasp_z = targets[0][2] - u.scorer.descend
    failures, clearances, placed = [], [], []

    obs, extras = env.reset(seed=0, options={"block_on": 0})
    for seed in (0, 1):
        t0, steps0 = time.time(), u.cell.steps
        r = extras.get("reset", {})
        s, g = r["start"][0], r["goal"][0]
        if s != seed:
            failures.append(f"seed {seed}: episode started on target {s}")
        for leg, idx in (("pick", s), ("place", g)):
            if leg == "place":
                # The place ends the episode; the auto-reset that follows
                # starts the next seed's.
                u.next_reset_options = {"block_on": 1 - seed}
            action = torch.tensor([[targets[idx][0], targets[idx][1], grasp_z, 0.0]])
            obs, rew, term, trunc, extras = env.step(action)
            inf = extras["leg"][0]
            gap = [float(v) for v in re.findall(r"([-+]\d+(?:\.\d+)?) mm", inf.get("clearance") or "")]
            clearances += gap
            log(f"seed {seed} {leg}@{idx}: failed={inf.get('failed')} "
                f"holding={inf['holding']} success={inf['success']} "
                f"clearance={inf.get('clearance')} "
                f"block_to_goal={inf['block_to_goal'] * 1000:.1f} mm "
                f"solve={inf.get('solve_ms', 0):.0f} ms waypoints={inf.get('waypoints')}")
            if "failed" in inf:
                failures.append(f"seed {seed} {leg}: {inf['failed']}")
            if leg == "pick" and not inf["holding"]:
                failures.append(f"seed {seed}: pick is not holding the block")
            if leg == "place":
                placed.append(inf["block_to_goal"] * 1000)
                # From the leg's own report: the auto-reset after a delivery
                # has already put the block back by the time step() returns.
                block = inf["block"]
                d = (block - u.scorer.rest[g]) * 1000
                log(f"  placed at {[round(float(v), 4) for v in block]}, "
                    f"off the rest pose by dx {d[0]:+.1f} dy {d[1]:+.1f} dz {d[2]:+.1f} mm")
                if not inf["success"]:
                    failures.append(f"seed {seed}: block not delivered "
                                    f"({inf['block_to_goal'] * 1000:.0f} mm from the goal)")
            if bool(term[0]) or bool(trunc[0]):
                break
        log(f"seed {seed}: {time.time() - t0:.1f} s, {u.cell.steps - steps0} sim steps")
        if seed == 0 and "reset" not in extras:
            obs, extras = env.reset(seed=1, options={"block_on": 1})

    ref = REFERENCE_MM[mapping]
    if clearances:
        log(f"A/B slab clearance, mm: isaac lab {[round(c) for c in clearances]}  "
            f"vs isaac sim {ref}")
    if placed:
        log(f"placement error, mm: {[round(p, 1) for p in placed]}")
    if not clearances:
        failures.append("the server reported no clearance to the slab")
    elif mapping and min(clearances) < base.BOUND_MM:
        failures.append(f"mapped route came {min(clearances):+.0f} mm from the slab "
                        f"(bound {base.BOUND_MM:+.0f})")
    elif not mapping and min(clearances) > base.BOUND_MM:
        failures.append(f"unmapped route cleared the slab by {min(clearances):+.0f} mm: "
                        f"it is no longer in the way, so the mapped check proves nothing")

    if not mapping and not failures:
        failures += check_rl_wrapper(env)
    return failures


def check_rl_wrapper(env):
    """Does Isaac Lab's own RL plumbing take this environment?"""
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except ImportError as exc:
        log(f"rsl_rl wrapper not installed ({exc}); skipped")
        return []
    wrapped = RslRlVecEnvWrapper(env)
    obs = wrapped.get_observations()
    policy = obs["policy"] if hasattr(obs, "keys") else obs
    ok = tuple(policy.shape) == (1, 25) and wrapped.num_actions == 4
    log(f"rsl_rl wrapper: observations {tuple(policy.shape)}, "
        f"{wrapped.num_actions} actions, max episode length {wrapped.max_episode_length}")
    return [] if ok else [f"rsl_rl wrapper sees observations {tuple(policy.shape)}"]


# --- parent: servers and child processes ----------------------------------------


def check(args, mapping, workdir):
    name = "mapping on" if mapping else "mapping off"
    log(f"=== isaac lab, {name} ===")
    srv_log = os.path.join(workdir, f"server_{'on' if mapping else 'off'}.log")
    server, err = base.start_server(mapping, srv_log)
    try:
        if err:
            log(f"FAIL {name}: planner server {err}\n{base.tail(srv_log)}")
            return False
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--child",
               "--mapping", "on" if mapping else "off", "--device", args.device,
               "--robot", args.robot]
        if args.depth_lag is not None:
            cmd += ["--depth-lag", str(args.depth_lag)]
        child = subprocess.run(cmd, cwd=ROOT, timeout=1800, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
        with open(os.path.join(workdir, f"child_{'on' if mapping else 'off'}.log"), "w") as f:
            f.write(child.stdout)
        for line in child.stdout.splitlines():
            if line.startswith("[check-lab]"):
                print(line, flush=True)
        if child.returncode != 0:
            if "[check-lab] RESULT" not in child.stdout:
                log("child output tail:\n" + "\n".join(child.stdout.splitlines()[-25:]))
            log(f"FAIL {name}")
            return False
        log(f"PASS {name}")
        return True
    except subprocess.TimeoutExpired:
        log(f"FAIL {name}: timed out")
        return False
    finally:
        base.stop(server)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mapping", choices=("on", "off", "both"), default="both")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--robot", default=DEFAULT_ROBOT)
    ap.add_argument("--depth-lag", type=int, default=None)
    ap.add_argument("--both-backends", action="store_true",
                    help="run tools/check_pick_place.py first, for the A/B")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        try:
            failures = run_mode(args, args.mapping == "on")
        except BaseException:
            import traceback
            traceback.print_exc()
            failures = ["the child raised (traceback above)"]
        for f in failures:
            log(f"  - {f}")
        log(f"RESULT {'FAIL' if failures else 'PASS'}")
        sys.stdout.flush()
        os._exit(1 if failures else 0)   # see check_pick_place.py

    if base.port_taken():
        sys.exit(f"[check-lab] {HOST}:{PORT} is already in use -- stop the running "
                 f"planner server first (ss -ltnp | grep {PORT})")
    results = []
    if args.both_backends:
        log("=== isaac sim backend (tools/check_pick_place.py) ===")
        old = subprocess.run([sys.executable, "-u", os.path.join(ROOT, "tools", "check_pick_place.py"),
                              "--mapping", args.mapping], cwd=ROOT)
        results.append(old.returncode == 0)
    modes = {"on": [True], "off": [False], "both": [False, True]}[args.mapping]
    workdir = tempfile.mkdtemp(prefix="check_pick_place_lab_")
    results += [check(args, m, workdir) for m in modes]
    log(f"logs in {workdir}")
    log("ALL PASSED" if all(results) else "FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
