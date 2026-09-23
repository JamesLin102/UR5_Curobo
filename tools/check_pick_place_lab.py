"""End-to-end check of the Isaac Lab backend, and its A/B against the Isaac Sim one.

tools/check_pick_place.py, with the cell built by Isaac Lab: one command, its
own planner servers, headless, an oracle gripping at the true pedestals. The
environment is created the way Isaac Lab's own scripts create one --
isaaclab_tasks' parse_env_cfg() on the registered gym id, then gym.make() --
so passing also proves the task is registered and constructible as any Isaac
Lab task is.

    python tools/check_pick_place_lab.py                    # both modes, one env
    python tools/check_pick_place_lab.py --mapping off      # the fast one
    python tools/check_pick_place_lab.py --both-backends    # + the Isaac Sim check, side by side
    python tools/check_pick_place_lab.py --device cpu       # CPU PhysX
    python tools/check_pick_place_lab.py --num-envs 4       # four cells, four servers
    python tools/check_pick_place_lab.py --num-envs 8 --mapping off --num-servers 2

Checked per mode over two seeds, in every environment, with the same pass
criteria as the Isaac Sim check (see there): every leg plans, the pick holds
the block, the place delivers it; mapped routes stay >= -10 mm from the slab
and unmapped ones do not. Environment e starts its block on target (e + seed)
% 2, so both directions run at once. Then, once, that Isaac Lab's rsl_rl
wrapper accepts the environment.

With several environments each one has its own planner server (its own map),
started here by scripts/planner_servers.py; with mapping off they may share
fewer (--num-servers), since the planner's world is then the same for all.

For comparison, what the Isaac Sim backend measured when this was written:
    mapping off   -26 / -20 / -16 / -20 mm     mapping on   +4 / +10 / +2 / -3..+2 mm

Exit status 0 means every check passed. The servers' ports (rig.PORT and up)
must be free.
"""

import argparse
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import check_pick_place as base  # noqa: E402  (the pass bound, the old check)
from rig import DEFAULT_ROBOT, HOST, PORT  # noqa: E402

REFERENCE_MM = {False: [-26, -20, -16, -20], True: [4, 10, 2, 2]}


def log(msg):
    print(f"[check-lab] {msg}", flush=True)


def gaps(info):
    return [float(v) for v in re.findall(r"([-+]\d+(?:\.\d+)?) mm", info.get("clearance") or "")]


# --- child: one mode, inside Isaac Lab ------------------------------------------


def run_mode(args, mapping):
    """Run the oracle episodes in every environment. Returns failure strings."""
    from isaaclab.app import AppLauncher

    AppLauncher(dict(headless=True, enable_cameras=mapping, device=args.device))

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import lab  # noqa: F401  (registers the ids)
    from lab.tasks import task_id

    n = args.num_envs
    tid = task_id("PickPlace", args.robot)
    cfg = parse_env_cfg(tid, device=args.device, num_envs=n)
    cfg.cell.mapping = mapping
    cfg.cell.verbose = False
    cfg.cell.num_servers = args.num_servers
    cfg.scene.replicate_physics = args.replicate_physics
    if args.camera_class:
        cfg.cell.camera_class = args.camera_class
    if args.depth_lag is not None:
        cfg.cell.depth_lag = args.depth_lag
    cfg.task.max_legs = 4
    t_build = time.time()
    env = gym.make(tid, cfg=cfg)
    u = env.unwrapped
    log(f"{n} env(s) built in {time.time() - t_build:.1f} s")
    targets = u.scene_spec.targets
    grasp_z = targets[0][2] - u.scorer.descend
    failures, clearances, placed = [], [], []

    t0 = time.time()
    obs, extras = env.reset(seed=0, options={"block_on": [e % 2 for e in range(n)]})
    log(f"reset (scan {'on' if mapping else 'off'}): {time.time() - t0:.1f} s")
    for seed in (0, 1):
        t0, steps0 = time.time(), u.cell.steps
        r = extras.get("reset", {})
        start = dict(zip(r["env_ids"], r["start"]))
        goal = dict(zip(r["env_ids"], r["goal"]))
        for e in range(n):
            if start.get(e) != (e + seed) % 2:
                failures.append(f"env {e} seed {seed}: episode started on target {start.get(e)}")
        for leg in ("pick", "place"):
            if leg == "place":
                # The place ends the episode; the auto-reset that follows
                # starts the next seed's, in every environment.
                u.next_reset_options = {"block_on": [(e + seed + 1) % 2 for e in range(n)]}
            idx = start if leg == "pick" else goal
            action = torch.tensor([[targets[idx[e]][0], targets[idx[e]][1], grasp_z, 0.0]
                                   for e in range(n)])
            ts = time.time()
            obs, rew, term, trunc, extras = env.step(action)
            dt = time.time() - ts
            for e in range(n):
                inf = extras["leg"][e]
                clearances.append(gaps(inf))
                tag = f"env {e} seed {seed} {leg}@{idx[e]}"
                line = (f"{tag}: failed={inf.get('failed')} holding={inf['holding']} "
                        f"success={inf['success']} clearance={inf.get('clearance')} "
                        f"block_to_goal={inf['block_to_goal'] * 1000:.1f} mm")
                bad = "failed" in inf or not (inf["holding"] if leg == "pick" else inf["success"])
                if n == 1 or bad:
                    log(line)
                if "failed" in inf:
                    failures.append(f"{tag}: {inf['failed']}")
                if leg == "pick" and not inf["holding"]:
                    failures.append(f"{tag}: pick is not holding the block")
                if leg == "place":
                    placed.append(inf["block_to_goal"] * 1000)
                    if not inf["success"]:
                        failures.append(f"{tag}: block not delivered "
                                        f"({inf['block_to_goal'] * 1000:.0f} mm from the goal)")
            log(f"seed {seed} {leg}: one env step, {n} leg(s) in lockstep, {dt:.1f} s")
        log(f"seed {seed}: {time.time() - t0:.1f} s, {u.cell.steps - steps0} sim steps")
        if seed == 0 and len(extras.get("reset", {}).get("env_ids", [])) != n:
            obs, extras = env.reset(seed=1, options={"block_on": [(e + 1) % 2 for e in range(n)]})

    flat = [c for leg in clearances for c in leg]
    if n == 1:
        log(f"A/B slab clearance, mm: isaac lab {[round(c) for c in flat]}  "
            f"vs isaac sim {REFERENCE_MM[mapping]}")
    else:
        for e in range(n):
            log(f"env {e} slab clearance, mm: {[round(c) for leg in clearances[e::n] for c in leg]}")
        log(f"isaac sim, one env, for reference: {REFERENCE_MM[mapping]}")
    if placed:
        log(f"placement error, mm: {min(placed):.1f}..{max(placed):.1f} over {len(placed)} places")
    if not flat:
        failures.append("the server reported no clearance to the slab")
    elif mapping and min(flat) < base.BOUND_MM:
        failures.append(f"a mapped route came {min(flat):+.0f} mm from the slab "
                        f"(bound {base.BOUND_MM:+.0f})")
    elif not mapping and min(flat) > base.BOUND_MM:
        failures.append(f"unmapped routes cleared the slab by {min(flat):+.0f} mm: "
                        f"it is no longer in the way, so the mapped check proves nothing")

    if not mapping and not failures:
        failures += check_rl_wrapper(env, n)
    return failures


def check_rl_wrapper(env, n):
    """Does Isaac Lab's own RL plumbing take this environment?"""
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except ImportError as exc:
        log(f"rsl_rl wrapper not installed ({exc}); skipped")
        return []
    wrapped = RslRlVecEnvWrapper(env)
    obs = wrapped.get_observations()
    policy = obs["policy"] if hasattr(obs, "keys") else obs
    ok = tuple(policy.shape) == (n, 25) and wrapped.num_actions == 4
    log(f"rsl_rl wrapper: observations {tuple(policy.shape)}, "
        f"{wrapped.num_actions} actions, max episode length {wrapped.max_episode_length}")
    return [] if ok else [f"rsl_rl wrapper sees observations {tuple(policy.shape)}"]


# --- parent: servers and child processes ----------------------------------------


def ports_taken(k):
    taken = []
    for i in range(k):
        with socket.socket() as s:
            if s.connect_ex((HOST, PORT + i)) == 0:
                taken.append(PORT + i)
    return taken


def start_servers(k, mapping, logfile):
    """scripts/planner_servers.py with k servers. (proc, error or None)."""
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "planner_servers.py"),
           "--num", str(k)] + ([] if mapping else ["--no-mapping"])
    out = open(logfile, "w")
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT,
                            start_new_session=True)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            return proc, "exited"
        with open(logfile) as f:
            text = f.read()
        if "[servers] all" in text:
            return proc, None
        time.sleep(1)
    return proc, "did not start listening within 900 s"


def stop_servers(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(40)
        except subprocess.TimeoutExpired:
            proc.kill()


def check(args, mapping, workdir):
    name = f"mapping {'on' if mapping else 'off'}, {args.num_envs} env(s)"
    k = args.num_servers or args.num_envs
    log(f"=== isaac lab, {name}, {k} planner server(s) ===")
    tag = "on" if mapping else "off"
    srv_log = os.path.join(workdir, f"servers_{tag}.log")
    servers, err = start_servers(k, mapping, srv_log)
    try:
        if err:
            log(f"FAIL {name}: planner servers {err}\n{base.tail(srv_log)}")
            return False
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--child",
               "--mapping", tag, "--device", args.device, "--robot", args.robot,
               "--num-envs", str(args.num_envs), "--num-servers", str(args.num_servers)]
        if args.depth_lag is not None:
            cmd += ["--depth-lag", str(args.depth_lag)]
        if args.camera_class:
            cmd += ["--camera-class", args.camera_class]
        if args.replicate_physics:
            cmd += ["--replicate-physics"]
        child = subprocess.run(cmd, cwd=ROOT, timeout=3600, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
        with open(os.path.join(workdir, f"child_{tag}.log"), "w") as f:
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
        stop_servers(servers)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mapping", choices=("on", "off", "both"), default="both")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--robot", default=DEFAULT_ROBOT)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--num-servers", type=int, default=0,
                    help="0: one per env (required with mapping on)")
    ap.add_argument("--camera-class", choices=("camera", "tiled"), default=None)
    ap.add_argument("--replicate-physics", action="store_true")
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

    modes = {"on": [True], "off": [False], "both": [False, True]}[args.mapping]
    if args.num_servers and True in modes and args.num_servers != args.num_envs:
        sys.exit("[check-lab] with mapping on every env needs its own server; "
                 "leave --num-servers at 0 or run --mapping off")
    k = args.num_servers or args.num_envs
    taken = ports_taken(k)
    if taken:
        sys.exit(f"[check-lab] ports {taken} are already in use -- stop the running "
                 f"planner server(s) first (ss -ltnp | grep {PORT})")
    results = []
    if args.both_backends:
        log("=== isaac sim backend (tools/check_pick_place.py) ===")
        old = subprocess.run([sys.executable, "-u", os.path.join(ROOT, "tools", "check_pick_place.py"),
                              "--mapping", args.mapping], cwd=ROOT)
        results.append(old.returncode == 0)
    workdir = tempfile.mkdtemp(prefix="check_pick_place_lab_")
    results += [check(args, m, workdir) for m in modes]
    log(f"logs in {workdir}")
    log("ALL PASSED" if all(results) else "FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
