"""End-to-end check: does pick_place still work, and does the map still matter?

One command, no second terminal and no display: this starts its own planner
server, runs PickPlaceEnv headless with an oracle that grips at the true
pedestals, and stops the server again. Each mode gets a fresh server and a
fresh Isaac Sim process, because Isaac Sim can only be launched once per
process.

    python tools/check_pick_place.py                  # both modes, ~5 min
    python tools/check_pick_place.py --mapping off    # the fast one, ~2 min

Checked, per mode, over two seeds (the block starts on each pedestal once):

    every leg plans and executes, the pick is holding the block, and the
    place delivers it to the goal pedestal (within 2 cm / 1 cm)

    mapping on   the planned routes stay clear of the slab the planner is
                 never told about: every leg >= -10 mm
    mapping off  the slab still sits on the unplanned route: some leg <= -10 mm.
                 If this fails, the mapped check above proves nothing.

The -10 mm bound sits between what the two modes measured when this was
written: -16..-26 mm unmapped, -4..+10 mm mapped (see the README).

Exit status 0 means every check passed. The port the server uses (rig.PORT)
must be free.
"""

import argparse
import os
import re
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from rig import HOST, PORT  # noqa: E402

BOUND_MM = -10.0


def log(msg):
    print(f"[check] {msg}", flush=True)


# --- child: one mode, inside Isaac Sim ------------------------------------------


def run_mode(mapping):
    """Run the oracle episodes. Returns a list of failure strings."""
    import numpy as np
    from pick_place.isaacsim_env import PickPlaceEnv, TaskCfg
    from sim.sim_env import EnvCfg

    env = PickPlaceEnv(EnvCfg(mapping=mapping, verbose=False),
                       TaskCfg(max_legs=4), headless=True)
    targets = env.sim.scene.targets
    grasp_z = targets[0][2] - env.descend      # the height the demo grips at
    failures, clearances = [], []
    for seed in (0, 1):
        t0 = time.time()
        _, info = env.reset(seed=seed, options={"block_on": seed})
        s, g = info["start"], info["goal"]
        for leg, idx in (("pick", s), ("place", g)):
            action = np.array([targets[idx][0], targets[idx][1], grasp_z, 0.0], np.float32)
            _, _, term, trunc, inf = env.step(action)
            gap = [float(v) for v in re.findall(r"([-+]\d+(?:\.\d+)?) mm", inf.get("clearance") or "")]
            clearances += gap
            log(f"seed {seed} {leg}@{idx}: failed={inf.get('failed')} "
                f"holding={inf['holding']} success={inf['success']} "
                f"clearance={inf.get('clearance')} "
                f"block_to_goal={inf['block_to_goal'] * 1000:.1f} mm")
            if "failed" in inf:
                failures.append(f"seed {seed} {leg}: {inf['failed']}")
            if leg == "pick" and not inf["holding"]:
                failures.append(f"seed {seed}: pick is not holding the block")
            if leg == "place" and not inf["success"]:
                failures.append(f"seed {seed}: block not delivered "
                                f"({inf['block_to_goal'] * 1000:.0f} mm from the goal)")
            if term or trunc:
                break
        log(f"seed {seed}: {time.time() - t0:.1f} s")

    if not clearances:
        failures.append("the server reported no clearance to the slab")
    elif mapping and min(clearances) < BOUND_MM:
        failures.append(f"mapped route came {min(clearances):+.0f} mm from the slab "
                        f"(bound {BOUND_MM:+.0f})")
    elif not mapping and min(clearances) > BOUND_MM:
        failures.append(f"unmapped route cleared the slab by {min(clearances):+.0f} mm: "
                        f"it is no longer in the way, so the mapped check proves nothing")
    return failures


# --- parent: servers and child processes ----------------------------------------


def port_taken():
    with socket.socket() as s:
        return s.connect_ex((HOST, PORT)) == 0


def start_server(mapping, logfile):
    cmd = [sys.executable, "-u", os.path.join(ROOT, "scripts", "planner_server.py")]
    if not mapping:
        cmd.append("--no-mapping")
    out = open(logfile, "w")
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT)
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            return proc, "exited"
        with open(logfile) as f:
            if "listening on" in f.read():
                return proc, None
        time.sleep(1)
    return proc, "did not start listening within 600 s"


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            proc.kill()


def tail(path, n=25):
    with open(path) as f:
        return "".join(f.readlines()[-n:])


def check(mapping, workdir):
    name = "mapping on" if mapping else "mapping off"
    log(f"=== {name} ===")
    srv_log = os.path.join(workdir, f"server_{'on' if mapping else 'off'}.log")
    server, err = start_server(mapping, srv_log)
    try:
        if err:
            log(f"FAIL {name}: planner server {err}\n{tail(srv_log)}")
            return False
        log("planner server listening; launching Isaac Sim headless "
            "(the first launch on a machine can take many minutes)")
        child = subprocess.run(
            [sys.executable, "-u", os.path.abspath(__file__), "--child",
             "--mapping", "on" if mapping else "off"],
            cwd=ROOT, timeout=1800, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        for line in child.stdout.splitlines():
            if line.startswith("[check]"):
                print(line, flush=True)
        if child.returncode != 0:
            if "[check] RESULT" not in child.stdout:
                log("child output tail:\n" + "\n".join(child.stdout.splitlines()[-25:]))
            log(f"FAIL {name}")
            return False
        log(f"PASS {name}")
        return True
    except subprocess.TimeoutExpired:
        log(f"FAIL {name}: timed out")
        return False
    finally:
        stop(server)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mapping", choices=("on", "off", "both"), default="both")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        failures = run_mode(args.mapping == "on")
        for f in failures:
            log(f"  - {f}")
        log(f"RESULT {'FAIL' if failures else 'PASS'}")
        sys.stdout.flush()
        # Not env.close(): a headless SimulationApp often fails to exit
        # cleanly and leaves the process holding the GPU. The result is out.
        os._exit(1 if failures else 0)

    if port_taken():
        sys.exit(f"[check] {HOST}:{PORT} is already in use -- stop the running "
                 f"planner server first (ss -ltnp | grep {PORT})")
    modes = {"on": [True], "off": [False], "both": [False, True]}[args.mapping]
    workdir = tempfile.mkdtemp(prefix="check_pick_place_")
    results = [check(m, workdir) for m in modes]
    log(f"logs in {workdir}")
    log("ALL PASSED" if all(results) else "FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
