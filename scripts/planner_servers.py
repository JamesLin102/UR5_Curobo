"""N planner servers on consecutive ports, one per environment of a multi-env run.

One planner_server.py holds one map. A cell's map must hold only what that
cell's cameras saw, so an Isaac Lab run with N environments and mapping on
talks to N servers: environment i to port PORT + i. With mapping off the
planner's world is the static scene, the same for every environment, so any
number of environments can share fewer servers (lab CellCfg.num_servers).

    python scripts/planner_servers.py --num 4                 # ports 5599..5602
    python scripts/planner_servers.py --num 4 --no-mapping

Every other flag is passed to each planner_server.py unchanged. Each server's
output is prefixed with its environment. Once all of them are listening this
prints "[servers] all N listening". Ctrl-C stops them all.

Measured on the development machine: a server takes about 2 GB of RAM and
0.55 GB of GPU memory, so RAM (31 GB, shared with Isaac Lab) is what limits N.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import PORT  # noqa: E402

SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planner_server.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num", type=int, required=True, help="how many servers")
    ap.add_argument("--port", type=int, default=PORT, help="the first one's port")
    args, passthrough = ap.parse_known_args()

    procs, ready = [], []
    lock = threading.Lock()

    def relay(i, proc):
        for line in proc.stdout:
            if "listening on" in line:
                with lock:
                    ready.append(i)
            print(f"[env {i}] {line}", end="", flush=True)

    for i in range(args.num):
        cmd = [sys.executable, "-u", SERVER, "--port", str(args.port + i)] + passthrough
        # Its own process group, so Ctrl-C reaches it once, from here, rather
        # than from the terminal and again from here.
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, start_new_session=True)
        procs.append(p)
        threading.Thread(target=relay, args=(i, p), daemon=True).start()

    try:
        announced = False
        while True:
            dead = [i for i, p in enumerate(procs) if p.poll() is not None]
            if dead:
                print(f"[servers] server for env {dead[0]} exited "
                      f"({procs[dead[0]].returncode}); stopping the rest", flush=True)
                break
            if not announced and len(ready) == args.num:
                print(f"[servers] all {args.num} listening on ports "
                      f"{args.port}..{args.port + args.num - 1}  (Ctrl-C to stop)", flush=True)
                announced = True
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        deadline = time.time() + 20
        for p in procs:
            try:
                p.wait(max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
        print("[servers] stopped", flush=True)


if __name__ == "__main__":
    main()
