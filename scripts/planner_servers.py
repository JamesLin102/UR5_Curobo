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

From another program -- training, evaluation, the checks -- launch() runs
this in the background and returns once every server listens:

    with planner_servers.launch(4, "--scene", "grasp", "--no-mapping"):
        ...                                   # or servers = launch(...); servers.stop()

Measured on the development machine: a server takes about 2 GB of RAM and
0.55 GB of GPU memory, so RAM (31 GB, shared with Isaac Lab) is what limits N.
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import HOST, PORT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "planner_server.py")


class ServersFailed(SystemExit):
    """The servers exited or never listened. A SystemExit, so a script just ends with it."""


class Servers:
    """What launch() started: one process group, this script and its servers."""

    def __init__(self, proc, logfile):
        self.proc, self.logfile = proc, logfile

    def tail(self, n=25):
        with open(self.logfile) as f:
            return "".join(f.readlines()[-n:])

    def stop(self, timeout=40):
        """Ctrl-C to the group, which stops every server; then SIGKILL if it hangs."""
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


def ports_taken(num, port=PORT):
    """The ports of port..port+num-1 something already listens on."""
    taken = []
    for p in range(port, port + num):
        with socket.socket() as s:
            if s.connect_ex((HOST, p)) == 0:
                taken.append(p)
    return taken


def launch(num, *flags, port=PORT, logfile=None, timeout=900, log=None):
    """Start `num` servers on port.. in the background; return a Servers once all listen.

    flags go to every planner_server.py ("--scene", "grasp", "--no-mapping",
    "--batch", "64", ...). Their own session, so a terminal's Ctrl-C reaches the
    caller only, which stops them. Raises ServersFailed, having stopped them, if
    one exits or they are not all listening within `timeout` s; the log says why.
    """
    logfile = logfile or os.path.join(tempfile.gettempdir(), f"planner_servers_{os.getpid()}.log")
    proc = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), "--num", str(num),
                             "--port", str(port), *flags],
                            stdout=open(logfile, "w"), stderr=subprocess.STDOUT,
                            cwd=os.path.dirname(HERE), start_new_session=True)
    servers = Servers(proc, logfile)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise ServersFailed(f"planner servers exited; see {logfile}:\n{servers.tail()}")
        with open(logfile) as f:
            if "[servers] all" in f.read():
                if log is not None:
                    log(f"{num} planner server(s) on port {port}.. ({' '.join(flags)}); log {logfile}")
                return servers
        time.sleep(1)
    servers.stop()
    raise ServersFailed(f"planner servers did not listen within {timeout} s; see {logfile}")


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
