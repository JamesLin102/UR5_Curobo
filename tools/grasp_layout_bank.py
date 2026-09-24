"""Build the grasp task's layout banks: random layouts, kept only if one can be gripped.

    python tools/grasp_layout_bank.py --name train --count 3000 --workers 4
    python tools/grasp_layout_bank.py --name eval  --count 300  --workers 4 --seed 1000

cuRobo only, no simulator. Each layout is the cube anywhere in CUBE_XY at any
yaw and 0-4 cylinders in CYL_XY at least MIN_GAP from it and from each other,
each put near the cube with chance NEAR_P (tools/grasp_reach.sample_layout). Both grips square to the cube's faces are
tried as tools/grasp_reach.try_grip tries them -- planned with the planner
told the cylinders (training mode), straight moves checked against the real
cylinders and the table -- and a layout with neither is dropped. What is kept,
per layout, is what the task's curriculum and error model need
(grasp.task.Bank): which grips work and how close they come, the nearest
cylinder, and how much of the cube each scan view sees.

Checking at reset instead would cost ~0.8 s per grip per environment, every
episode; this pays it once (docs/grasp_rl_plan.md §2). Writes
scripts/grasp/layouts/<name>.npz. Different seeds for train and eval keep the
two banks apart.
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from grasp import scene as G  # noqa: E402
from grasp import task as T  # noqa: E402


# The chance each cylinder is put near the cube (15-120 mm from it) rather
# than anywhere: layouts worth learning from are crowded ones, and uniform
# placement leaves most cubes in the open (tools/grasp_reach.py: 112 of 200
# layouts had nothing within 100 mm).
NEAR_P = 0.6


def say(msg):
    print(f"[bank] {msg}", flush=True)


def work(count, seed, out):
    """Keep `count` grippable layouts; write them to `out`."""
    import grasp_reach as R

    arm = R.Arm()
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ("cube", "cyl", "feasible", "clear", "gap", "vis")}
    tried, t0 = 0, time.time()
    while len(rows["cube"]) < count:
        tried += 1
        n_cyl = int(rng.integers(0, G.MAX_CYLINDERS + 1))
        (x, y, yaw), cyl, gap, _ = R.sample_layout(rng, n_cyl, near_p=NEAR_P)
        arm.set_cylinders(cyl)
        feas, clear = [], []
        for face in (0.0, math.pi / 2):
            r, c = R.try_grip(arm, x, y, float(T.wrap_yaw(yaw + face)))
            feas.append(r is None)
            clear.append(c[0] if (r is None and c is not None) else np.nan)
        if not any(feas):
            continue
        cyl_arr = np.zeros((G.MAX_CYLINDERS, 3))
        for i, (cx, cy) in enumerate(cyl):
            cyl_arr[i] = (cx, cy, 1.0)
        rows["cube"].append((x, y, yaw))
        rows["cyl"].append(cyl_arr)
        rows["feasible"].append(feas)
        rows["clear"].append([min(v, 1.0) for v in clear])
        rows["gap"].append(min(gap, 1.0))
        rows["vis"].append(T.cube_visibility((x, y, yaw), cyl, T.SCAN_EYES))
        k = len(rows["cube"])
        if k % 50 == 0:
            say(f"worker {seed}: {k}/{count} kept of {tried} tried, "
                f"{(time.time() - t0) / k:.1f} s per kept layout")
    np.savez(out, **{k: np.asarray(v, dtype=np.float64 if k != "feasible" else bool)
                     for k, v in rows.items()}, tried=tried)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="train, eval, or a file name")
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)   # count seed out
    args = ap.parse_args()
    if args.worker:
        work(int(args.worker[0]), int(args.worker[1]), args.worker[2])
        return

    out_dir = os.path.join(ROOT, "scripts", "grasp", "layouts")
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, f"{args.name}.npz")
    tmp = tempfile.mkdtemp(prefix="grasp_bank_")
    per = [args.count // args.workers + (i < args.count % args.workers)
           for i in range(args.workers)]
    procs = []
    for i, n in enumerate(per):
        part = os.path.join(tmp, f"part{i}.npz")
        log = open(os.path.join(tmp, f"part{i}.log"), "w")
        procs.append((part, subprocess.Popen(
            [sys.executable, "-u", __file__, "--name", args.name, "--count", "0",
             "--worker", str(n), str(args.seed * 100 + i), part],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)))
    say(f"{args.workers} workers, {args.count} layouts, logs in {tmp}")
    for part, p in procs:
        if p.wait() != 0:
            raise SystemExit(f"a worker failed; see {tmp}")
    parts = [np.load(part) for part, _ in procs]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files if k != "tried"}
    tried = int(sum(int(p["tried"]) for p in parts))
    np.savez(target, **merged)
    n = len(merged["cube"])
    nc = merged["cyl"][..., 2].sum(axis=1).astype(int)
    both = merged["feasible"].all(axis=1).sum()
    say(f"wrote {target}: {n} layouts kept of {tried} tried "
        f"({100 * n / tried:.0f}%); both grips work in {both}, one in {n - both}")
    say("  cylinders:  " + "  ".join(f"{k}: {int((nc == k).sum())}" for k in range(G.MAX_CYLINDERS + 1)))
    gap = merged["gap"][nc > 0]
    say(f"  nearest cylinder, where there is one: {1000 * np.percentile(gap, 10):.0f} mm (10%), "
        f"{1000 * np.median(gap):.0f} mm (median)")
    vis = merged["vis"].max(axis=1)
    say(f"  cube seen by the best view below 60%: {int((vis < 0.6).sum())} layouts")


if __name__ == "__main__":
    main()
