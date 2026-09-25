"""Score grasp.perception against the truth, on views tools/grasp_capture.py saved.

    python tools/check_grasp_perception.py /tmp/grasp_capture
    python tools/check_grasp_perception.py DIR --depth-noise 0.003     # D435-like noise added

No simulator. For each captured layout: the cube's position and yaw error
(yaw on the 4-fold circle), how much of it the best view saw, and for the
cylinders the centre error of each one found, the ones missed and the ones
reported that are not there. The summary is what the training error model
(grasp.task.TaskCfg: pos_sigma, yaw_sigma_deg, cyl_sigma, miss_p, false_p)
should be set from.

--depth-noise adds Gaussian noise proportional to depth squared, as a
stereo camera's is (sigma = given value at 1 m), so the simulator's perfect
depth is not the only thing measured.
"""

import argparse
import glob
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from grasp import perception as P  # noqa: E402
from grasp import task as T  # noqa: E402


def cube_truth(pose):
    w, x, y, z = pose[3:7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return float(pose[0]), float(pose[1]), (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("--depth-noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    files = sorted(glob.glob(os.path.join(args.dir, "*.npz")))
    pos_err, yaw_err, vis, cyl_err = [], [], [], []
    missed = found_false = n_cyl = unseen = 0
    for f in files:
        d = np.load(f)
        views = []
        for i in range(len(d["depth"])):
            depth = d["depth"][i].copy()
            if args.depth_noise:
                ok = depth > 0
                depth[ok] += rng.normal(0, 1, ok.sum()) * args.depth_noise * depth[ok] ** 2
            views.append(dict(depth=depth, rgb=d["rgb"][i], K=d["K"][i], pose=d["pose"][i]))
        est = P.perceive(views)
        tx, ty, tyaw = cube_truth(d["cube_pose"])
        truth_cyl = [tuple(c) for c in d["cylinders"]]
        n_cyl += len(truth_cyl)
        if est is None:
            unseen += 1
            print(f"[perception] {os.path.basename(f)}: cube NOT seen")
            continue
        e = 1000 * math.hypot(est.cube[0] - tx, est.cube[1] - ty)
        dy = math.degrees((est.cube[2] - tyaw + math.pi / 4) % (math.pi / 2) - math.pi / 4)
        pos_err.append(e)
        yaw_err.append(dy)
        vis.append(est.visibility)
        got = [tuple(c[:2]) for c in est.cylinders if c[2] > 0.5]
        used = set()
        for cx, cy in truth_cyl:
            dists = [math.hypot(cx - gx, cy - gy) if k not in used else 1e9
                     for k, (gx, gy) in enumerate(got)]
            if dists and min(dists) < 0.04:
                k = int(np.argmin(dists))
                used.add(k)
                cyl_err.append(1000 * dists[k])
            else:
                missed += 1
        found_false += len(got) - len(used)
        print(f"[perception] {os.path.basename(f)}: cube {e:4.1f} mm, yaw {dy:+5.2f} deg, "
              f"seen {100 * est.visibility:3.0f}%; cylinders {len(truth_cyl)} there, "
              f"{len(got)} reported")
    n = len(files)
    print(f"[perception] {n} layouts; cube not seen in {unseen}")
    if pos_err:
        print(f"[perception] cube position error: median {np.median(pos_err):.1f} mm, "
              f"95% {np.percentile(pos_err, 95):.1f} mm, max {max(pos_err):.1f} mm "
              f"(sigma per axis ~{np.sqrt(np.mean(np.square(pos_err)) / 2):.1f} mm)")
        print(f"[perception] cube yaw error: median |{np.median(np.abs(yaw_err)):.2f}| deg, "
              f"95% |{np.percentile(np.abs(yaw_err), 95):.2f}| deg (sigma ~{np.std(yaw_err):.2f})")
        print(f"[perception] best-view visibility: min {min(vis):.2f}, median {np.median(vis):.2f}")
    if n_cyl:
        print(f"[perception] cylinders: {n_cyl} there, {missed} missed "
              f"({100 * missed / n_cyl:.1f}%), {found_false} false; centre error median "
              f"{np.median(cyl_err) if cyl_err else float('nan'):.1f} mm, max "
              f"{max(cyl_err) if cyl_err else float('nan'):.1f} mm")
    cfg = T.TaskCfg()
    print(f"[perception] training's error model now: pos_sigma {1000 * cfg.pos_sigma:.1f} mm, "
          f"yaw_sigma {cfg.yaw_sigma_deg:.1f} deg, cyl_sigma {1000 * cfg.cyl_sigma:.1f} mm, "
          f"miss {cfg.miss_p:.2f}, false {cfg.false_p:.2f}")
    ok = unseen == 0 and pos_err and np.percentile(pos_err, 95) < 10 and missed == 0
    print(f"[perception] RESULT {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
