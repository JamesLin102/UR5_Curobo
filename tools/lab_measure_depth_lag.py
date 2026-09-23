"""How many sim steps does an Isaac Lab depth image trail the physics by?

    python tools/lab_measure_depth_lag.py --headless

The mapper pairs every depth frame with the joint state it was rendered at,
and the robot's self-mask is only right if that pairing is. On the Isaac Sim
side the depth annotator trailed the physics by 2 steps (EnvCfg.depth_lag).
Isaac Lab renders synchronously, so that number cannot be assumed here: this
measures it, with no planner.

Two events, each at a known step, each watched through every camera:

  payload   the block teleports to the other target (a rigid body)
  arm       the first arm joint jumps by --jump rad (the articulation, which
            is what the self-mask is aligned against)

The lag is the number of steps between the step that moved something and the
first depth frame that shows it. CellCfg.depth_lag should be the largest one
measured; pass it with --depth-lag to the clients if it differs.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lab import app as lab_app  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
lab_app.add_args(ap)
ap.add_argument("--jump", type=float, default=0.3, help="arm joint step, rad")
ap.add_argument("--watch", type=int, default=6, help="steps to watch after each event")
ARGS = ap.parse_args()
ARGS.no_mapping = False
ARGS.planner = "none"
APP = lab_app.launch(ARGS)

import numpy as np  # noqa: E402
import torch  # noqa: E402


def say(msg):
    print(f"[lag] {msg}", flush=True)


def frames(cell):
    return {n: np.nan_to_num(c.data.output["distance_to_image_plane"][0, :, :, 0]
                             .cpu().numpy(), posinf=0.0).copy()
            for n, c in cell.cams.items()}


def watch(cell, event, apply):
    """Settle, apply the event before the next tick, and see when each camera notices."""
    cell.idle([0], 30)
    before = frames(cell)
    apply()
    first = {}
    for k in range(ARGS.watch):
        cell.tick()
        now = frames(cell)
        for name in now:
            changed = int((np.abs(now[name] - before[name]) > 0.01).sum())
            if name not in first and changed > 50:
                first[name] = k
        say(f"  {event} +{k + 1} step(s): " + "  ".join(
            f"{n} {int((np.abs(now[n] - before[n]) > 0.01).sum()):6d} px changed"
            for n in now))
    for name in cell.cams:
        if name in first:
            say(f"{event}: {name} shows it in the frame read after step "
                f"{first[name] + 1} -> lag {first[name]}")
        else:
            say(f"{event}: {name} never showed it within {ARGS.watch} steps")
    return first


def main():
    from lab.tasks.base import CellEnv, CellEnvCfg

    cfg = CellEnvCfg()
    cfg.sim.device = ARGS.device
    lab_app.apply_cell_args(cfg.cell, ARGS)
    env = CellEnv(cfg)
    cell = env.cell
    if not cell.cams:
        raise SystemExit("the scene has no cameras to measure")
    lags = []

    if cell.payload and len(cell.spec.targets) >= 2:
        name, obj = next(iter(cell.payload.items()))
        t = cell.spec.targets[1]
        z = next(b for b in cell.spec.payload if b[0] == name)[2][2]

        def move_payload():
            pose = torch.tensor([[t[0], t[1], z, 1.0, 0.0, 0.0, 0.0]], device=cell.device)
            pose[0, :3] += torch.tensor(cell.origins[0], device=cell.device, dtype=torch.float32)
            obj.write_root_pose_to_sim(pose, env_ids=cell._ids([0]))

        lags += list(watch(cell, "payload", move_payload).values())

    def jump_arm():
        q = cell.q_now(0)
        q[0] += ARGS.jump
        full = cell.robot.data.joint_pos[:1].clone()
        full[0, cell._arm_ids] = torch.tensor(q, dtype=full.dtype, device=cell.device)
        cell.robot.write_joint_state_to_sim(full, torch.zeros_like(full), env_ids=cell._ids([0]))
        cell.command_arm(0, q)

    lags += list(watch(cell, "arm", jump_arm).values())
    if lags:
        say(f"RESULT depth_lag = {max(lags)}  (per event and camera: {lags}; "
            f"CellCfg.depth_lag is {cfg.cell.depth_lag})")
    else:
        say("RESULT no camera saw either event")
    sys.stdout.flush()
    os._exit(0 if lags else 1)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # A Kit app often hangs on the way out after an exception, holding the
        # GPU. Report and leave.
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
