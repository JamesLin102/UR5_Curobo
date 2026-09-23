"""Do the shared task pieces reproduce the Isaac Sim environment exactly?

No simulator, no planner: a fake cell records every primitive it is asked for.

    python tools/test_leg_ops.py

Checks, against the code the Isaac Sim backend actually runs:

  leg_ops       run_ops_blocking(cell, leg_ops(...)) makes the same calls, in
                the same order, as PickPlaceEnv._leg -- on success and on every
                failure path -- and reports the same info and reason.
  Scorer        rewards, dones and observation vectors match the scoring
                PickPlaceEnv.step did before it moved into pick_place_task,
                re-implemented below from that version.
  CellLike      SimEnv provides every member the protocol names.

Exit status 0 means every check passed.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scenes  # noqa: E402
from cell_api import CellLike, GripResult, MoveResult, Observation, run_ops_blocking  # noqa: E402
from pick_place_env import PickPlaceEnv  # noqa: E402
from pick_place_task import Scorer, TaskCfg, leg_info, leg_ops  # noqa: E402

FAILED = []


def check(ok, what):
    print(f"[test] {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        FAILED.append(what)


class FakeCell:
    """Records primitives. `fail` names which ones return ok=False."""

    def __init__(self, fail=()):
        self.calls, self.fail, self._z = [], set(fail), 0

    def move_to(self, pose):
        self.calls.append(("move_to", tuple(round(v, 6) for v in pose)))
        if "move_to" in self.fail:
            return MoveResult(ok=False, reason="goal in collision")
        return MoveResult(ok=True, solve_ms=12.5, waypoints=90, clearance="drag_me +4 mm")

    def move_tool_z(self, dz, n_steps=70):
        self._z += 1
        self.calls.append(("move_tool_z", round(dz, 6), n_steps))
        which = "down" if dz < 0 else "up"
        return MoveResult(ok=f"z_{which}" not in self.fail)

    def grip(self, close):
        self.calls.append(("grip", close))
        return GripResult(settled=True, steps=40, target=0.8 if close else 0.0, error=0.0)

    def idle(self, n):
        self.calls.append(("idle", n))

    def scan(self):
        self.calls.append(("scan",))
        return 0


def test_leg_ops():
    scene = scenes.load(scenes.DEFAULT)
    descend, lift = scene.pick["descend_m"], scene.pick["lift_m"]
    pose = [0.45, -0.40, 0.14, 0.0, 1.0, 0.0, 0.0]
    for fail in ((), ("move_to",), ("z_down",), ("z_up",)):
        for close, open_first in ((True, False), (True, True), (False, False)):
            # The reference: what PickPlaceEnv.step does -- open first if
            # asked, then _leg -- run unbound against the fake.
            ref = FakeCell(fail)
            if open_first:
                ref.grip(close=False)
            me = SimpleNamespace(sim=ref, descend=descend, lift=lift)
            info_ref = {}
            ok_ref, why_ref = PickPlaceEnv._leg(me, pose, close=close, info=info_ref)

            new = FakeCell(fail)
            res = run_ops_blocking(new, leg_ops(pose, close, open_first, descend, lift))
            info_new = leg_info(res)

            case = f"fail={fail or 'none'} close={close} open_first={open_first}"
            check(new.calls == ref.calls, f"leg_ops calls, {case}")
            check((res.ok, res.why) == (ok_ref, why_ref),
                  f"leg_ops result {res.ok, res.why} vs {ok_ref, why_ref}, {case}")
            check(info_new == info_ref, f"leg_ops info, {case}")


def old_scoring(task, scene, start, observations, oks, running=True):
    """PickPlaceEnv's scoring as it was before pick_place_task existed."""
    block = scene.payload[0][0]
    rest_z = float(scene.payload[0][2][2])
    rest = [np.array([t[0], t[1], rest_z], dtype=np.float32) for t in scene.targets]
    goal = 1 - start

    def to_goal(o):
        return float(np.linalg.norm(o.objects[block][:3] - rest[goal]))

    def vec(o):
        return np.concatenate([
            o.q, [o.gripper], o.tool_pose, o.objects[block],
            [1.0 if o.holding else 0.0], rest[goal],
        ]).astype(np.float32)

    out = []
    dist, legs = to_goal(observations[0]), 0
    out.append(vec(observations[0]))
    for o, ok in zip(observations[1:], oks):
        reward = task.r_step
        if not ok:
            reward += task.r_failed_motion
        d = to_goal(o)
        reward += task.r_progress * (dist - d)
        dist = d
        legs += 1
        p, g = o.objects[block][:3], rest[goal]
        success = (not o.holding
                   and float(np.linalg.norm(p[:2] - g[:2])) < task.success_xy
                   and abs(float(p[2] - g[2])) < task.success_z)
        dropped = (not o.holding) and float(o.objects[block][2]) < task.dropped_z
        if success:
            reward += task.r_success
        if dropped:
            reward += task.r_dropped
        terminated = success or dropped or not running
        truncated = (not terminated) and legs >= task.max_legs
        out.append((vec(o), reward, terminated, truncated, success, dropped, d, legs))
    return out


def test_scorer():
    scene = scenes.load(scenes.DEFAULT)
    task = TaskCfg(max_legs=3)
    rng = np.random.default_rng(0)
    block = scene.payload[0][0]
    rest_z = float(scene.payload[0][2][2])

    def obs(xyz, holding):
        return Observation(
            q=rng.normal(size=6).astype(np.float32), gripper=float(rng.random()),
            tool_pose=rng.normal(size=7).astype(np.float32),
            objects={block: np.array([*xyz, 1, 0, 0, 0], dtype=np.float32)},
            holding=holding, sim_time=0.0)

    for start in (0, 1):
        g = scene.targets[1 - start]
        s0 = scene.targets[start]
        seq = [
            obs([s0[0], s0[1], rest_z], False),                     # reset
            obs([s0[0] + 0.01, s0[1], rest_z + 0.15], True),        # picked
            obs([g[0] + 0.004, g[1] - 0.003, rest_z + 0.002], False),  # delivered
        ]
        oks = [True, True]
        drop = [seq[0], obs([0.4, 0.0, 0.03], False)]               # fell on the table
        stall = [seq[0]] + [obs([s0[0], s0[1], rest_z], False)] * 3  # nothing moves
        for name, observations, ok_list in (
                ("deliver", seq, oks), ("drop", drop, [False]),
                ("truncate", stall, [False, True, False])):
            ref = old_scoring(task, scene, start, observations, ok_list)
            sc = Scorer(scene, task)
            sc.begin_one(start, observations[0])
            same = np.array_equal(sc.vec_one(observations[0]), ref[0])
            for o, ok, r in zip(observations[1:], ok_list, ref[1:]):
                s = sc.score_one(o, failed=not ok)
                same &= np.array_equal(sc.vec_one(o), r[0])
                same &= abs(s["reward"] - r[1]) < 1e-6
                same &= (s["terminated"], s["truncated"], s["success"], s["dropped"]) == r[2:6]
                same &= abs(s["block_to_goal"] - r[6]) < 1e-6 and s["legs"] == r[7]
            check(bool(same), f"Scorer matches the old scoring, start={start} {name}")


def test_protocol():
    import sim_env
    members = [n for n in dir(CellLike) if not n.startswith("_")] + \
        list(CellLike.__annotations__)
    # Attributes set in __init__ cannot be seen on the class; the rest can.
    instance_attrs = {"scene", "payload", "bodies", "cams"}
    missing = [n for n in members
               if n not in instance_attrs and not hasattr(sim_env.SimEnv, n)]
    check(not missing, f"SimEnv provides CellLike (missing: {missing or 'none'})")


def main():
    test_leg_ops()
    test_scorer()
    test_protocol()
    print(f"[test] {'ALL PASSED' if not FAILED else f'{len(FAILED)} FAILED'}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
