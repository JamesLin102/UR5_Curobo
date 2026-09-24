"""cell_api ops, interpreted one physics tick at a time, for N environments at once.

    results = ProgramRunner(cell).run(env_ids, [ops_for_env_0, ops_for_env_1, ...])

cell_api.run_ops_blocking() plays a program by calling blocking primitives,
and that is the reference: the semantics here are the same step for step. What
blocking calls cannot do is run several environments at once, each at its own
point in its own program, while the physics advances for all of them together.
This does:

  - every tick, each live environment's current op sets its commands, the
    physics steps ONCE for everybody, and each op looks at the result;
  - an op that needs the planner asks when it starts, and every environment
    starting one on the same tick is asked for in one call (PlannerPool),
    so a batched planner gets a batch;
  - an environment whose program has ended holds its last targets while the
    others finish -- run() returns when all of them have.

An op that takes no simulation time (a refused plan, an Idle(0)) finishes
without a tick, as its blocking counterpart returns without a step.
"""

import numpy as np

from cell_api import (GRIP_EXTRA_STEPS, SCAN_BLEND_STEPS, SCAN_STEPS, SETTLE_MAX_STEPS,
                      SETTLE_QUIET_STEPS, SETTLE_TOL, Grip, GripResult, Idle, MoveJ,
                      MoveResult, MoveTo, MoveZ, ProgramResult, Retrace, Scan, fail_reason)
from rig import SIM_DT


class _Run:
    """One op in one environment.

    request()  what it needs from the planner to start: None, or
               ("plan", target pose) | ("plan_joint", joint goal) |
               ("ik", target pose, check the straight move?)
    start()    given the planner's reply; may finish at once
    command()  before each tick: set this tick's targets
    after()    after each tick: look, and maybe finish
    """

    def __init__(self, op, cell, env):
        self.op, self.cell, self.env = op, cell, env
        self.done, self.result = False, None

    def finish(self, result=None):
        self.done, self.result = True, result

    def request(self):
        return None

    def start(self, reply):
        pass

    def command(self):
        pass

    def after(self):
        pass


class _MoveTo(_Run):
    """Plan from where the arm is, then play one waypoint per tick, fusing as it goes."""

    def request(self):
        start = self.cell.plan_end[self.env] if self.op.from_plan_end else None
        return ("plan", list(self.op.pose), start)

    def start(self, reply):
        if not reply.get("ok"):
            self.finish(MoveResult(ok=False, reason=reply.get("goal", reply.get("status"))))
            return
        self.reply, self.traj, self.k, self.t0 = reply, reply["traj"], 0, self.cell.steps

    def command(self):
        self.cell.command_arm(self.env, self.traj[self.k])

    def after(self):
        if self.cell.cams and self.k % self.cell.cfg.map_every == 0:
            self.cell.fuse([self.env])
        self.k += 1
        if self.k == len(self.traj):
            self.cell.goal[self.env] = list(self.op.pose)
            self.cell.plan_end[self.env] = np.asarray(self.traj[-1], dtype=np.float64)
            self.cell.plan_goal[self.env] = list(self.op.pose)
            self.finish(MoveResult(ok=True, solve_ms=self.reply["solve_ms"],
                                   waypoints=len(self.traj),
                                   clearance=self.reply.get("clearance"),
                                   sim_steps=self.cell.steps - self.t0))


class _MoveJ(_MoveTo):
    """Plan to a joint configuration, then play it as MoveTo does."""

    def request(self):
        start = self.cell.plan_end[self.env] if self.op.from_plan_end else None
        return ("plan_joint", list(self.op.q), start)

    def after(self):
        if self.cell.cams and self.k % self.cell.cfg.map_every == 0:
            self.cell.fuse([self.env])
        self.k += 1
        if self.k == len(self.traj):
            # Not a tool goal: a MoveZ after this has nothing to go straight from.
            self.cell.goal[self.env] = None
            self.finish(MoveResult(ok=True, solve_ms=self.reply["solve_ms"],
                                   waypoints=len(self.traj),
                                   clearance=self.reply.get("clearance"),
                                   sim_steps=self.cell.steps - self.t0))


class _Retrace(_Run):
    """Joint-space blend back to where the last planned move ended (cell.plan_end)."""

    def start(self, reply):
        end = self.cell.plan_end[self.env]
        if end is None:
            self.finish(MoveResult(ok=False, reason="nothing planned to go back to"))
            return
        self.q0 = np.asarray(self.cell.q_now(self.env), dtype=np.float64)
        self.q1 = end
        self.i, self.t0 = 0, self.cell.steps

    def command(self):
        blend = (self.i + 1) / self.op.n_steps
        self.cell.command_arm(self.env, self.q0 + (self.q1 - self.q0) * blend)

    def after(self):
        self.i += 1
        if self.i == self.op.n_steps:
            # Back where the planned move put the tool: a MoveZ from here goes
            # straight from that pose.
            self.cell.goal[self.env] = self.cell.plan_goal[self.env]
            self.finish(MoveResult(ok=True, sim_steps=self.cell.steps - self.t0))


class _MoveZ(_Run):
    """Straight up or down from the last goal: IK at the end, interpolate to it."""

    def request(self):
        goal = self.cell.goal[self.env]
        if goal is None:
            return None
        self.target = list(goal)
        self.target[2] += self.op.dz
        return ("ik", self.target, self.op.check if self.op.check == "escape" else bool(self.op.check))

    def start(self, reply):
        if self.cell.goal[self.env] is None:
            self.finish(MoveResult(ok=False, reason="no goal yet: move_to first"))
            return
        joints, why = reply if isinstance(reply, tuple) else (reply, None)
        if joints is None:
            self.cell.log(f"  no IK for z{self.op.dz:+.3f} m ({why or 'no IK'}); skipping",
                          self.env)
            self.finish(MoveResult(ok=False, reason=why or "no IK"))
            return
        reply = joints
        self.q0 = np.asarray(self.cell.q_now(self.env), dtype=np.float64)
        self.q1 = np.asarray(reply, dtype=np.float64)
        n = self.op.n_steps
        # A 12 cm vertical move is a small joint move. If it is not, the IK
        # came back on a different branch from the one the arm is standing in.
        travel = np.abs(self.q1 - self.q0)
        if n > 0:
            self.cell.log(f"  z{self.op.dz:+.3f}: joint travel "
                          f"{' '.join(f'{v:.3f}' for v in travel)} rad"
                          f"   worst {travel.max():.3f} over {n} steps "
                          f"({travel.max() / (n * SIM_DT):.2f} rad/s)", self.env)
        self.i, self.t0 = 0, self.cell.steps
        if n <= 0:
            self._done()

    def command(self):
        blend = (self.i + 1) / self.op.n_steps
        self.cell.command_arm(self.env, self.q0 + (self.q1 - self.q0) * blend)

    def after(self):
        self.i += 1
        if self.i == self.op.n_steps:
            self._done()

    def _done(self):
        self.cell.goal[self.env] = self.target
        self.finish(MoveResult(ok=True, sim_steps=self.cell.steps - self.t0))


class _Grip(_Run):
    """Ramp the stroke open-loop, then wait for the fingers to actually stop."""

    def start(self, reply):
        c = self.cell
        self.stroke = int(abs(c.grip_closed - c.grip_open) / c.grip_speed / SIM_DT) \
            + GRIP_EXTRA_STEPS
        self.end = c.grip_closed if self.op.close else c.grip_open
        self.want = c.grip_mult * self.end
        self.i, self.settling = 0, False

    def command(self):
        if self.settling:
            return
        c = self.cell
        f = (self.i + 1) / self.stroke
        f = f if self.op.close else 1 - f
        c.command_gripper(self.env, c.grip_open + (c.grip_closed - c.grip_open) * f)

    def after(self):
        c = self.cell
        if not self.settling:
            self.i += 1
            if self.i == self.stroke:
                self.settling, self.waited, self.quiet = True, 0, 0
                self.prev = c.grip_driven_q(self.env)
            return
        self.waited += 1
        now = c.grip_driven_q(self.env)
        self.quiet = self.quiet + 1 if float(np.abs(now - self.prev).max()) < SETTLE_TOL else 0
        self.prev = now
        if self.quiet >= SETTLE_QUIET_STEPS:
            self._done(True, now)
        elif self.waited >= SETTLE_MAX_STEPS:
            self._done(False, now)

    def _done(self, settled, now):
        c = self.cell
        c.closed[self.env] = self.op.close
        self.finish(GripResult(settled=settled, steps=self.waited, target=self.end,
                               error=float(np.abs(now - self.want).max()),
                               spread=c.grip_spread(self.env)))


class _Idle(_Run):
    def start(self, reply):
        self.i = 0
        if self.op.n <= 0:
            self.finish()

    def after(self):
        self.i += 1
        if self.i >= self.op.n:
            self.finish()


class _Scan(_Run):
    """Sweep the scene's scan poses, fusing as it goes, so the map starts with content."""

    def start(self, reply):
        self.poses = [np.asarray(p, dtype=np.float64) for p in self.cell.spec.scan_poses]
        self.p, self.s, self.fused = 0, 0, 0
        self.cell.log("scanning the cell before planning...", self.env)
        if not self.poses:
            self._done()

    def command(self):
        blend = min(1.0, (self.s + 1) / float(SCAN_BLEND_STEPS))
        q = np.asarray(self.cell.q_now(self.env), dtype=np.float64)
        self.cell.command_arm(self.env, (1 - blend) * q + blend * self.poses[self.p])

    def after(self):
        if self.s % self.cell.cfg.map_every == 0:
            self.fused += self.cell.fuse([self.env])
        self.s += 1
        if self.s == SCAN_STEPS:
            self.p, self.s = self.p + 1, 0
            if self.p == len(self.poses):
                self._done()

    def _done(self):
        self.fused += self.cell.fuse([self.env])
        self.cell.log(f"scan done: {self.fused} frames sent to the mapper", self.env)
        self.finish(self.fused)


RUNS = {MoveTo: _MoveTo, MoveJ: _MoveJ, MoveZ: _MoveZ, Retrace: _Retrace, Grip: _Grip,
        Idle: _Idle, Scan: _Scan}


class _Program:
    """One environment's program: a queue of ops and what has come of them."""

    def __init__(self, env, ops):
        self.env = env
        self.queue = [(op, True) for op in ops]    # (op, record its result?)
        self.run = None
        self.done = not self.queue
        self.result = ProgramResult(ok=True)

    def next_op(self):
        return self.queue.pop(0) if self.queue else (None, False)

    def finish_op(self, op, record, result):
        if record:
            self.result.results.append((op, result))
        if isinstance(op, (MoveTo, MoveZ, MoveJ, Retrace)) and not result.ok:
            if self.result.ok:
                self.result.ok, self.result.why = False, fail_reason(op, result)
            if op.fatal:
                self.queue = []
            if op.fail_idle:
                # Held, but not reported: the blocking version idles here
                # without recording it either.
                self.queue.insert(0, (Idle(op.fail_idle), False))
        self.done = not self.queue


class ProgramRunner:
    def __init__(self, cell):
        self.cell = cell

    def run(self, env_ids, programs):
        """Run programs[k] in env_ids[k], all together. One ProgramResult each."""
        progs = [_Program(int(e), ops) for e, ops in zip(env_ids, programs)]
        while True:
            self._start_ops(progs)
            live = [p for p in progs if not p.done]
            if not live:
                break
            if not self.cell.running:
                for p in live:
                    self._abort(p)
                break
            for p in live:
                p.run.command()
            self.cell.tick()
            for p in live:
                p.run.after()
                if p.run.done:
                    p.finish_op(p.run.op, p.record, p.run.result)
                    p.run = None
        return [p.result for p in progs]

    def _start_ops(self, progs):
        """Give every idle program its next op; ask the planner for all of them at once.

        Loops because a start can finish immediately (a refused plan), and
        then that program needs its next op before this tick, too.
        """
        while True:
            starting = []
            for p in progs:
                if p.done or p.run is not None:
                    continue
                op, record = p.next_op()
                p.run, p.record = RUNS[type(op)](op, self.cell, p.env), record
                starting.append(p)
            if not starting:
                return
            replies = self._ask(starting)
            for p in starting:
                p.run.start(replies.get(p.env))
                if p.run.done:
                    p.finish_op(p.run.op, p.record, p.run.result)
                    p.run = None

    def _ask(self, starting):
        """Batch every planner request made on this tick, by kind."""
        replies, asks = {}, {"plan": [], "plan_joint": [], "ik": []}
        for p in starting:
            req = p.run.request()
            if req is not None:
                asks[req[0]].append((p.env, req[1:]))
        pool = self.cell.pool
        for kind in asks:
            if not asks[kind]:
                continue
            envs = [e for e, _ in asks[kind]]
            # A request may name where to plan from (MoveJ.from_plan_end).
            q = np.stack([a[1] if kind in ("plan", "plan_joint") and len(a) > 1
                          and a[1] is not None else self.cell.q_now(e)
                          for e, a in asks[kind]])
            targets = np.asarray([a[0] for _, a in asks[kind]], dtype=np.float64)
            if kind == "plan":
                got = pool.plan(envs, q, targets)
            elif kind == "plan_joint":
                got = pool.plan_joint(envs, q, targets)
            else:
                got = pool.ik(envs, q, targets, [a[1] for _, a in asks[kind]])
            for e, r in zip(envs, got):
                replies[e] = r
        return replies

    def _abort(self, p):
        """The app is closing: end the op as its blocking version would."""
        op = p.run.op
        result = MoveResult(ok=False, reason="simulation closed") \
            if isinstance(op, (MoveTo, MoveZ, MoveJ, Retrace)) else p.run.result
        p.finish_op(op, p.record, result)
        p.run, p.queue, p.done = None, [], True
