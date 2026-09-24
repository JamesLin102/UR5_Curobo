"""The grasp task on Isaac Lab: Isaac-Grasp-{Robot}-v0.

One env step is one grasp ATTEMPT in every environment, in lockstep
(docs/grasp_rl_plan.md §6):

    action  (num_envs, 4) in [-1, 1]: dx, dy, dz, tool yaw -- relative to where
            perception puts the cube, world-aligned (grasp.task.action_to_grip)
    obs     {"policy": (num_envs, 29) what the real arm has,
             "critic": (num_envs, 45) that plus the truth}   grasp.task
    extras  {"attempt": per-env dict: outcome, success, why, contact, ...}

An attempt is planned to above the grip, lowered (the straight move checked
by the planner), closed, lifted (checked), held. Lifted and still held by the
gripper's own reading ends the episode; otherwise the arm opens, goes back to
HOME -- straight up first if it is low, then planned -- and perception looks
again, all inside the same step, so the next observation is from HOME.

Training mode (the default: cell.mapping False) tells each environment's
planner where that environment's cylinders are, and the policy sees a
synthesised perception estimate (grasp.task.synth_estimate). Evaluation mode
(cfg.cell.mapping = True, one mapping planner server per env) tells the
planner nothing: at reset the wrist camera scans -- the map is built from
it, and at each view the images go to grasp.perception, whose estimate is
what the policy sees -- and the straight moves are checked against the map,
less a box round the estimated cube.

Layouts come from the bank (grasp.task.Bank). reset(options=...) takes
"layouts": [bank row per env] to choose them.
"""

from dataclasses import fields

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils import configclass

from cell_api import Grip, Idle, MoveJ, MoveTo, MoveZ, ResetOptions, Retrace, Scan
from lab.tasks.base import CellEnv, CellEnvCfg
from legs import leg_ops, tool_pose

from . import perception as P
from . import scene as G
from . import task as T


def _mirror(dc):
    ns = {"__annotations__": {f.name: f.type for f in fields(dc)}}
    ns.update({f.name: f.default for f in fields(dc)})
    return configclass(type(f"Lab{dc.__name__}", (), ns))


LabTaskCfg = _mirror(T.TaskCfg)


@configclass
class GraspEnvCfg(CellEnvCfg):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(T.ACT_DIM,), dtype=np.float32)
    observation_space = T.OBS_DIM
    state_space = T.STATE_DIM
    task: LabTaskCfg = LabTaskCfg()

    def __post_init__(self):
        self.cell.scene = "grasp"
        self.cell.mapping = False          # training mode; set True for evaluation
        self.cell.markers = "off"          # overlays show up in colour images
        self.cell.verbose = False


class GraspEnv(CellEnv):
    cfg: GraspEnvCfg

    def __init__(self, cfg: GraspEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.task = T.TaskCfg(**{f.name: getattr(cfg.task, f.name) for f in fields(T.TaskCfg)})
        self.bank = T.Bank(T.bank_path(self.task.bank))
        self.rng = np.random.default_rng(cfg.seed)
        n = self.num_envs
        self.training_mode = not cfg.cell.mapping
        self.cube_name = G.CUBE[0]
        self.layout = np.zeros(n, dtype=int)
        self.cylinders = [[] for _ in range(n)]
        self.est = [None] * n
        self.attempt = np.zeros(n, dtype=int)
        self.last_action = np.zeros((n, T.ACT_DIM))
        self.last_outcome = np.zeros(n, dtype=int)
        self._rew = torch.zeros(n, device=self.device)
        self._term = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._trunc = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.stuck = 0                 # returns home that had to be teleports
        self.home_tool = None          # the tool pose at HOME, from the first reset
        self.unseen = 0                # evaluation: resets where perception found no cube
        self.stuck_why = []

    @property
    def max_episode_length(self):
        return self.task.max_attempts

    # --- episodes -------------------------------------------------------------------------

    def reset_cells(self, env_ids, options):
        env_ids = list(env_ids)
        chosen = options.get("layouts")
        if chosen is not None:
            rows = np.asarray([chosen[e] if np.ndim(chosen) else chosen for e in env_ids])
        else:
            rows = self.bank.draw(self.rng, len(env_ids), self.task)
        opts, bodies = [], []
        for e, row in zip(env_ids, rows):
            cube, cyl = T.layout_from_bank(self.bank, int(row))
            self.layout[e], self.cylinders[e] = int(row), cyl
            opts.append(ResetOptions(
                payload_poses={self.cube_name: [cube[0], cube[1], G.TABLE_TOP + G.CUBE_SIZE / 2]
                               + T.cube_quat(cube[2])},
                body_poses=T.body_poses(cyl),
                clear_map=True, scan=False))
            bodies.append(T.bodies_for_planner(cyl))
        if self.training_mode:
            self.pool.set_world(env_ids, bodies)
        o = self.cell.reset(env_ids, opts)
        if self.home_tool is None:
            self.home_tool = [float(v) for v in o.tool_pose[0]]     # the arm is at HOME
        if self.training_mode:
            for k, e in enumerate(env_ids):
                self.est[e] = self._perceive(e, o.objects[self.cube_name][k])
        else:
            self._scan_and_perceive(env_ids, list(G.SCAN_POSES[:-1]))
        self.attempt[env_ids] = 0
        self.last_action[env_ids] = 0.0
        self.last_outcome[env_ids] = 0
        self.extras["reset"] = {"env_ids": env_ids, "layouts": rows.tolist()}

    def _perceive(self, e, cube_pose):
        """Training: what perception would report, the error model on the truth."""
        return T.synth_estimate(self.rng, self._cube_xyyaw(cube_pose), self.cylinders[e], self.task)

    def _scan_and_perceive(self, env_ids, poses):
        """Evaluation: stop at each view (mapping on the way), look, and estimate.

        The arm is at HOME before and after: the last pose is HOME's own view.
        """
        views = {e: [] for e in env_ids}
        for q in list(poses) + [list(G.HOME)]:
            self.cell.run(env_ids, [[Scan(poses=[q]), Idle(10)] for _ in env_ids])
            for e, v in zip(env_ids, self.cell.camera_view("wrist", env_ids)):
                views[e].append(v)
        boxes = []
        for e in env_ids:
            est = P.perceive(views[e])
            if est is None:
                # Not seen: the policy is told so (visibility 0) and aims at
                # the middle of the workspace -- the attempt will fail, honestly.
                self.unseen += 1
                est = T.Estimate(cube=np.array([np.mean(G.CUBE_XY[0]), np.mean(G.CUBE_XY[1]), 0.0]),
                                 visibility=0.0, residual=G.CUBE_SIZE, cylinders=np.zeros((G.MAX_CYLINDERS, 3)))
            self.est[e] = est
            x, y = est.cube[:2]
            h = G.CUBE_SIZE / 2 + 0.02
            boxes.append(((x - h, y - h, G.TABLE_TOP - 0.01), (x + h, y + h, G.TABLE_TOP + G.CUBE_SIZE + 0.02)))
        self.pool.set_exclude(env_ids, boxes)

    @staticmethod
    def _cube_xyyaw(pose):
        """(x, y, yaw) of a cube pose [x, y, z, qw, qx, qy, qz]; yaw folded into [-45, 45) deg."""
        w, x, y, z = pose[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        return (float(pose[0]), float(pose[1]), float(yaw))

    # --- one attempt per env step ---------------------------------------------------------

    def _pre_physics_step(self, actions):
        self.extras.pop("reset", None)
        n, ids = self.num_envs, list(range(self.num_envs))
        a = np.clip(actions.detach().cpu().numpy().astype(np.float64).reshape(n, -1), -1.0, 1.0)
        est_xy = np.array([self.est[e].cube[:2] for e in ids])
        grips = T.action_to_grip(a, est_xy, self.task)
        before = self.cell.observe(ids).objects[self.cube_name][:, :3].copy()
        self.cell.clear_contacts(ids)

        programs = [leg_ops(tool_pose(*g), close=True, open_first=bool(self.cell.closed[e]),
                            descend=G.DESCEND, lift=G.SCENE.pick["lift_m"], check=True,
                            hold=self.task.hold_steps)
                    for e, g in enumerate(grips)]
        results = self.cell.run(ids, programs)

        o = self.cell.observe(ids)
        cube_now = o.objects[self.cube_name]
        rew = np.zeros(n)
        term = np.zeros(n, dtype=bool)
        trunc = np.zeros(n, dtype=bool)
        info = []
        for e, r in enumerate(results):
            rise = float(cube_now[e, 2] - before[e, 2])
            held = T.holding(o.gripper[e], self.cell.closed[e], self.task)
            lifted = rise >= self.task.lift_min and held
            success = bool(r.ok and lifted)
            contact = bool(self.cell.contact_peak[e] > self.task.contact_force)
            moved = float(np.linalg.norm(cube_now[e, :2] - before[e, :2]))
            pushed = moved > self.task.pushed_xy
            lost = not success and not (G.CUBE_XY[0][0] - 0.05 <= cube_now[e, 0] <= G.CUBE_XY[0][1] + 0.05
                                        and G.CUBE_XY[1][0] - 0.05 <= cube_now[e, 1] <= G.CUBE_XY[1][1] + 0.05)
            outcome = T.classify(r.ok, r.why, o.gripper[e], contact, self.task, lifted)
            rew[e] = T.reward(outcome, success, pushed, contact, self.task)
            term[e] = success or lost
            trunc[e] = not term[e] and self.attempt[e] + 1 >= self.task.max_attempts
            info.append(dict(outcome=T.OUTCOMES[outcome], success=success, why=r.why,
                             contact=float(self.cell.contact_peak[e]), rise=rise, moved=moved,
                             gripper=float(o.gripper[e]), grip=grips[e].tolist(),
                             layout=int(self.layout[e]), attempt=int(self.attempt[e])))
            self.last_action[e] = a[e]
            self.last_outcome[e] = outcome

        # Back to HOME to look again, where the episode goes on.
        going_on = [e for e in ids if not term[e] and not trunc[e]]
        if going_on:
            self._return_home(going_on, o)
            if self.training_mode:
                o2 = self.cell.observe(going_on)
                for k, e in enumerate(going_on):
                    self.est[e] = self._perceive(e, o2.objects[self.cube_name][k])
            else:
                self._scan_and_perceive(going_on, [])      # HOME's view again
        self.attempt += 1
        self._rew = self.to_torch(rew)
        self._term = self.to_torch(term, torch.bool)
        self._trunc = self.to_torch(trunc, torch.bool)
        self.extras["attempt"] = info
        self.extras["stuck"] = self.stuck
        # What rsl_rl's logger averages and plots (extras["log"]).
        done = term | trunc
        outcomes = np.array([i["outcome"] for i in info])
        log = {f"attempt/{name.replace(' ', '_')}": float((outcomes == name).mean())
               for name in T.OUTCOMES}
        log["attempt/success"] = float(np.mean([i["success"] for i in info]))
        if done.any():
            log["episode/success"] = float(np.mean([info[e]["success"] for e in np.nonzero(done)[0]]))
            log["episode/attempts"] = float(np.mean(self.attempt[done]))
        log["stuck_total"] = float(self.stuck)
        if not self.training_mode:
            log["cube_unseen_total"] = float(self.unseen)
        self.extras["log"] = log

    def _return_home(self, env_ids, o):
        """Open, back to where the planned move ended, then planned to HOME; a teleport if that fails.

        Back is Retrace -- to the pre-grasp the planner reached -- wherever the
        arm stopped: below it (the checked descent in reverse) or above it
        (lifted, 25 mm over it on the same vertical). Not a fresh IK for the
        way up, which came back 2 mm from a cylinder where the descent had
        passed at 5 and was refused (random policy, 1 in 60 attempts); and not
        a plan from where the arm stands, which is the planned pre-grasp only
        to within the drive's tracking error: beside a cylinder that was
        enough to put the start inside the planner's margin, and the plan home
        was refused (oracle, a grip with 5 mm to spare). So the plan home
        starts from the planned pre-grasp itself (MoveJ.from_plan_end).
        """
        programs = []
        for k, e in enumerate(env_ids):
            ops = [Grip(close=False)] if self.cell.closed[e] else []
            back = self.cell.plan_end[e] is not None and self.cell.goal[e] is not None
            if back:
                ops += [Retrace(), Idle(20)]
            ops.append(MoveJ(list(G.HOME), from_plan_end=back))
            programs.append(ops)
        results = self.cell.run(env_ids, programs)
        # cuRobo's joint-space planner gives up among the cylinders where its
        # pose planner does not, so the fallbacks, from the pre-grasp: plan to
        # HOME's tool pose and finish in joint space; then straight up clear of
        # the cylinders' tops (checked) and home. Each measured to be needed:
        # random policy, 1 in ~140 attempts each.
        rise = G.TABLE_TOP + G.CYL_HEIGHT + 0.05 - (G.GRASP_Z + G.DESCEND)
        for fallback in (
                lambda: [MoveTo(list(self.home_tool), why="no plan to HOME's pose",
                                from_plan_end=True), MoveJ(list(G.HOME))],
                lambda: [MoveZ(rise, why="no way up over the cylinders", check="escape"),
                         MoveJ(list(G.HOME))]):
            again = [e for e, r in zip(env_ids, results)
                     if not r.ok and self.cell.plan_end[e] is not None
                     and self.cell.goal[e] is not None and self.home_tool is not None]
            if not again:
                break
            fixed = dict(zip(again, self.cell.run(again, [fallback() for _ in again])))
            results = [fixed.get(e, r) for e, r in zip(env_ids, results)]
        stuck = [e for e, r in zip(env_ids, results) if not r.ok]
        for e, r in zip(env_ids, results):
            if not r.ok:
                self.stuck_why.append(f"env {e} layout {self.layout[e]}: {r.why}")
                print(f"[grasp] could not get back to HOME in env {e} "
                      f"(layout {self.layout[e]}): {r.why}", flush=True)
        if stuck:
            # In simulation the arm is put back; on the real arm this is where
            # a person is needed. Counted, so it cannot hide.
            self.stuck += len(stuck)
            self.cell._home(stuck)
            self.cell.idle(stuck, 30)

    # --- what the RL loop reads -----------------------------------------------------------

    def _get_dones(self):
        return self._term, self._trunc

    def _get_rewards(self):
        return self._rew

    def _get_observations(self):
        pol, cri = [], []
        cubes = self.cell.observe(range(self.num_envs)).objects[self.cube_name]
        for e in range(self.num_envs):
            est = self.est[e]
            actor = T.actor_obs(est, self.attempt[e], self.last_action[e], self.last_outcome[e],
                                self.task)
            truth = self._cube_xyyaw(cubes[e])
            pol.append(actor)
            cri.append(T.critic_obs(actor, truth, self.cylinders[e]))
        return {"policy": self.to_torch(np.stack(pol)), "critic": self.to_torch(np.stack(cri))}

    def programs(self, actions):
        return None        # _pre_physics_step runs the attempt itself
