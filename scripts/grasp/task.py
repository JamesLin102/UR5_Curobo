"""The grasp task, simulator-free: layouts, what the policy sees, what it is paid.

The design in numpy, shared by the Isaac Lab environment (lab_env.py) and anything that wants to score or sample
without a simulator (the layout bank, checks). Nothing here imports Isaac or
cuRobo: the perception half has to run on the real arm as well.

One episode: a layout from the bank (the cube and 0-4 cylinders), then up to
max_attempts grips. One step is one attempt:

    the policy says where to grip, relative to where perception puts the cube
    plan above it, down (checked), close, up (checked), hold
    lifted and held -> success, the episode ends
    otherwise       -> open, back to HOME, look again, next attempt

Frames. Every position the policy sees or gives is in a frame aligned with the
WORLD and centred on the ESTIMATED cube, never one turned with the cube: the
cube's yaw is only defined modulo 90 deg, and a frame turned by it would jump
by 90 deg at +-45 while cos 4th, sin 4th did not.
"""

import math
from dataclasses import dataclass

import numpy as np

from . import scene as G

# --- configuration ----------------------------------------------------------------


@dataclass
class TaskCfg:
    # One attempt per episode, among layouts where only one grip works
    # (decided 2026-09-24). With three attempts random actions lifted the cube
    # in 96% of episodes and the oracle in 100%, so there was little to learn;
    # first attempts on one-grip layouts are 50% random, 96% oracle.
    # The observation keeps the last-attempt fields for when this is raised.
    max_attempts: int = 1
    hold_steps: int = 30            # held this long at the top before it counts
    lift_min: float = 0.075         # the cube must rise this much (half of lift_m)
    # Holding, read the way the real 2F-85 can: the leader joint stalls short of
    # closed on a 45 mm cube (+0.405..+0.426 rad measured) and closes fully on
    # nothing. Holding = closed and below this.
    hold_below: float = 0.60
    contact_force: float = 1.0      # N: a solid body felt more than this = touched
    pushed_xy: float = 0.03         # the cube moved this far and was not lifted
    # Reward
    r_success: float = 1.0
    r_attempt: float = -0.05
    r_failed_motion: float = -0.1   # no plan, no IK, a straight move refused
    r_contact: float = -0.3
    r_pushed: float = -0.5
    # Action ranges: [-1, 1] maps onto these
    dxy: float = 0.02
    dz_lo: float = -0.010
    dz_hi: float = 0.015
    # Perception model: what the estimate the policy sees is off by
    pos_sigma: float = 0.002        # m, at full visibility
    yaw_sigma_deg: float = 2.0
    cyl_sigma: float = 0.003
    miss_p: float = 0.02            # a cylinder not reported
    false_p: float = 0.02           # a cylinder reported that is not there
    min_visibility: float = 0.6     # below this the cube counts as not seen
    noise: bool = True              # False: the estimate is the truth (debugging)
    # The bank: "train" or "eval", or a path
    bank: str = "train"
    # Curriculum: the most cylinders a layout may have, and the fewest mm
    # between the cube and the nearest one. Widened from outside.
    max_cylinders: int = G.MAX_CYLINDERS
    min_gap: float = 0.015
    # The share of episodes drawn from layouts where only ONE of the two
    # face-square grips works: where choosing is the task. The rest
    # are drawn from every eligible layout.
    one_grip_fraction: float = 1.0
    # A grip counts as working only with this much between the arm's collision
    # spheres and the nearest cylinder, all the way down and up. The spheres do
    # not cover the arm's meshes exactly: grips the bank kept at 6 mm touched a
    # cylinder in the simulator at ~300 N (tools/check_grasp_env.py). The
    # planner server's straight-move check holds the same margin.
    min_clearance: float = 0.015


# Outcome of an attempt, as the policy is told next time (one-hot).
OUTCOMES = ("none", "no plan", "no IK", "move refused", "empty", "contact")
N_OUT = len(OUTCOMES)

ACT_DIM = 4
OBS_DIM = 2 + 2 + 2 + 3 * G.MAX_CYLINDERS + 1 + ACT_DIM + N_OUT          # 29
STATE_DIM = OBS_DIM + 4 + 3 * G.MAX_CYLINDERS                           # 45


def wrap_yaw(a):
    """Into [-pi/2, pi/2): a parallel gripper at a and at a + pi grips the same faces."""
    return (np.asarray(a) + math.pi / 2) % math.pi - math.pi / 2


def cube_quat(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def action_to_grip(action, est_xy, cfg: TaskCfg):
    """(k, 4) in [-1, 1] -> (k, 4) grip (x, y, z, tool yaw), world frame, env-local."""
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1, ACT_DIM), -1.0, 1.0)
    x = est_xy[:, 0] + a[:, 0] * cfg.dxy
    y = est_xy[:, 1] + a[:, 1] * cfg.dxy
    z = G.GRASP_Z + cfg.dz_lo + (a[:, 2] + 1) / 2 * (cfg.dz_hi - cfg.dz_lo)
    yaw = a[:, 3] * math.pi / 2
    return np.stack([x, y, z, yaw], axis=1)


def oracle_action(est_yaw, face):
    """The grip an oracle makes: centred, at GRASP_Z, square to face 0 or 1."""
    yaw = wrap_yaw(est_yaw + face * math.pi / 2)
    z = (0.0 - TaskCfg.dz_lo) / (TaskCfg.dz_hi - TaskCfg.dz_lo) * 2 - 1
    return np.array([0.0, 0.0, z, yaw / (math.pi / 2)])


# --- geometry: what the wrist camera can see ------------------------------------------


def _ray_hits_cylinder(p, d, c, r, z0, z1):
    """Does the segment p -> p + d (d not normalised, t in [0, 1]) pass through the
    vertical cylinder centred (c) radius r between heights z0..z1?"""
    ox, oy = p[..., 0] - c[0], p[..., 1] - c[1]
    dx, dy = d[..., 0], d[..., 1]
    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    cc = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * cc
    hit = np.zeros(p.shape[:-1], dtype=bool)
    ok = (disc >= 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    for t in ((-b - sq) / (2 * np.where(ok, a, 1)), (-b + sq) / (2 * np.where(ok, a, 1))):
        z = p[..., 2] + t * d[..., 2]
        hit |= ok & (t > 1e-6) & (t < 1) & (z >= z0) & (z <= z1)
    return hit


def cube_visibility(cube, cylinders, eyes, n=7):
    """Fraction of the cube's top face each camera eye can see, (len(eyes),).

    cube (x, y, yaw); cylinders [(x, y)]; eyes [(x, y, z)]. A grid of points on
    the top face, each visible if the segment to the eye misses every cylinder.
    Geometry only -- lighting, the arm in view and depth noise are not in it.
    """
    x, y, yaw = cube
    h = G.CUBE_SIZE / 2 * (1 - 1.0 / n)
    u = np.linspace(-h, h, n)
    gu, gv = np.meshgrid(u, u)
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.stack([x + c * gu - s * gv, y + s * gu + c * gv,
                    np.full_like(gu, G.TABLE_TOP + G.CUBE_SIZE)], axis=-1).reshape(-1, 3)
    out = []
    for eye in eyes:
        d = np.asarray(eye, dtype=np.float64) - pts
        blocked = np.zeros(len(pts), dtype=bool)
        for cx, cy in cylinders:
            blocked |= _ray_hits_cylinder(pts, d, (cx, cy), G.CYL_RADIUS,
                                          G.TABLE_TOP, G.TABLE_TOP + G.CYL_HEIGHT)
        out.append(1.0 - blocked.mean())
    return np.asarray(out)


SCAN_EYES = [v[1] for v in G.VIEWS]       # the camera positions the scan looks from


# --- the layout bank ---------------------------------------------------------------------


class Bank:
    """Layouts checked offline to have a grip (tools/grasp_layout_bank.py).

    Arrays, one row per layout:
        cube      (N, 3)     x, y, yaw
        cyl       (N, M, 3)  x, y, present (1/0), M = MAX_CYLINDERS
        feasible  (N, 2)     which of the two face-square grips works
        clear     (N, 2)     its clearance to the nearest cylinder, m (nan if not)
        gap       (N,)       nearest cylinder to the cube, surface to surface, m
        vis       (N, V)     how much of the cube's top each scan view sees
    """

    def __init__(self, path):
        d = np.load(path)
        self.path = path
        self.cube, self.cyl = d["cube"], d["cyl"]
        self.feasible, self.clear = d["feasible"], d["clear"]
        self.gap, self.vis = d["gap"], d["vis"]
        self.n_cyl = self.cyl[..., 2].sum(axis=1).astype(int)

    def __len__(self):
        return len(self.cube)

    def usable(self, cfg: TaskCfg):
        """(N, 2): which grips work with cfg.min_clearance to spare."""
        return self.feasible & (np.nan_to_num(self.clear, nan=-1.0) >= cfg.min_clearance)

    def eligible(self, cfg: TaskCfg):
        return np.nonzero((self.n_cyl <= cfg.max_cylinders)
                          & (self.gap >= cfg.min_gap - 1e-9)
                          & (self.vis.max(axis=1) >= cfg.min_visibility)
                          & self.usable(cfg).any(axis=1))[0]

    def draw(self, rng, k, cfg: TaskCfg):
        idx = self.eligible(cfg)
        if not len(idx):
            raise ValueError(f"no layout in {self.path} fits the curriculum "
                             f"(max_cylinders {cfg.max_cylinders}, min_gap {cfg.min_gap})")
        one = idx[self.usable(cfg)[idx].sum(axis=1) == 1]
        pick_one = rng.random(k) < (cfg.one_grip_fraction if len(one) else 0.0)
        out = rng.choice(idx, size=k)
        if pick_one.any():
            out[pick_one] = rng.choice(one, size=int(pick_one.sum()))
        return out


def bank_path(name):
    import os
    if name in ("train", "eval"):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "layouts", f"{name}.npz")
    return name


# --- perception, as the policy will get it on the real arm -----------------------------------


@dataclass
class Estimate:
    cube: np.ndarray            # (3,) x, y, yaw
    visibility: float
    residual: float             # m, how badly a square fits what was seen
    cylinders: np.ndarray       # (M, 3) x, y, present


def synth_estimate(rng, cube, cylinders, cfg: TaskCfg) -> Estimate:
    """What perception would report, made from the truth and the error model.

    cube (x, y, yaw) and cylinders [(x, y)] are the truth, where they stand now.
    The visibility is the best scan view's; the less of the cube is seen, the
    worse the fit. Training uses this; evaluation and the real arm use
    perception.py, whose errors this is to be calibrated against: on the
    simulator by tools/check_grasp_perception.py, on the real arm by measuring.
    """
    vis = float(cube_visibility(cube, cylinders, SCAN_EYES).max())
    scale = 1.0 + 2.0 * (1.0 - vis)
    est = np.array(cube, dtype=np.float64)
    residual = 0.0005
    if cfg.noise:
        est[:2] += rng.normal(0.0, cfg.pos_sigma * scale, 2)
        est[2] += rng.normal(0.0, math.radians(cfg.yaw_sigma_deg) * scale)
        residual = abs(rng.normal(0.0005, 0.0005 * scale))
    est[2] = (est[2] + math.pi / 4) % (math.pi / 2) - math.pi / 4
    cyl = np.zeros((G.MAX_CYLINDERS, 3))
    seen = []
    for cx, cy in cylinders:
        if cfg.noise and rng.random() < cfg.miss_p:
            continue
        n = rng.normal(0.0, cfg.cyl_sigma, 2) if cfg.noise else np.zeros(2)
        seen.append((cx + n[0], cy + n[1]))
    if cfg.noise and rng.random() < cfg.false_p and len(seen) < G.MAX_CYLINDERS:
        seen.append((rng.uniform(*G.CYL_XY[0]), rng.uniform(*G.CYL_XY[1])))
    for i, (cx, cy) in enumerate(seen[:G.MAX_CYLINDERS]):
        cyl[i] = (cx, cy, 1.0)
    return Estimate(cube=est, visibility=vis, residual=residual, cylinders=cyl)


# --- observations -------------------------------------------------------------------------


def _cylinders_rel(cyl, centre):
    """(M, 3) -> (3M,): offsets from `centre`, world-aligned, nearest first; absent last."""
    present = cyl[:, 2] > 0.5
    rel = np.zeros_like(cyl)
    rel[:, :2] = cyl[:, :2] - centre
    rel[:, 2] = present
    dist = np.where(present, np.linalg.norm(rel[:, :2], axis=1), np.inf)
    rel = rel[np.argsort(dist)]
    rel[rel[:, 2] < 0.5, :2] = 0.0
    return rel.reshape(-1)


def actor_obs(est: Estimate, attempt, last_action, last_outcome, cfg: TaskCfg):
    """(OBS_DIM,) float32 -- everything in it is something the real arm has."""
    x, y, yaw = est.cube
    one_hot = np.zeros(N_OUT)
    one_hot[last_outcome] = 1.0
    v = np.concatenate([
        [(x - 0.45) / 0.15, y / 0.35],                      # roughly [-1, 1] over CUBE_XY
        [math.cos(4 * yaw), math.sin(4 * yaw)],
        [est.visibility, est.residual / 0.005],
        _cylinders_rel(est.cylinders, est.cube[:2]) / np.array([0.2, 0.2, 1.0] * G.MAX_CYLINDERS),
        [attempt / max(cfg.max_attempts - 1, 1)],
        np.asarray(last_action, dtype=np.float64),
        one_hot,
    ])
    return v.astype(np.float32)


def critic_obs(actor, cube, cylinders):
    """(STATE_DIM,): the actor's, plus the truth it is estimating (simulation only)."""
    x, y, yaw = cube
    cyl = np.zeros((G.MAX_CYLINDERS, 3))
    for i, (cx, cy) in enumerate(cylinders[:G.MAX_CYLINDERS]):
        cyl[i] = (cx, cy, 1.0)
    return np.concatenate([
        actor,
        [(x - 0.45) / 0.15, y / 0.35, math.cos(4 * yaw), math.sin(4 * yaw)],
        _cylinders_rel(cyl, np.array([x, y])) / np.array([0.2, 0.2, 1.0] * G.MAX_CYLINDERS),
    ]).astype(np.float32)


# --- outcomes and reward ---------------------------------------------------------------------


def classify(ok, why, closed_angle, contact, cfg: TaskCfg, lifted):
    """One attempt -> index into OUTCOMES (what the policy is told next time)."""
    if not ok:
        w = (why or "").lower()
        if "straight move" in w:
            return OUTCOMES.index("move refused")
        if "no plan" in w:
            return OUTCOMES.index("no plan")
        return OUTCOMES.index("no IK")
    if contact:
        return OUTCOMES.index("contact")
    if not lifted:
        return OUTCOMES.index("empty")
    return OUTCOMES.index("none")


def reward(outcome, success, pushed, contact, cfg: TaskCfg):
    r = cfg.r_attempt
    if OUTCOMES[outcome] in ("no plan", "no IK", "move refused"):
        r += cfg.r_failed_motion
    if contact:
        r += cfg.r_contact
    if pushed and not success:
        r += cfg.r_pushed
    if success:
        r += cfg.r_success
    return r


def holding(gripper_angle, closed, cfg: TaskCfg):
    """Closed, and stalled short of fully closed: something is between the fingers."""
    return bool(closed) and float(gripper_angle) < cfg.hold_below


def bodies_for_planner(cylinders):
    """The planner's world for an environment: [(name, shape, dims, pose)], env-local."""
    return [(f"cyl_{i}", "cylinder", [2 * G.CYL_RADIUS, 2 * G.CYL_RADIUS, G.CYL_HEIGHT],
             [cx, cy, G.TABLE_TOP + G.CYL_HEIGHT / 2, 1.0, 0.0, 0.0, 0.0])
            for i, (cx, cy) in enumerate(cylinders)]


def body_poses(cylinders) -> dict:
    """ResetOptions.body_poses for a layout: the scene's cylinders placed, the rest out."""
    out = {}
    for i, (name, *_rest) in enumerate(G.CYLINDERS):
        if i < len(cylinders):
            cx, cy = cylinders[i]
            out[name] = [cx, cy, G.TABLE_TOP + G.CYL_HEIGHT / 2, 1.0, 0.0, 0.0, 0.0]
        else:
            out[name] = None
    return out


def layout_from_bank(bank: Bank, i):
    """(cube (x, y, yaw), [(x, y)] cylinders) of bank row i."""
    cyl = [(float(c[0]), float(c[1])) for c in bank.cyl[i] if c[2] > 0.5]
    return tuple(float(v) for v in bank.cube[i]), cyl
