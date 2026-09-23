"""UR5 + 2F-85 + wrist D435i: pick and place, planned by cuRobo 0.8 against a live map.

The demo loop, and nothing else: the cell itself -- stage, cameras, gripper,
the motion primitives -- is sim_env.SimEnv. This shuttles the scene's payload
between its targets forever, which is what the README's numbers measure.

The scene's `unmapped` bodies are never described to the planner. The arm can
only discover them through the cameras, so avoiding them is proof the map is
actually feeding the planner. Scenes live in scripts/scenes/, and both
processes must be started with the same --scene.

This process runs Isaac Sim ONLY and must never import cuRobo -- see sim_env.

Start planner_server.py first, then:
    python scripts/isaacsim_client.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scenes  # noqa: E402
from rig import DEFAULT_ROBOT, ROBOTS  # noqa: E402
import sim_env  # noqa: E402

# planner_server.py flushes every line; without the same here, this side's
# output sits in the stdout buffer whenever it is redirected to a file, and the
# demo looks hung next to a server that is visibly working.
sys.stdout.reconfigure(line_buffering=True)

_ap = argparse.ArgumentParser()
_ap.add_argument("--robot", default=DEFAULT_ROBOT, choices=sorted(ROBOTS))
_ap.add_argument("--scene", default=scenes.DEFAULT, choices=scenes.available(),
                 help="scene module under scripts/scenes/; must match the server")
_ap.add_argument("--no-mapping", action="store_true")
_ap.add_argument("--no-overhead", action="store_true",
                 help="wrist camera only, for A/B against the fixed camera")
_ap.add_argument("--map-every", type=int, default=6, help="fuse a frame every N sim steps")
_ap.add_argument("--static", action="store_true",
                 help="hold the arm at HOME; no planning, just look")
_ap.add_argument("--depth-lag", type=int, default=2,
                 help="sim steps the depth annotator trails the physics by")
_ap.add_argument("--headless", action="store_true",
                 help="no window; renders only if the cameras need it")
ARGS = _ap.parse_args()

# The only loop this demo has is pick-and-place. Checked before Isaac Sim
# starts, so a scene without anything to pick fails in a second, not a minute.
if not scenes.load(ARGS.scene).payload:
    raise SystemExit(f"scene {ARGS.scene!r} has no payload; this client only "
                     f"runs pick-and-place")


def say(msg):
    print(f"[demo] {msg}", flush=True)


def pick_place_cycle(env, plan_no, src, dst):
    """Lift the payload off `src` and set it down on `dst`.

    Called with src and dst swapped each time, so the block shuttles back and
    forth rather than being picked once and abandoned.
    """
    scene = env.scene
    for idx, grab in ((src, True), (dst, False)):
        what = f"pick@{idx}" if grab else f"place@{idx}"
        r = env.move_to(scene.targets[idx])
        if not r.ok:
            say(f"plan #{plan_no} ({what}) blocked - waiting")
            env.idle(30)
            return False
        say(f"plan #{plan_no} -> {what}: solve {r.solve_ms:.0f} ms, "
            f"{r.waypoints} waypoints")
        env.idle(20)

        if not env.move_tool_z(-scene.pick["descend_m"]).ok:
            return False
        env.idle(15)
        g = env.grip(close=grab)
        say(f"  {'close' if grab else 'open'}: settled after {g.steps} steps"
            f"{'' if g.settled else ' (TIMED OUT, still moving)'}"
            f", want {g.target:+.3f} rad")
        say("    " + " ".join(f"{n}={v:+.3f}" for n, v in g.spread.items()))
        env.move_tool_z(scene.pick["lift_m"])
        env.idle(20)

        for name, pose in env.observe().objects.items():
            say(f"  after {what}: {name} at "
                f"[{pose[0]:+.3f} {pose[1]:+.3f} {pose[2]:+.3f}]")
    return True


def main():
    sim_env.launch(headless=ARGS.headless)
    env = sim_env.SimEnv(sim_env.EnvCfg(
        scene=ARGS.scene, robot=ARGS.robot, mapping=not ARGS.no_mapping,
        overhead=not ARGS.no_overhead, map_every=ARGS.map_every,
        depth_lag=ARGS.depth_lag))
    # The map is empty on a fresh server; nothing to clear.
    env.reset(sim_env.ResetOptions(block_on=0, clear_map=False, scan=True))

    say("-------------------------------------------------------------")
    for name in env.bodies:
        say(f" Drag /World/{name} into the arm's path in the viewport.")
    say(" The planner is never told where it is.")
    if env.cams:
        say(f" The cameras ({', '.join(env.cams)}) have to find it, and the")
        say(" arm should route around it.")
    else:
        say(" Mapping is off, so nothing can find it: the arm will")
        say(" drive straight through. This is the A/B control.")
    say("-------------------------------------------------------------")

    if ARGS.static:
        say("static mode: arm held at HOME, no planning. "
            "Inspect the d435i view, then Ctrl-C or close the window.")
        while env.running:
            env.command_arm(env.scene.home)
            env.idle(1)
        env.close()
        return

    pick = env.scene.pick
    say(f"pick-and-place: {list(env.payload)} shuttling between "
        f"{len(env.scene.targets)} pedestals, descend {pick['descend_m']:.3f} m / "
        f"lift {pick['lift_m']:.3f} m by IK, the rest planned")
    plan_no, src, dst = 0, 0, 1
    while env.running:
        plan_no += 1
        if pick_place_cycle(env, plan_no, src, dst):
            src, dst = dst, src      # next time, bring it back

    env.close()


if __name__ == "__main__":
    main()
