"""The pick-and-place demo loop, for any backend's cell.

Written against cell_api.CellLike only, so the demo does not depend on how the
cell is simulated (isaaclab_client.py runs it on Isaac Lab).
"""

from cell_api import CellLike


def say(msg):
    print(f"[demo] {msg}", flush=True)


def pick_place_cycle(env: CellLike, plan_no, src, dst):
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


def banner(env: CellLike, body_path=lambda name: f"/World/{name}"):
    """What to try in the viewport. `body_path` names an unmapped body's prim."""
    say("-------------------------------------------------------------")
    for name in env.bodies:
        say(f" Drag {body_path(name)} into the arm's path in the viewport.")
    say(" The planner is never told where it is.")
    if env.cams:
        say(f" The cameras ({', '.join(env.cams)}) have to find it, and the")
        say(" arm should route around it.")
    else:
        say(" Mapping is off, so nothing can find it: the arm will")
        say(" drive straight through. This is the A/B control.")
    say("-------------------------------------------------------------")


def static_loop(env: CellLike, on_cycle=None, every=60):
    """Hold HOME and keep stepping, so the cameras can be looked through.

    on_cycle(), if given, is called every `every` steps (a viewer's update, say).
    """
    say("static mode: arm held at HOME, no planning. "
        "Inspect the d435i view, then Ctrl-C or close the window.")
    step = 0
    while env.running:
        env.command_arm(env.scene.home)
        env.idle(1)
        step += 1
        if on_cycle is not None and step % every == 0:
            on_cycle()


def run_forever(env: CellLike, on_cycle=None):
    """Shuttle the payload between the first two targets until the app closes.

    on_cycle(), if given, is called after every pick-and-place cycle.
    """
    pick = env.scene.pick
    say(f"pick-and-place: {list(env.payload)} shuttling between "
        f"{len(env.scene.targets)} pedestals, descend {pick['descend_m']:.3f} m / "
        f"lift {pick['lift_m']:.3f} m by IK, the rest planned")
    plan_no, src, dst = 0, 0, 1
    while env.running:
        plan_no += 1
        if pick_place_cycle(env, plan_no, src, dst):
            src, dst = dst, src      # next time, bring it back
        if on_cycle is not None:
            on_cycle()
