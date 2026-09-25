"""The pick-and-place example: a block shuttled between two pedestals past a slab.

    scene.py            the SceneSpec (`--scene pick_place`)
    task.py             goals, rewards, observations; no simulator
    demo_loop.py        the demo loop, for any cell_api.CellLike
    lab_env.py          Isaac Lab gym task, Isaac-PickPlace-{Robot}-v0
    isaaclab_client.py  the demo on Isaac Lab

Importing the package imports nothing: the planner server reads scene.py
from a process that has neither simulator, so keep this file empty of code.
"""
