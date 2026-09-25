"""The grasp example: pick a cube from among tall cylinders, wrist camera only.

    scene.py            the SceneSpec (`--scene grasp`); HOME and the scan views
    task.py             layouts, observations, rewards; no simulator
    perception.py       cube and cylinders from the scan's images; numpy and scipy only
    lab_env.py          Isaac Lab gym task, Isaac-Grasp-{Robot}-v0
    lab_rl_cfg.py       PPO (rsl_rl)
    lab_policy.py       a checkpoint, the oracle or random actions, as obs -> action
    isaaclab_train.py   training, starting its own planner servers
    isaaclab_eval.py    a checkpoint, the oracle or random actions, in either mode
    layouts/            the layout banks (tools/grasp_layout_bank.py)
    policies/first.pt   the first training run's last checkpoint (model_499.pt)

Importing the package imports nothing: the planner server reads scene.py
from a process that has neither simulator, so keep this file empty of code.
"""
