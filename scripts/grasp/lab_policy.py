"""Something to act in Isaac-Grasp-*: a trained checkpoint, the oracle, or random actions.

    act = make(env, "grasp/policies/first.pt")     # or "oracle", or "random"
    obs, _ = env.reset()
    obs, rew, term, trunc, extras = env.step(act(obs))

The oracle knows from the layout bank which face-square grip can be planned
and aims it at perception's estimate: what a policy could at best learn, since
it reads the one thing the policy cannot see. On a second attempt it turns to
the other face if both work. A checkpoint is rsl_rl's (grasp.lab_rl_cfg), run
deterministically. Loading one wraps the env in Isaac Lab's rsl_rl wrapper,
which resets it: anything counting resets should start counting after make().
"""

import numpy as np
import torch

from . import task as T


def oracle_actions(env):
    """(num_envs, ACT_DIM): the face the bank says works, aimed at the estimate."""
    u = env.unwrapped
    usable = u.bank.usable(u.task)
    acts = []
    for e in range(u.num_envs):
        feas = usable[u.layout[e]]
        face = 0 if feas[0] else 1
        if u.attempt[e] > 0 and feas.all():
            face = int(u.attempt[e] % 2 == 1) ^ face
        acts.append(T.oracle_action(u.est[e].cube[2], face))
    return torch.tensor(np.array(acts), dtype=torch.float32, device=u.device)


def make(env, policy, rl_device="cuda:0", log=print):
    """obs -> actions, for "oracle", "random" or a model_N.pt checkpoint's path."""
    u = env.unwrapped
    if policy == "oracle":
        return lambda obs: oracle_actions(env)
    if policy == "random":
        return lambda obs: torch.rand((u.num_envs, T.ACT_DIM), device=u.device) * 2 - 1

    import importlib.metadata as md

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    from .lab_rl_cfg import GraspPPORunnerCfg

    agent = handle_deprecated_rsl_rl_cfg(GraspPPORunnerCfg(), md.version("rsl-rl-lib"))
    agent.device = rl_device
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(policy)
    net = runner.get_inference_policy(device=agent.device)
    log(f"policy from {policy}")

    def act(obs):
        with torch.inference_mode():
            # The networks may be on the GPU and the env's physics on the CPU.
            return net(wrapped.get_observations().to(agent.device)).to(u.device).clamp(-1, 1)
    return act
