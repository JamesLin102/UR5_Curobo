"""Something to act in any registered task: a trained checkpoint, the task's oracle, or random.

    act = make(env, "scripts/grasp/policies/first.pt")   # or "oracle", or "random"
    obs, _ = env.reset()
    obs, rew, term, trunc, extras = env.step(act(obs))

A checkpoint is rsl_rl's, loaded with the runner config the task registered
(lab.tasks.TASKS "rsl_rl") and run deterministically. "oracle" is the function
the task registered ("oracle"): one that knows what the policy cannot see.
"random" is uniform over the action space. Loading a checkpoint wraps the env
in Isaac Lab's rsl_rl wrapper, which resets it once.
"""

import torch

from .tasks import entry, load


def runner_cfg(tid, rl_device="cuda:0"):
    """The task's rsl_rl runner config, ready for this rsl_rl version."""
    import importlib.metadata as md

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    cfg_entry = entry(tid, "rsl_rl_cfg_entry_point")
    if cfg_entry is None:
        raise SystemExit(f"{tid} has no rsl_rl runner config; add \"rsl_rl\" to its "
                         f"lab.tasks.TASKS entry to train or load a policy for it")
    agent = handle_deprecated_rsl_rl_cfg(load(cfg_entry)(), md.version("rsl-rl-lib"))
    agent.device = rl_device
    return agent


def make(env, policy, rl_device="cuda:0", log=print):
    """obs -> actions tensor, for "oracle", "random" or a model_N.pt checkpoint's path."""
    u = env.unwrapped
    tid = env.spec.id
    if policy == "random":
        low = torch.as_tensor(u.single_action_space.low, dtype=torch.float32, device=u.device)
        high = torch.as_tensor(u.single_action_space.high, dtype=torch.float32, device=u.device)
        return lambda obs: low + (high - low) * torch.rand((u.num_envs, len(low)), device=u.device)
    if policy == "oracle":
        oracle_entry = entry(tid, "oracle_entry_point")
        if oracle_entry is None:
            raise SystemExit(f"{tid} has no oracle; add \"oracle\" to its lab.tasks.TASKS entry")
        oracle = load(oracle_entry)
        return lambda obs: oracle(env)

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    agent = runner_cfg(tid, rl_device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(policy)
    net = runner.get_inference_policy(device=agent.device)
    log(f"policy from {policy}")
    clip = agent.clip_actions

    def act(obs):
        with torch.inference_mode():
            # The networks may be on the GPU and the env's physics on the CPU.
            a = net(wrapped.get_observations().to(agent.device)).to(u.device)
            return a.clamp(-clip, clip) if clip else a
    return act
