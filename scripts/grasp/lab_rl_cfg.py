"""PPO for the grasp task (rsl_rl 5, through Isaac Lab's wrapper).

A starting point, not a tuned one -- nothing has been trained yet. What shaped it:

  - An episode is at most max_attempts (3) steps and pays out at its end, so
    the horizon is short: gamma 0.9 discounts a success two attempts later
    by 0.81, which still makes trying again worth it.
  - The actor sees what the real arm has (29 numbers), the critic that plus the
    truth (45): obs_groups maps the env's "policy" and "critic" groups onto them.
  - Both are small MLPs with observation normalisation; the observations are
    already roughly unit-scaled (grasp.task.actor_obs).
  - A step costs seconds of simulation, so a rollout is short and every sample
    is reused: 8 steps per env per iteration, 8 epochs.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class GraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 8
    max_iterations = 500
    save_interval = 25
    experiment_name = "grasp"
    clip_actions = 1.0
    empirical_normalization = False     # the models normalise their own inputs
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.9,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
