import gymnasium as gym
import jax
import jax.numpy as jnp

from barl_jax.networks.actor import StochasticActor
from barl_jax.networks.critic import QCriticBase
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.policy import Policy, frozen, register_policy


@register_policy
class SAC(Policy):
  """Soft Actor-Critic (Haarnoja et al., 2018): entropy-regularized off-policy
  actor-critic, reparameterized tanh-Gaussian actor, automatic temperature
  tuning. Pessimism comes from whichever critic the `critic:` slot names."""

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    params = self.resolve_params()
    self.entropy_scale = params["entropy_scale"]

    self.critic: QCriticBase = self.build_component("critic", QCriticBase)
    self.actor: StochasticActor = self.build_component("actor", StochasticActor)
    self.entropy: AutotuneRegulator = self.build_component(
      "entropy", AutotuneRegulator, target=self.action_dim * self.entropy_scale)

  def rsample(self, obs, key=None):
    """Reparameterized (action, entropy), differentiable through obs. The tanh
    Jacobian correction needs the extra `act_scale` term since log_prob is the
    density of an affine-transformed variable, not of `action` itself."""
    action, x, (mean, std) = self.actor.rsample(obs, key)

    from barl_jax.networks.actor import normal_log_prob
    log_prob = (normal_log_prob(x, mean, std)
                - self.actor.squash.log_abs_det_jacobian(x, action)
                - jnp.log(self.act_scale[...]))
    return action, -log_prob.sum(axis=-1, keepdims=True)

  def train_step(self, batch):
    obs      = self.normalize_obs(batch["obs"])
    next_obs = self.normalize_obs(batch["next_obs"])
    action   = batch["act"]
    reward   = batch["rew"]
    term     = batch["term"]

    losses = {}
    alpha = self.entropy.value()

    # The bootstrap action and its entropy are targets, not things to train
    # through -- stop_gradient stands in for the torch version's no_grad.
    next_action, next_entropy = jax.lax.stop_gradient(self.rsample(next_obs))
    entropy_bonus = jax.lax.stop_gradient(alpha) * next_entropy

    new_action, entropy = self.rsample(obs)

    self.critic.td_loss(obs, action, next_obs, next_action, reward, term,
                        losses, shift=entropy_bonus)

    self.entropy.loss(entropy, losses)

    # The actor improves against the critic but must not train it. In torch
    # this leak was discarded by giving each optimizer only its own params;
    # with one backward over the whole tree it has to be explicit.
    q_new = frozen(self.critic).value(obs, new_action)
    actor_loss = -(q_new + jax.lax.stop_gradient(alpha) * entropy).mean()
    losses["actor"] = actor_loss

    self.actor.metrics.dist("entropy", entropy)
    self.actor.metrics.vec("action", new_action, limit=1.0)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_obs(observation)
    return self.actor.sample(obs, key), {}
