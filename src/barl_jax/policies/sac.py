import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barl_jax.networks import gaussian
from barl_jax.networks.critic import QCriticBase
from barl_jax.networks.mlp import MLP
from barl_jax.networks.optim import adam
from barl_jax.networks.registry import get_component, register_component
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.policy import Actor, Buffer, Critic, Policy, frozen, register_policy


@register_component
class StochasticActor(Actor):
  """Diagonal-Normal MLP actor shared by SAC/PPO/IQL. `dist()` stays an
  unconstrained Normal; sample()/mode() squash it (`squash_cls`, resolved by
  name from config/models.yaml -- any no-arg Squash works generically here).

  `log_std` is a state-independent parameter rather than a network head --
  barl's deliberate deviation from canonical SAC, kept as-is."""

  def __init__(self, env_stats, *, latent_dim: int, layers: int, squash_cls: str,
               lr: float, adam_b1: float, grad_clip: float, grad_clip_mode: str,
               log_std_min: float, log_std_max: float, log_prob_clip: float,
               activation: str, norm: str,
               norm_input: bool, norm_stem: bool, norm_after_act: bool,
               dropout: float, final_activation: bool,
               residual: bool, norm_kwargs: dict, dtype: str,
               rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, **kwargs)

    self.lr, self.adam_b1 = float(lr), float(adam_b1)
    self.grad_clip, self.grad_clip_mode = float(grad_clip), str(grad_clip_mode)
    self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)
    self.log_prob_clip = float(log_prob_clip)
    self.log_std = nnx.Param(jnp.zeros((self.action_dim,), jnp.float32))
    self.squash  = get_component(squash_cls)(env_stats)

    low, high = env_stats.get("act_low"), env_stats.get("act_high")
    scale = 1.0 if low is None else (np.asarray(high) - np.asarray(low)) / 2
    self.log_act_scale = Buffer(jnp.log(jnp.asarray(scale, jnp.float32)))

    self.net = MLP(self.obs_dim, self.action_dim, latent_dim, layers,
                   activation=activation, norm=norm, norm_input=norm_input,
                         norm_stem=norm_stem, norm_after_act=norm_after_act,
                   dropout=dropout, final_activation=final_activation,
                   residual=residual, dtype=dtype, rngs=rngs, **(norm_kwargs or {}))
    self.rngs = rngs

  def dist(self, s):
    """(mean, std) of the pre-squash Normal."""
    mean = self.net(s).astype(jnp.float32)
    std  = jnp.exp(jnp.clip(self.log_std[...], self.log_std_min, self.log_std_max))
    return mean, std

  def log_prob(self, dist, action):
    """`action` is the squashed (canonical [-1, 1]) value; inverts the squash
    and applies the change-of-variables correction."""
    mean, std = dist
    x = self.squash.inverse(jnp.clip(action, -self.log_prob_clip, self.log_prob_clip))
    lp = gaussian.log_prob(x, mean, std) - self.squash.log_abs_det_jacobian(x, action)
    return lp.sum(axis=-1)

  def rsample(self, s, key=None):
    """Reparameterized (action, pre-squash x, dist) -- differentiable through s."""
    mean, std = self.dist(s)
    key = self.rngs.noise() if key is None else key
    x = mean + std * jax.random.normal(key, mean.shape, mean.dtype)
    return self.squash(x), x, (mean, std)

  def generate_action(self, s, key=None):
    """Reparameterized (action, entropy), differentiable through `s`. The squash
    Jacobian correction needs the extra `act_scale` term since log_prob is the
    density of an affine-transformed variable, not of `action` itself."""
    action, x, (mean, std) = self.rsample(s, key)

    log_prob = (gaussian.log_prob(x, mean, std)
                - self.squash.log_abs_det_jacobian(x, action)
                - self.log_act_scale[...])
    return action, -log_prob.sum(axis=-1, keepdims=True)

  def sample(self, s, key=None, critic: Critic = None):
    action, _, _ = self.rsample(s, key)
    return action

  def mode(self, s):
    """Deterministic (mean) action -- IQL trains via AWR but deploys this."""
    mean, _ = self.dist(s)
    return self.squash(mean)

  def actor_loss(self, obs, critic: Critic, alpha, losses: dict, key=None):
    """SAC's reparameterized max-entropy objective, -(Q + alpha*H). Returns the
    entropy so the owning Policy can feed its temperature dual the same estimate.

    The actor improves against the critic but must not train it. In torch this
    leak was discarded by giving each optimizer only its own params; with one
    backward over the whole tree it has to be explicit."""
    action, entropy = self.generate_action(obs, key=key)
    q = frozen(critic).value(obs, action)
    losses["actor"] = -(q + jax.lax.stop_gradient(alpha) * entropy).mean()

    self.metrics.dist("entropy", entropy)
    self.metrics.vec("action", action, limit=1.0)
    return entropy

  def create_optimizer(self) -> dict:
    return {"actor": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}


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

  def train_step(self, batch):
    obs      = self.normalize_obs(batch["obs"])
    next_obs = self.normalize_obs(batch["next_obs"])
    action   = batch["act"]
    reward   = batch["rew"]
    term     = batch["term"]

    losses = {}
    alpha = self.entropy.value()

    # Every draw this step descends from one key, split here: the bootstrap
    # sample and the reparameterized one must not share a stream, and a step's
    # randomness should be readable off train_step rather than hidden in each
    # component's `rngs`.
    keys = jax.random.split(self.rngs.noise(), 2)

    # The bootstrap action and its entropy are targets, not things to train
    # through -- stop_gradient stands in for the torch version's no_grad.
    next_action, next_entropy = jax.lax.stop_gradient(
      self.actor.generate_action(next_obs, key=keys[0]))
    entropy_bonus = jax.lax.stop_gradient(alpha) * next_entropy

    self.critic.td_loss(obs, action, next_obs, next_action, reward, term,
                        losses, shift=entropy_bonus)

    entropy = self.actor.actor_loss(obs, self.critic, alpha, losses, key=keys[1])

    self.entropy.loss(entropy, losses)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_obs(observation)
    return self.actor.sample(obs, key=key), {}
