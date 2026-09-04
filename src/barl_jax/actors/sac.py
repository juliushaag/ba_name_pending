import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barl_jax.configs import ConfigMixin
from barl_jax.hparam import Config, HParam
from barl_jax.networks import gaussian
from barl_jax.networks.mlp import MLP, MLP_STYLES
from barl_jax.networks.optim import adam, ActorOptim
from barl_jax.networks.registry import get_component, register_component
from barl_jax.policy import Actor, Buffer, Critic, frozen


@register_component
class StochasticActor(Actor, ActorOptim):
  """Diagonal-Normal MLP actor shared by SAC/PPO/IQL. `dist()` stays an
  unconstrained Normal; sample()/mode() squash it (the `squash_cls` Config
  selects which -- any no-arg Squash works generically here).

  The shallow Gaussian policy has just a state-independent `log_std` plus the
  std bounds; the net is the `layer` style (a LayerNorm stem over silu) by
  default -- switch it with `net_style` (a configuration value, not
  inheritance).

  `log_std` is a state-independent parameter rather than a network head --
  barl's deliberate deviation from canonical SAC, kept as-is."""

  kind = "actor"
  net_style = Config("layer", style=MLP_STYLES)
  squash_cls = Config("TanhSquash")
  log_std_min  = Config(-20.0)
  log_std_max  = Config(2.0)
  log_prob_clip = Config(0.99)   # tanh atanh guard in log_prob

  def __init__(self, env_stats, config: ConfigMixin, *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, config, **kwargs)

    self.log_std = nnx.Param(jnp.zeros((self.action_dim,), jnp.float32))
    self.squash  = get_component(self.squash_cls)(env_stats, ConfigMixin())

    low, high = env_stats.get("act_low"), env_stats.get("act_high")
    scale = 1.0 if low is None else (np.asarray(high) - np.asarray(low)) / 2
    self.log_act_scale = Buffer(jnp.log(jnp.asarray(scale, jnp.float32)))

    # latent_dim/layers are size-ladder-owned (see size ladders in hparam.py).
    latent_dim, layers = config.get("latent_dim", int), config.get("layers", int)

    self.net = MLP(self.obs_dim, self.action_dim, latent_dim, layers,
                   activation=self.activation, norm=self.norm, norm_input=self.norm_input,
                         norm_stem=self.norm_stem, norm_after_act=self.norm_after_act,
                   dropout=self.dropout, final_activation=self.final_activation,
                   residual=self.residual, dtype=self.dtype, rngs=rngs, **(self.hparam("norm_kwargs") or {}))
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

  def generate_action(self, s, *, critic: Critic = None, key=None):
    """Reparameterized (action, entropy), differentiable through `s`. The squash
    Jacobian correction needs the extra `act_scale` term since log_prob is the
    density of an affine-transformed variable, not of `action` itself.
    `critic` is unused -- SAC's actor needs none to sample -- and accepted only
    to satisfy the shared `Actor.generate_action` contract."""
    action, x, (mean, std) = self.rsample(s, key)

    log_prob = (gaussian.log_prob(x, mean, std)
                - self.squash.log_abs_det_jacobian(x, action)
                - self.log_act_scale[...])
    return action, -log_prob.sum(axis=-1, keepdims=True)

  def sample(self, s, *, critic: Critic = None, key=None):
    action, _, _ = self.rsample(s, key)
    return action

  def mode(self, s):
    """Deterministic (mean) action -- IQL trains via AWR but deploys this."""
    mean, _ = self.dist(s)
    return self.squash(mean)

  def loss(self, obs, critic: Critic, alpha, *, key=None):
    """SAC's reparameterized max-entropy objective, -(Q + alpha*H).

    The actor improves against the critic but must not train it. In torch this
    leak was discarded by giving each optimizer only its own params; with one
    backward over the whole tree it has to be explicit."""
    action, entropy = self.generate_action(obs, key=key)
    q = frozen(critic).value(obs, action)
    return -(q + jax.lax.stop_gradient(alpha) * entropy).mean()

  def create_optimizer(self) -> dict:
    return {"actor": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}
