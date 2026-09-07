import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barl_jax.components import Actor, Buffer, Critic, PathwiseActorMixin
from barl_jax.config.toolkit import ConfigMixin
from barl_jax.hparam import Config
from barl_jax.networks import gaussian
from barl_jax.networks.mlp import MLP, MLP_PRESETS
from barl_jax.optimization import ActorOptim, adam
from barl_jax.registration import get_component, register_component


@register_component
class StochasticActor(PathwiseActorMixin, Actor, ActorOptim):
  """Diagonal-Normal MLP actor shared by SAC/PPO/IQL. `dist()` stays an
  unconstrained Normal; sample()/mode() squash it (the `squash_cls` Config
  selects which -- any no-arg Squash works generically here).

  The shallow Gaussian policy has just a state-independent `log_std` plus the
  std bounds; the net is the `layer` style (a LayerNorm stem over silu) by
  default -- switch it with `mlp_preset` (a configuration value, not
  inheritance).

  `log_std` is a state-independent parameter rather than a network head --
  barl's deliberate deviation from canonical SAC, kept as-is."""

  kind = "actor"
  mlp_preset = Config("layer", style=MLP_PRESETS)
  squash_cls = Config("TanhSquash")
  log_std_min  = Config(-20.0)
  log_std_max  = Config(2.0)
  log_prob_clip = Config(0.99)   # tanh atanh guard in log_prob

  def __init__(self, env_stats, config: ConfigMixin, *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, config, **kwargs)

    self.log_std = nnx.Param(jnp.zeros((self.plan_dim,), jnp.float32))
    self.squash  = get_component(self.squash_cls)(env_stats, ConfigMixin())

    low, high = env_stats.get("act_low"), env_stats.get("act_high")
    scale = 1.0 if low is None else (np.asarray(high) - np.asarray(low)) / 2
    scale = np.tile(np.asarray(scale, dtype=np.float32).reshape(-1),
                    self.chunk_size).reshape(-1)
    self.log_act_scale = Buffer(jnp.log(jnp.asarray(scale, jnp.float32)))

    # latent_dim/layers are tier-ladder-owned (see tier ladders in hparam.py).
    latent_dim, layers = config.get("latent_dim", int), config.get("layers", int)

    self.net = MLP(self.obs_dim, self.plan_dim, latent_dim, layers,
                   activation=self.activation, norm=self.norm, norm_input=self.norm_input,
                         norm_stem=self.norm_stem, norm_after_act=self.norm_after_act,
                   dropout=self.dropout, final_activation=self.final_activation,
                   residual=self.residual, dtype=self.dtype, rngs=rngs, **(self.hparam("norm_kwargs") or {}))
    self.rngs = rngs

  def _plan(self, flat):
    return flat.reshape(flat.shape[0], self.chunk_size, self.action_dim)

  def dist(self, s):
    """(mean, std) of the pre-squash Normal, over the flat plan."""
    mean = self.net(s).astype(jnp.float32)
    std  = jnp.exp(jnp.clip(self.log_std[...], self.log_std_min, self.log_std_max))
    return mean, std

  def log_prob(self, dist, action):
    """`action` is the squashed (canonical [-1, 1]) value; inverts the squash
    and applies the change-of-variables correction."""
    mean, std = dist
    flat = action.reshape(action.shape[0], -1)
    x = self.squash.inverse(jnp.clip(flat, -self.log_prob_clip, self.log_prob_clip))
    lp = gaussian.log_prob(x, mean, std) - self.squash.log_abs_det_jacobian(x, flat)
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
    action_flat, x, (mean, std) = self.rsample(s, key)

    log_prob = (gaussian.log_prob(x, mean, std)
                - self.squash.log_abs_det_jacobian(x, action_flat)
                - self.log_act_scale[...])
    # Per-chunk-position entropy, (B, chunk_size): summing within each
    # position's action_dim block rather than over the whole flattened plan --
    # the same shape (and at chunk_size 1 the same value) as the old (B, 1)
    # joint entropy, so a scalar alpha reads the chunk-summed total unchanged.
    entropy = -self.chunk_sum(log_prob)
    return self._plan(action_flat), entropy

  def sample(self, s, *, critic: Critic = None, key=None):
    action_flat, _, _ = self.rsample(s, key)
    return self._plan(action_flat)

  def mode(self, s):
    """Deterministic (mean) action -- IQL trains via AWR but deploys this."""
    mean, _ = self.dist(s)
    return self._plan(self.squash(mean))

  # loss: PathwiseActorMixin's reparameterized max-entropy objective,
  # -(Q + alpha*H) -- SAC's actor improves against the critic but must not
  # train it (frozen(critic) inside the mixin), same as every actor here.

  def create_optimizer(self) -> dict:
    return {"actor": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}
