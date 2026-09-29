import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Actor, Buffer, Critic, frozen
from barl_jax.config.schemas import ComponentsConfig, ModelConfig, register_model
from barl_jax.critics.categorical import C51CriticConfig
from barl_jax.networks import gaussian
from barl_jax.networks.base import NetworkConfig
from barl_jax.networks.diffusion import TimeConditionedField, controlled_sde
from barl_jax.networks.regulator import AutotuneRegulatorConfig
from barl_jax.networks.simba import SimbaConfig
from barl_jax.networks.time_embedding import FourierTimeConfig, TimeEmbeddingConfig
from barl_jax.optimization import ActorOptim, ActorOptimConfig
from barl_jax.policies.rl import MaxEntPolicyConfig, MaxEntRLPolicy
from barl_jax.registration import get_component, register_component, register_policy


@dataclass(frozen=True)
class DISActorConfig(ActorOptimConfig):
  class_name: str = "DISActor"
  latent_dim: int = 128
  layers: int = 3
  mlp: NetworkConfig = field(default_factory=lambda: SimbaConfig(activation="gelu", norm="none", dtype="float32"))
  time: TimeEmbeddingConfig = field(default_factory=FourierTimeConfig)
  
  squash_cls: str = "TanhSquash"
  adam_b1: float = 0.5
  num_steps: int = 16
  prior_std: float = 2.5
  out_clip: float = 10000.0
  dt_init: float = 0.1
  dt_min: float = 0.001
  dt_s: float = 0.008
  dt_pow: float = 2.0
  friction_init: float = 1.0
  per_dim_friction: bool = True
  learn_dt: bool = True
  learn_friction: bool = True

def _inv_softplus(x: float) -> float:
  """dime_am's `inverse_softplus`, so a raw parameter initialized with this
  passes through softplus back to the value the config states."""
  return float(x) if x > 20.0 else float(math.log(math.expm1(x)))


@register_component
class DISActor(ActorOptim, Actor):
  """The time-reversed diffusion sampler (DIS) DIME actually uses --
  `dime_am/diffusion/od/dis.py` plus `od_integrators.py`.

  Not a denoiser, and not a reparameterization of one. The chain starts at a
  sample from a **fixed** Gaussian prior and is integrated *forward* under a
  learned control, accumulating the log-ratio between the controlled forward
  kernel and an uncontrolled backward kernel. That log-ratio is the path's KL,
  which is what DIME's entropy term is -- so here the entropy falls out of the
  integration rather than being read off a closed-form density, and the step
  size and diffusion coefficient are themselves learned.

  One Euler-Maruyama step at index i, with `f` the control net:

      dt    = softplus(dt_raw) * cosine_schedule(i)
      s2    = 1 / softplus(friction_raw)          # per action dim
      eta   = dt * s2 ;  scale = sqrt(2 * eta)
      drift = grad log p_prior(x) = -x / prior_std^2
      x'    ~ N(x + eta * (drift + f(i, x, obs)), scale^2)
      log_w += log N(x | x' + eta * grad log p_prior(x'), scale)
             - log N(x' | x  + eta * (drift + f), scale)

  `generate` returns a raw, unsquashed terminal state, so `aux` stops one term
  short of the reference's `running_cost - terminal_cost`: `squash_correction`
  supplies the log-det, and adding it inside the integration would count it
  twice."""

  kind = "actor"
  config_type = DISActorConfig

  def __init__(self, env_stats, config: DISActorConfig, *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, config, rngs=rngs, **kwargs)

    # Both are stored through softplus, the reference's parameterization, so the
    # positivity a step size and a friction need is structural rather than a
    # clamp. `learn_*` picks Param vs Buffer -- the reference spells the same
    # choice as a stop_gradient on the read.
    hold = (lambda v: nnx.Param(v)) if self.config.learn_dt else (lambda v: Buffer(v))
    self.dt_raw = hold(jnp.full((1,), _inv_softplus(self.config.dt_init), jnp.float32))
    hold = (lambda v: nnx.Param(v)) if self.config.learn_friction else (lambda v: Buffer(v))
    friction_shape = (self.plan_dim,) if self.config.per_dim_friction else (1,)
    self.friction_raw = hold(jnp.full(friction_shape, _inv_softplus(self.config.friction_init), jnp.float32))

    # dime_am's `get_cosine_schedule`, tabulated once: `per_step_dt` is False
    # there, so the learned dt is a single scalar the schedule shapes.
    i = jnp.arange(self.config.num_steps, dtype=jnp.float32)
    t = (self.config.num_steps - i) / self.config.num_steps
    offset = 1.0 + self.config.dt_s
    self.dt_scale = Buffer((1.0 - self.config.dt_min) * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** self.config.dt_pow + self.config.dt_min)

    latent_dim, layers = config.latent_dim, config.layers

    self.field = TimeConditionedField(
      self.obs_dim, self.plan_dim, latent_dim=latent_dim, layers=layers,
      time=config.time, mlp=config.mlp, rngs=rngs)

    self.field.zero_init_head()

    self.squash = get_component(self.config.squash_cls)(env_stats, None)

  # ---- the process ---------------------------------------------------------

  def step_coef(self, i: int):
    """(eta, scale) at step `i`; both broadcast over the action dimension."""
    dt    = jax.nn.softplus(self.dt_raw[...]) * self.dt_scale[...][i]
    eta   = dt / jax.nn.softplus(self.friction_raw[...])
    return eta, jnp.sqrt(2.0 * eta)

  def prior_score(self, x):
    """grad_x log N(x; 0, prior_std^2 I), the reference's uncontrolled drift."""
    return -x / self.config.prior_std**2

  def control_at(self, obs, x, i: int):
    t = jnp.full((x.shape[0], 1), float(i), jnp.float32)
    u = self.field(obs, x, t).astype(jnp.float32)
    return jnp.clip(u, -self.config.out_clip, self.config.out_clip)

  def generate_action(self, obs, *, critic: Critic = None, key=None, squash=True):
    """`critic` is unused -- DIME's actor needs none to sample -- and accepted
    only to satisfy the shared `Actor.generate_action` contract."""
    key = self.next_key(key)
    key, prior_key = jax.random.split(key)

    x = self.config.prior_std * jax.random.normal(prior_key, (obs.shape[0], self.plan_dim), jnp.float32)

    def euler_step(x, i, step_key):
      eta, scale = self.step_coef(i)
      fwd_mean = x + eta * (self.prior_score(x) + self.control_at(obs, x, i))
      x_new    = fwd_mean + scale * jax.random.normal(step_key, x.shape, jnp.float32)

      bwd_mean = x_new + eta * self.prior_score(x_new)
      # Per-chunk-position log-ratio, (B, chunk_size): the diagonal kernel's
      # log-density is a sum of independent per-dimension terms, so summing
      # only within each position's action_dim block (chunk_sum) rather than
      # over the whole plan_dim is the same total at chunk_size 1 and a
      # genuine decomposition at chunk_size > 1.
      return x_new, self.chunk_sum(gaussian.log_prob(x, bwd_mean, scale)) \
                    - self.chunk_sum(gaussian.log_prob(x_new, fwd_mean, scale))

    X, entropy =  controlled_sde(euler_step, x, self.config.num_steps, key,
                          aux=-self.chunk_sum(gaussian.log_prob(x, 0.0, self.config.prior_std)))

    if squash:
      entropy = entropy + self.chunk_sum(self.squash.log_abs_det_jacobian(X))
      X = self.squash(X)

    return X.reshape(obs.shape[0], self.chunk_size, self.action_dim), entropy

  def derived(self) -> dict:
    return {"dt_0": float(jax.nn.softplus(self.dt_raw[...])[0] * self.dt_scale[...][0]),
            "dt_last": float(jax.nn.softplus(self.dt_raw[...])[0] * self.dt_scale[...][-1]),
            "prior_var": self.config.prior_std**2}

  def loss(self, obs, critic: "Critic", alpha, *, key=None):
    """This actor's own training objective at `obs`, priced against a frozen
    `critic` -- a dict with one scalar per optimizer key it trains. Every key
    must be one `create_optimizer()` also declares.

    The default is maximum-entropy Q ascent through the reparameterized sample:
    `alpha` is a scalar or `(chunk_size,)` and broadcasts against the per-position
    `(B, chunk_size)` entropy, whose chunk sum is the `(B, 1)` bonus."""
    action, entropy = self.generate_action(obs, key=key)
    q = frozen(critic).value(obs, action.reshape(action.shape[0], -1))
    bonus = (jax.lax.stop_gradient(alpha) * entropy).sum(axis=-1, keepdims=True)
    return {"actor": -(q + bonus).mean()}


@register_policy
class DIME(MaxEntRLPolicy):
  """Binds DISActor into the shared max-entropy actor-critic template -- the
  actor-policy integration for this algorithm."""

  actor_cls = DISActor


@register_model
@dataclass(frozen=True)
class DIMEModelConfig(ModelConfig):
  policy_class: str = "DIME"
  params: MaxEntPolicyConfig = field(default_factory=lambda: MaxEntPolicyConfig(
    entropy_scale=-4.0, policy_delay=3))
  components: ComponentsConfig = field(default_factory=lambda: ComponentsConfig(
    actor=DISActorConfig(), critic=C51CriticConfig(),
    entropy=AutotuneRegulatorConfig()))
