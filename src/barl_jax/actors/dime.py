import math

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Actor, Buffer, Critic, frozen
from barl_jax.config.toolkit import ConfigMixin
from barl_jax.hparam import Config, HParam
from barl_jax.networks import gaussian
from barl_jax.networks.diffusion import TimeConditionedField, controlled_sde
from barl_jax.networks.flow import TIME_STYLES
from barl_jax.networks.mlp import MLP_PRESETS
from barl_jax.networks.optim import ActorOptim, adam
from barl_jax.registration import get_component, register_component


def _inv_softplus(x: float) -> float:
  """dime_am's `inverse_softplus`, so a raw parameter initialized with this
  passes through softplus back to the value the config states."""
  return float(x) if x > 20.0 else float(math.log(math.expm1(x)))


@register_component
class DISActor(Actor, ActorOptim):
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
  # fp32: the reference is fp32, and the log-ratio accumulates. Both are Config
  # selectors, not inheritance -- switch the net / time embedding with
  # `mlp_preset` / `time_emb`.
  mlp_preset = Config("gelu_fp32", style=MLP_PRESETS)
  time_emb  = Config("pisgrad", style=TIME_STYLES)
  squash_cls = Config("TanhSquash")
  # dime_am trains the sampler's nets with the critic's b1, not the actor's
  # usual 0.9 -- the one optimizer override this actor carries.
  adam_b1      = Config(0.5)
  num_steps    = Config(16)
  prior_std    = HParam(2.5)   # a std, not a variance -- the reference's sampler.init_std
  out_clip     = Config(10000.0)   # PISGRADNet's `outer_clip`
  dt_init      = HParam(0.1)       # configs/base.yaml `dt`, shaped by the cosine schedule below
  dt_min       = HParam(0.001)     # configs/sampler/dt_schedule/cosine.yaml
  dt_s         = Config(0.008)
  dt_pow       = Config(2.0)
  friction_init = Config(1.0)      # configs/sampler/dis.yaml `friction`
  per_dim_friction = Config(True)
  learn_dt     = Config(True)
  learn_friction = Config(True)

  def __init__(self, env_stats, config: ConfigMixin, *, rngs: nnx.Rngs, **kwargs):
    self.resolve_hparams(config, env_stats=env_stats)

    # Both are stored through softplus, the reference's parameterization, so the
    # positivity a step size and a friction need is structural rather than a
    # clamp. `learn_*` picks Param vs Buffer -- the reference spells the same
    # choice as a stop_gradient on the read.
    hold = (lambda v: nnx.Param(v)) if self.learn_dt else (lambda v: Buffer(v))
    self.dt_raw = hold(jnp.full((1,), _inv_softplus(self.dt_init), jnp.float32))
    hold = (lambda v: nnx.Param(v)) if self.learn_friction else (lambda v: Buffer(v))
    friction_shape = (self.plan_dim,) if self.per_dim_friction else (1,)
    self.friction_raw = hold(jnp.full(friction_shape, _inv_softplus(self.friction_init), jnp.float32))

    # dime_am's `get_cosine_schedule`, tabulated once: `per_step_dt` is False
    # there, so the learned dt is a single scalar the schedule shapes.
    i = jnp.arange(self.num_steps, dtype=jnp.float32)
    t = (self.num_steps - i) / self.num_steps
    offset = 1.0 + self.dt_s
    self.dt_scale = Buffer((1.0 - self.dt_min) * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** self.dt_pow + self.dt_min)

    # latent_dim/layers are tier-ladder-owned.
    latent_dim, layers = config.get("latent_dim", int), config.get("layers", int)

    self.field = TimeConditionedField(
      self.obs_dim, self.plan_dim, latent_dim=latent_dim, layers=layers,
      time_dim=self.time_dim, time_embed=self.time_embed, time_kwargs=self.hparam("time_kwargs"),
      activation=self.activation, norm=self.norm, norm_input=self.norm_input, norm_stem=self.norm_stem,
      norm_after_act=self.norm_after_act, dropout=self.dropout, final_activation=self.final_activation,
      residual=self.residual, norm_kwargs=self.hparam("norm_kwargs"), dtype=self.dtype, rngs=rngs)

    self.field.zero_init_head()

    self.squash = get_component(self.squash_cls)(env_stats, ConfigMixin())
    self.rngs = rngs

  # ---- the process ---------------------------------------------------------

  def step_coef(self, i: int):
    """(eta, scale) at step `i`; both broadcast over the action dimension."""
    dt    = jax.nn.softplus(self.dt_raw[...]) * self.dt_scale[...][i]
    eta   = dt / jax.nn.softplus(self.friction_raw[...])
    return eta, jnp.sqrt(2.0 * eta)

  def prior_score(self, x):
    """grad_x log N(x; 0, prior_std^2 I), the reference's uncontrolled drift."""
    return -x / self.prior_std**2

  def prior_log_prob(self, x):
    return gaussian.diag_log_prob(x, 0.0, self.prior_std)

  def control_at(self, obs, x, i: int):
    t = jnp.full((x.shape[0], 1), float(i), jnp.float32)
    u = self.field(obs, x, t).astype(jnp.float32)
    return jnp.clip(u, -self.out_clip, self.out_clip)

  def generate_action(self, obs, *, critic: Critic = None, key=None, squash=True):
    """`critic` is unused -- DIME's actor needs none to sample -- and accepted
    only to satisfy the shared `Actor.generate_action` contract."""
    key = self.rngs.noise() if key is None else key
    key, prior_key = jax.random.split(key)

    x = self.prior_std * jax.random.normal(prior_key, (obs.shape[0], self.plan_dim), jnp.float32)

    def euler_step(x, i, step_key):
      eta, scale = self.step_coef(i)
      fwd_mean = x + eta * (self.prior_score(x) + self.control_at(obs, x, i))
      x_new    = fwd_mean + scale * jax.random.normal(step_key, x.shape, jnp.float32)

      bwd_mean = x_new + eta * self.prior_score(x_new)
      return x_new, gaussian.diag_log_prob(x, bwd_mean, scale) \
                    - gaussian.diag_log_prob(x_new, fwd_mean, scale)

    X, entropy =  controlled_sde(euler_step, x, self.num_steps, key,
                          aux=-self.prior_log_prob(x))

    if squash:
      entropy = entropy + self.squash.log_abs_det_jacobian(X).sum(axis=-1, keepdims=True)
      X = self.squash(X)

    return X.reshape(obs.shape[0], self.chunk_size, self.action_dim), entropy

  def loss(self, obs, critic: Critic, alpha, *, key=None):
    """DIME's path-KL objective, -(Q + alpha*entropy)."""
    action, entropy = self.generate_action(obs, key=key)

    q = frozen(critic).value(obs, action.reshape(action.shape[0], -1))
    return {"actor": -(q + jax.lax.stop_gradient(alpha) * entropy).mean()}

  def create_optimizer(self) -> dict:
    return {"actor": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}

  def derived(self) -> dict:
    return {"dt_0": float(jax.nn.softplus(self.dt_raw[...])[0] * self.dt_scale[...][0]),
            "dt_last": float(jax.nn.softplus(self.dt_raw[...])[0] * self.dt_scale[...][-1]),
            "prior_var": self.prior_std**2}
