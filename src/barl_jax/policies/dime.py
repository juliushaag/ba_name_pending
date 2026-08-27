import math

import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks import gaussian
from barl_jax.networks.critic import QCriticBase
from barl_jax.networks.diffusion import TimeConditionedField, controlled_sde
from barl_jax.networks.optim import adam
from barl_jax.networks.registry import get_component, register_component
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.policy import (Actor, Buffer, Critic, Policy, frozen, gated_loss,
                             register_policy)


def _inv_softplus(x: float) -> float:
  """dime_am's `inverse_softplus`, so a raw parameter initialized with this
  passes through softplus back to the value the config states."""
  return float(x) if x > 20.0 else float(math.log(math.expm1(x)))


@register_component
class DISActor(Actor):
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

  def __init__(self, env_stats, *, latent_dim: int, layers: int, num_steps: int,
               squash_cls: str, prior_std: float,
               dt_init: float, dt_min: float, dt_s: float, dt_pow: float,
               friction_init: float, per_dim_friction: bool,
               learn_dt: bool, learn_friction: bool, out_clip: float,
               time_dim: int, time_embed: str, time_kwargs: dict,
               lr: float, adam_b1: float, grad_clip: float, grad_clip_mode: str,
               activation: str, norm: str,
               norm_input: bool, norm_stem: bool, norm_after_act: bool,
               dropout: float, final_activation: bool,
               residual: bool, norm_kwargs: dict, dtype: str,
               rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, **kwargs)

    self.num_steps = int(num_steps)
    self.time_dim  = int(time_dim)
    self.prior_std = float(prior_std)
    self.out_clip  = float(out_clip)
    self.lr        = float(lr)
    self.adam_b1   = float(adam_b1)
    self.grad_clip, self.grad_clip_mode = float(grad_clip), str(grad_clip_mode)

    # Both are stored through softplus, the reference's parameterization, so the
    # positivity a step size and a friction need is structural rather than a
    # clamp. `learn_*` picks Param vs Buffer -- the reference spells the same
    # choice as a stop_gradient on the read.
    hold = (lambda v: nnx.Param(v)) if learn_dt else (lambda v: Buffer(v))
    self.dt_raw = hold(jnp.full((1,), _inv_softplus(dt_init), jnp.float32))
    hold = (lambda v: nnx.Param(v)) if learn_friction else (lambda v: Buffer(v))
    friction_shape = (self.action_dim,) if per_dim_friction else (1,)
    self.friction_raw = hold(jnp.full(friction_shape, _inv_softplus(friction_init), jnp.float32))

    # dime_am's `get_cosine_schedule`, tabulated once: `per_step_dt` is False
    # there, so the learned dt is a single scalar the schedule shapes.
    i = jnp.arange(self.num_steps, dtype=jnp.float32)
    t = (self.num_steps - i) / self.num_steps
    offset = 1.0 + dt_s
    self.dt_scale = Buffer((1.0 - dt_min) * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** dt_pow + dt_min)

    self.field = TimeConditionedField(
      self.obs_dim, self.action_dim, latent_dim=latent_dim, layers=layers,
      time_dim=self.time_dim, time_embed=time_embed, time_kwargs=time_kwargs,
      activation=activation, norm=norm, norm_input=norm_input, norm_stem=norm_stem,
      norm_after_act=norm_after_act, dropout=dropout, final_activation=final_activation,
      residual=residual, norm_kwargs=norm_kwargs, dtype=dtype, rngs=rngs)

    self.field.zero_init_head()

    self.squash = get_component(squash_cls)(env_stats)
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

  def generate(self, obs, key=None, squash=True):

    key = self.rngs.noise() if key is None else key
    key, prior_key = jax.random.split(key)

    x = self.prior_std * jax.random.normal(prior_key, (obs.shape[0], self.action_dim), jnp.float32)

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
    
    return X, entropy
  
  def actor_loss(self, obs, critic: Critic, alpha, key=None):
    """Returns the loss rather than writing it into a `losses` dict, because the
    owning Policy runs it under `gated_loss` -- a branch has to hand its value
    back, it cannot mutate a dict on one side and not the other."""
    action, entropy = self.generate(obs, key=key)

    q = frozen(critic).value(obs, action)
    return -(q + jax.lax.stop_gradient(alpha) * entropy).mean()

  def sample(self, s, key=None, critic: Critic = None):
    return self.generate(s, key=key)[0]

  def create_optimizer(self) -> dict:
    return {"actor": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}

  def derived(self) -> dict:
    return {"dt_0": float(jax.nn.softplus(self.dt_raw[...])[0] * self.dt_scale[...][0]),
            "dt_last": float(jax.nn.softplus(self.dt_raw[...])[0] * self.dt_scale[...][-1]),
            "prior_var": self.prior_std**2}


@register_policy
class DIME(Policy):
  """Diffusion-policy MaxEnt actor-critic over a path-entropy sampler's chain,
  with an entropy dual auto-tuned against a target entropy.

  The actor loss is differentiated through the *whole* sampling chain -- 16
  network evaluations deep with a critic on top -- so it stores 16 sets of
  activations, which is what bounds the batch it fits in rather than its
  runtime: at batch 256 the chain adds ~130ms to a ~190ms step, where SDAC's
  best-of-32 sampler runs the same kind of chain over 64x the rows."""

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    params = self.resolve_params()
    self.entropy_scale = params["entropy_scale"]
    self.policy_delay           = params["policy_delay"]

    self.critic: QCriticBase = self.build_component("critic", QCriticBase)
    self.actor: DISActor = self.build_component("actor", DISActor)
    self.entropy: AutotuneRegulator = self.build_component(
      "entropy", AutotuneRegulator, target=self.action_dim * self.entropy_scale)

  def optimizer_cadence(self) -> dict:
    return {"actor": self.policy_delay, "entropy_alpha": self.policy_delay}

  def train_step(self, batch):
    obs      = self.normalize_obs(batch["obs"])
    next_obs = self.normalize_obs(batch["next_obs"])
    action   = batch["act"]
    reward   = batch["rew"]
    term     = batch["term"]

    alpha  = self.entropy.value()
    losses = {}

    # Every draw this step descends from one key, split here: the bootstrap
    # chain and the differentiated chain must not share a stream, and a step's
    # randomness should be readable off train_step rather than hidden in each
    # component's `rngs`.
    keys = jax.random.split(self.rngs.noise(), 2)

    # ---- critic: the bootstrap sample is a target, not a path to train through
    next_action, entropy = jax.lax.stop_gradient(
      self.actor.generate(next_obs, key=keys[0]))

    entropy_bonus = jax.lax.stop_gradient(alpha * entropy)
    self.critic.td_loss(obs, action, next_obs, next_action, reward, term,
                        losses, shift=entropy_bonus)

    # ---- actor: backprop through the reverse chain into the diffusion net.
    # Skipped outright on the two steps in three `policy_delay` would mask the
    # update on anyway -- the loss carries no gradient into the critic (it prices
    # through `frozen`), so this is parameter-identical and drops the chain's
    # 16 forward + 16 backward passes, the dominant term in the step.
    losses["actor"] = gated_loss(
      self, self.policy_delay,
      lambda actor, critic: actor.actor_loss(obs, critic, alpha, key=keys[1]),
      self.actor, self.critic)

    self.entropy.loss(entropy, losses)

    self.actor.metrics.dist("entropy", entropy)
    self.actor.metrics.vec("action", next_action, limit=1.0)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_obs(observation)
    return self.actor.sample(obs, key=key), {}
