import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Actor, Buffer, frozen
from barl_jax.config.schemas import ComponentsConfig, ModelConfig, register_model
from barl_jax.critics.categorical import C51CriticConfig
from barl_jax.networks.base import NetworkConfig
from barl_jax.networks.diffusion import TimeConditionedField, reverse_chain
from barl_jax.networks.simba import AdaLNSimbaConfig
from barl_jax.networks.time_embedding import FourierTimeConfig, TimeEmbeddingConfig
from barl_jax.optimization import ActorOptim, ActorOptimConfig
from barl_jax.policies.rl import PolicyConfig, RLPolicy
from barl_jax.registration import register_component, register_policy


@dataclass(frozen=True)
class DDPMActorConfig(ActorOptimConfig):
  class_name: str = "DDPMActor"
  latent_dim: int = 128
  layers: int = 3
  mlp: NetworkConfig = field(default_factory=lambda: AdaLNSimbaConfig(
    activation="mish", dtype="float32"))
  time: TimeEmbeddingConfig = field(default_factory=lambda: FourierTimeConfig(
    time_dim=64, num_freqs=32, hidden=64, concat_input=False, learn_freqs=True))
  num_steps: int = 20
  t_min: float = 1.0e-3
  t_max: float = 0.9946
  cosine_s: float = 0.008
  clip_sampler: bool = True
  num_candidates: int = 500
  temp: float = 0.25
  grad_clip_mode: str = "none"

  def problems(self):
    problems = super().problems()
    if self.num_steps < 1:
      problems.append("num_steps must be >= 1")
    if not (0 < self.t_min < self.t_max < 1):
      problems.append("time bounds require 0 < t_min < t_max < 1")
    if not (0 <= self.cosine_s < math.inf):
      problems.append("cosine_s must be finite and >= 0")
    return problems


@register_component
class DDPMActor(ActorOptim, Actor):
  """Continuous-time VP-cosine DDPM actor trained by reverse-sampling score
  matching (SDAC, Ma et al. 2025), following flowrl's `SDACAgent`.

  The clipped x0 is the action -- there is no squash -- so candidates, buffer
  actions and samples all live in the same [-1, 1] box."""

  kind = "actor"
  config_type = DDPMActorConfig

  def __init__(self, env_stats, config: DDPMActorConfig, *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, config, rngs=rngs, **kwargs)
    problems = self.config_problems(config)
    if problems:
      raise ValueError(f"{type(self).__name__}: " + "; ".join(problems))
    self.field = TimeConditionedField(
      self.obs_dim, self.plan_dim, latent_dim=config.latent_dim, layers=config.layers,
      time=config.time, mlp=config.mlp, rngs=rngs)
    self.t_grid = Buffer(jnp.linspace(config.t_min, config.t_max, config.num_steps + 1, dtype=jnp.float32))

  # ---- the noise process ---------------------------------------------------
  def alpha_sigma(self, t):
    """Cosine VP schedule: x_t = alpha(t) x0 + sigma(t) eps, alpha^2 + sigma^2 = 1."""
    s = self.config.cosine_s
    offset = math.pi / 2 * s / (1 + s)
    angle = jnp.pi / 2 * jnp.clip(t, 0.0, self.config.t_max) / (1 + s)
    alpha = jnp.cos(offset + angle) / math.cos(offset)
    # cos²(offset) - cos²(offset + angle) = sin(angle) sin(2 offset + angle).
    # This avoids rounding sigma to zero when a positive t is close to zero.
    sigma_sq = jnp.sin(angle) * jnp.sin(2 * offset + angle) / math.cos(offset) ** 2
    return alpha, jnp.sqrt(jnp.maximum(sigma_sq, 0.0))

  def x0_from_eps(self, x_t, eps, t):
    alpha, sigma = self.alpha_sigma(t)
    return (x_t - sigma * eps) / alpha

  def predict_eps(self, obs, x, t):
    return self.field(obs, x, t).astype(jnp.float32)

  # ---- the reverse chain ---------------------------------------------------
  def reverse_step(self, obs, x, n, key):
    t_n, t_prev = self.t_grid[...][n], self.t_grid[...][n - 1]
    alpha_n, sigma_n = self.alpha_sigma(t_n)
    alpha_prev, sigma_prev = self.alpha_sigma(t_prev)
    a_n = (alpha_n / alpha_prev) ** 2
    beta_n = (sigma_n ** 2 - sigma_prev ** 2) / alpha_prev ** 2

    x0 = self.x0_from_eps(x, self.predict_eps(obs, x, t_n), t_n)
    if self.config.clip_sampler:
      x0 = jnp.clip(x0, -1.0, 1.0)
    mean = (alpha_prev * beta_n * x0 + jnp.sqrt(a_n) * sigma_prev ** 2 * x) / sigma_n ** 2
    noise = jnp.sqrt(beta_n) * jax.random.normal(key, x.shape, x.dtype)
    return mean + jnp.where(n > 1, noise, jnp.zeros_like(noise))

  def generate_action(self, obs, *, critic=None, key=None):
    key = self.next_key(key)
    key, prior_key = jax.random.split(key)
    x = jax.random.normal(prior_key, (obs.shape[0], self.plan_dim), jnp.float32)
    x = reverse_chain(lambda x, n, k: self.reverse_step(obs, x, n, k), x, self.config.num_steps, key)
    return self.to_plan(jnp.clip(x, -1.0, 1.0)), None

  # ---- reverse-sampling score matching ---------------------------------------
  def loss(self, obs, critic, alpha, *, key=None, action=None):
    """Diffuse `action` (the replay action; a fresh sample when None) to a random
    t, draw K eps whose x0 reconstructions stay in [-1, 1], and regress the eps
    prediction onto them weighted by softmax(Q(x0_k) / (temp * q_scale)) over K,
    where `q_scale` is the cross-candidate Q std averaged over this batch's
    states (one scalar per batch, not per state) -- not a
    running, cross-*state* statistic (e.g. a critic-wide value scale): on
    DogRun the Q range across different candidate actions at one state is
    ~15-30x smaller than the Q range across different states (measured on a
    trained checkpoint: cross-candidate std ~0.01-0.06 vs. cross-state std
    ~0.9), so dividing by anything but the local, per-example spread leaves
    the softmax within numerical noise of uniform regardless of `temp` --
    the actor degenerates into plain behavior cloning of `action` and the
    critic's ranking of candidates never reaches the gradient. `alpha` is
    unused -- SDAC has no entropy dual (`RLPolicy`) -- and stays a parameter
    only because every actor's `loss()` shares this call signature (see
    `RLPolicy.actor_objective`)."""
    keys = jax.random.split(self.next_key(key), 5)
    if action is None:
      action = jax.lax.stop_gradient(self.generate_action(obs, key=keys[4])[0])
    B, K = obs.shape[0], self.config.num_candidates
    a0 = action.reshape(B, -1)

    t = jax.random.uniform(keys[0], (B, 1), minval=self.config.t_min, maxval=self.config.t_max)
    alpha_t, sigma_t = self.alpha_sigma(t)
    x_t = alpha_t * a0 + sigma_t * jax.random.normal(keys[1], a0.shape)

    # x0 = (x_t - sigma eps) / alpha lies in [-1, 1] iff eps lies in [lo, hi].
    lo, hi = (x_t - alpha_t) / sigma_t, (x_t + alpha_t) / sigma_t
    eps = jax.random.truncated_normal(keys[2], lo, hi, (K, B, self.plan_dim))
    # truncated_normal returns NaN when [lo, hi] is far in a tail.
    eps = jnp.where(jnp.isnan(eps), jnp.clip(jax.random.normal(keys[3], eps.shape), lo, hi), eps)
    candidates = self.x0_from_eps(x_t[None], eps, t)

    qv = frozen(critic).value(
      jnp.broadcast_to(obs[None], (K, B, obs.shape[-1])).reshape(K * B, -1),
      candidates.reshape(K * B, -1)).reshape(K, B)
    q_scale = jax.lax.stop_gradient(qv.std(axis=0).mean()) + 1.0e-6
    weight = jax.lax.stop_gradient(jax.nn.softmax(qv / (self.config.temp * q_scale), axis=0))

    eps_pred = self.predict_eps(obs, x_t, t)
    loss = (weight[..., None] * (eps_pred[None] - eps) ** 2).mean()

    self.metrics.scalar("q_scale", q_scale)
    self.metrics.dist("weight_ess", 1.0 / (weight ** 2).sum(axis=0))
    self.metrics.dist("weight_max", weight.max(axis=0))
    self.metrics.vec("eps_pred", eps_pred)
    return {"actor": loss}


@dataclass(frozen=True)
class SDACPolicyConfig(PolicyConfig):
  utd_ratio: float = 0.2


@register_policy
class SDAC(RLPolicy):
  """DDPMActor on the shared actor-critic template. No entropy dual: the
  candidate-weighting temperature (`DDPMActor.loss`) is set from the critic's
  own running value scale instead. The actor loss is priced at the replay
  action rather than a fresh sample."""

  actor_cls = DDPMActor

  def actor_objective(self, actor, critic, obs, action, alpha, key):
    return actor.loss(obs, critic, alpha, key=key, action=action)


@register_model
@dataclass(frozen=True)
class SDACModelConfig(ModelConfig):
  policy_class: str = "SDAC"
  params: SDACPolicyConfig = field(default_factory=SDACPolicyConfig)
  components: ComponentsConfig = field(default_factory=lambda: ComponentsConfig(
    actor=DDPMActorConfig(), critic=C51CriticConfig(), entropy=None))
