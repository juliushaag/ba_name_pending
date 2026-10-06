import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Actor, Buffer, Critic, frozen
from barl_jax.config.schemas import ComponentsConfig, ModelConfig, register_model
from barl_jax.critics.categorical import C51CriticConfig
from barl_jax.networks.base import NetworkConfig
from barl_jax.networks.diffusion import TimeConditionedField, controlled_sde
from barl_jax.networks.regulator import AutotuneRegulatorConfig
from barl_jax.networks.simba import AdaLNSimbaConfig, SimbaConfig
from barl_jax.networks.time_embedding import FourierTimeConfig, TimeEmbeddingConfig
from barl_jax.optimization import ActorOptim, ActorOptimConfig
from barl_jax.policies.rl import MaxEntPolicyConfig, MaxEntRLPolicy
from barl_jax.registration import get_component, register_component, register_policy


@dataclass(frozen=True)
class ScoreContinuousActorConfig(ActorOptimConfig):
  class_name: str = "ScoreContinuousActor"
  latent_dim: int = 128
  layers: int = 3
  mlp: NetworkConfig = field(default_factory=lambda: SimbaConfig(
    activation="silu", norm="layer", norm_kwargs={"epsilon": 1e-5}, dtype="bfloat16"))
  time: TimeEmbeddingConfig = field(default_factory=FourierTimeConfig)
  squash_cls: str = "ErfSquash"
  sigma_min: float = 0.01
  sigma_max: float = 2.0
  tau_min: float = 0.0
  tau_max: float = 1.0
  max_steps: int = 16
  out_clip: float = 10000.0
  q_score_max_norm: float = 5.0
  q_score_max_percentile: float = 95.0
  batch_rep: int = 3

  def problems(self):
    problems = super().problems()
    if not (0 < self.sigma_min < self.sigma_max < math.inf):
      problems.append("sigma bounds require 0 < sigma_min < sigma_max < infinity")
    if not (0 <= self.tau_min < self.tau_max <= 1):
      problems.append("time bounds require 0 <= tau_min < tau_max <= 1")
    if self.max_steps < 1 or self.batch_rep < 1:
      problems.append("max_steps and batch_rep must be >= 1")
    return problems


def clip_score(score, max_norm: float, percentile: float):
  norms      = jnp.linalg.norm(score, axis=-1, keepdims=True)
  threshold  = jnp.minimum(jnp.percentile(norms, percentile), max_norm)
  scale      = jnp.where(norms > threshold, threshold / (norms + 1e-6), 1.0)
  return score * scale, jnp.mean(scale < 1.0)


@register_component
class ScoreContinuousActor(ActorOptim, Actor):
  """AMDP's controlled-SDE actor. The net uses GELU in float32 (the chain accumulates
  a log-weight) and the ctrl time coder by default. MLP settings are grouped under
  `mlp`.
  Only the SDE and score-clipping knobs are this class's own."""

  kind = "actor"
  config_type = ScoreContinuousActorConfig

  def __init__(self, env_stats, config: ScoreContinuousActorConfig, *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, config, rngs=rngs, **kwargs)
    problems = self.config_problems(config)
    if problems:
      raise ValueError(f"{type(self).__name__}: " + "; ".join(problems))

    self._log_r    = math.log(self.config.sigma_max / self.config.sigma_min)

    latent_dim, layers = config.latent_dim, config.layers

    self.field = TimeConditionedField(
      self.obs_dim, self.plan_dim, latent_dim=latent_dim, layers=layers,
      time=config.time, mlp=config.mlp, rngs=rngs)

    self.field.zero_init_head()   # u ~ 0 at the beginning

    taus   = jnp.linspace(self.config.tau_min, self.config.tau_max, self.config.max_steps + 1, dtype=jnp.float32)
    self.taus     = Buffer(taus)
    self.step_var = Buffer(self._variance_between(taus[:-1], taus[1:]))

    self._terminal_var = float(self.sigma_sq_int_fn(jnp.asarray(config.tau_max, jnp.float32)))

    self.squash = get_component(self.config.squash_cls)(
      env_stats, None, erf_k=(2 * self._terminal_var) ** -0.5)

  def sigma_fn(self, tau):
    return self.config.sigma_max * jnp.exp(-self._log_r * tau) * math.sqrt(2.0 * self._log_r)

  def _variance_between(self, start, end):
    # Integrate directly over the interval: subtracting cumulative variances
    # loses the late steps when the cumulative value has rounded to sigma_max².
    span = -2.0 * self._log_r * (end - start)
    # expm1 keeps short intervals accurate; exp keeps long-interval time
    # derivatives accurate (JAX differentiates expm1 using expm1(x) + 1).
    mass = jnp.where(span > -math.log(2), -jnp.expm1(span), 1.0 - jnp.exp(span))
    return (self.config.sigma_max ** 2 * jnp.exp(-2.0 * self._log_r * start)
            * mass)

  def sigma_sq_int_fn(self, tau):
    """int_{tau_min}^{tau} sigma^2(t) dt -- matches what the sampler injects."""
    return self._variance_between(self.config.tau_min, tau)

  def bridge_sample(self, endpoint, tau, noise):
    sigma_sq = self.sigma_sq_int_fn(tau)
    ratio    = sigma_sq / self._terminal_var
    std      = jnp.sqrt(jnp.maximum(sigma_sq * (1.0 - ratio), 0.0))
    return ratio * endpoint + std * noise

  @property
  def terminal_var(self) -> float:
    """Var(X_tau_max) at zero control. erf_k MUST be derived from this."""
    return self._terminal_var

  def squash_correction(self, X):
    return self.chunk_sum(self.squash.log_abs_det_jacobian(X))

  def derived(self) -> dict:
    return self.squash.derived()

  def prior_surprisal(self, X):
    const = 0.5 * self.action_dim * math.log(2 * math.pi * self._terminal_var)
    return const + 0.5 * self.chunk_sum(X ** 2) / self._terminal_var

  def control(self, obs, x, tau):
    u = self.field(obs, x, self.scaled_time(tau))
    return jnp.clip(u, -self.config.out_clip, self.config.out_clip)

  def scaled_time(self, tau):
    """The level the control net conditions on. The reference scales it into
    [-1, 1] over the full [0, 1] diffusion horizon before its Fourier coder
    (ctrl_net.py). A truncated tau interval keeps its absolute time coordinates."""
    return 2.0 * tau - 1.0

  def euler_step(self, obs, X, i, key):
    """The same controlled transition for sampling and adjoint training."""
    tau = self.taus[...][i]
    u_tilde = self.control(obs, X, tau)
    v = self.step_var[...][i]
    sqrt_v = jnp.sqrt(v)
    eps = jax.random.normal(key, u_tilde.shape, u_tilde.dtype)
    log_w = (-0.5 * v * self.chunk_sum(u_tilde ** 2)
             - sqrt_v * self.chunk_sum(eps * u_tilde))
    return X + v * u_tilde + sqrt_v * eps, log_w

  def generate_action(self, obs, *, critic: Critic = None, squash=True, key=None):
    """`critic` is unused -- AMDP's actor needs none to sample -- and accepted
    only to satisfy the shared `Actor.generate_action` contract."""
    B = obs.shape[0]
    key = self.next_key(key)

    X = jnp.zeros((B, self.plan_dim), jnp.float32)
    X, log_w = controlled_sde(lambda x, i, k: self.euler_step(obs, x, i, k),
                              X, self.config.max_steps, key,
                              aux=jnp.zeros((B, self.chunk_size), jnp.float32))

    log_w = log_w + self.prior_surprisal(X)

    if squash:
      log_w = log_w + self.squash_correction(X)
      X = self.squash(X)

    return X.reshape(B, self.chunk_size, self.action_dim), log_w

  def loss(self, obs, critic: Critic, alpha, *, key=None):
    """Adjoint matching (Adjoint Matching, arXiv:2409.08861, Sec. 5.2)"""

    key = self.next_key(key)
    rollout_key, tau_key, noise_key = jax.random.split(key, 3)
    alpha = jax.lax.stop_gradient(jnp.asarray(alpha))


    X, _ = self.generate_action(obs, key=rollout_key, squash=False)
    X = jax.lax.stop_gradient(X).reshape(X.shape[0], -1)

    def q_values(critic, x):
      return critic.value(obs, self.squash(x)).sum()

    x_score = nnx.grad(q_values, argnums=1)(frozen(critic), X)
    x_score, clipped_frac = clip_score(x_score, self.config.q_score_max_norm, self.config.q_score_max_percentile)

    def terminal_entropy(actor, x):
      return (alpha * (actor.prior_surprisal(x) + actor.squash_correction(x))).sum()

    entropy_score = nnx.grad(terminal_entropy, argnums=1)(frozen(self), X)

    score = jnp.repeat(x_score + entropy_score, self.config.batch_rep, axis=0)
    obs_r = jnp.repeat(obs, self.config.batch_rep, axis=0)
    X_r   = jnp.repeat(X, self.config.batch_rep, axis=0)

    tau   = jax.random.uniform(tau_key, (X_r.shape[0], 1), minval=self.config.tau_min, maxval=self.config.tau_max)
    noise = jax.random.normal(noise_key, X_r.shape)
    a_t   = self.bridge_sample(X_r, tau, noise)
    u_tilde   = self.control(obs_r, a_t, tau)
    sigma_tau = self.sigma_fn(tau)

    u      = sigma_tau * u_tilde
    target = jax.lax.stop_gradient(sigma_tau * (score / self.chunk_broadcast(alpha)))
    loss   = (0.5 * ((u - target) ** 2).sum(axis=-1) / sigma_tau.squeeze(-1)).mean()

    self.metrics.scalar("score_loss", loss)
    self.metrics.dist("score_target", target)
    self.metrics.vec("u", u)
    self.metrics.scalar("score_clipped_frac", clipped_frac)
    critic.metrics.vec("score", x_score)

    return { "actor" : loss }


@register_policy
class AMDP(MaxEntRLPolicy):
  """Binds ScoreContinuousActor into the shared max-entropy actor-critic
  template -- the actor-policy integration for this algorithm."""

  actor_cls = ScoreContinuousActor


@register_model
@dataclass(frozen=True)
class AMDPModelConfig(ModelConfig):
  policy_class: str = "AMDP"
  params: MaxEntPolicyConfig = field(default_factory=lambda: MaxEntPolicyConfig(
    normalize_obs=False, policy_delay=3))
  components: ComponentsConfig = field(default_factory=lambda: ComponentsConfig(
    actor=ScoreContinuousActorConfig(), critic=C51CriticConfig(),
    entropy=AutotuneRegulatorConfig()))
