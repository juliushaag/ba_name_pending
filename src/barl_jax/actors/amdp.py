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
from barl_jax.networks.mlp import MLPConfig
from barl_jax.networks.regulator import AutotuneRegulatorConfig
from barl_jax.networks.time_embedding import FourierTimeConfig, TimeEmbeddingConfig
from barl_jax.optimization import ActorOptim, ActorOptimConfig
from barl_jax.policies.rl import MaxEntPolicyConfig, MaxEntRLPolicy
from barl_jax.registration import get_component, register_component, register_policy


@dataclass(frozen=True)
class ScoreContinuousActorConfig(ActorOptimConfig):
  class_name: str = "ScoreContinuousActor"
  latent_dim: int = 512
  layers: int = 3
  mlp: NetworkConfig = field(default_factory=lambda: MLPConfig(activation="gelu", norm="none", dtype="float32"))
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
  batch_rep: int = 1


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

    self._log_r    = math.log(self.config.sigma_max / self.config.sigma_min)
    self._int_base = float(self._raw_int(jnp.asarray(self.config.tau_min, jnp.float32)))

    latent_dim, layers = config.latent_dim, config.layers

    self.field = TimeConditionedField(
      self.obs_dim, self.plan_dim, latent_dim=latent_dim, layers=layers,
      time=config.time, mlp=config.mlp, rngs=rngs)

    self.field.zero_init_head()   # u ~ 0 at the beginning

    taus   = jnp.linspace(self.config.tau_min, self.config.tau_max, self.config.max_steps + 1, dtype=jnp.float32)
    sig_sq = self.sigma_sq_int_fn(taus)                        # exact, rebased to tau_min
    self.taus     = Buffer(taus)
    self.step_var = Buffer(jnp.maximum(sig_sq[1:] - sig_sq[:-1], 0.0))

    self._terminal_var = float(self.step_var[...].sum())

    self.squash = get_component(self.config.squash_cls)(
      env_stats, None, erf_k=(2 * self._terminal_var) ** -0.5)

  def sigma_fn(self, tau):
    r = self.config.sigma_max / self.config.sigma_min
    return self.config.sigma_min * r ** (1.0 - tau) * math.sqrt(2.0 * self._log_r)

  def _raw_int(self, tau):
    r = self.config.sigma_max / self.config.sigma_min
    return self.config.sigma_max ** 2 * (1.0 - r ** (-2.0 * tau))

  def sigma_sq_int_fn(self, tau):
    """int_{tau_min}^{tau} sigma^2(t) dt -- matches what the sampler injects."""
    return self._raw_int(tau) - self._int_base

  def bridge_sample(self, endpoint, tau, noise):
    sigma_sq = self.sigma_sq_int_fn(tau)
    ratio    = sigma_sq / self._terminal_var
    std      = jnp.sqrt(jnp.maximum(sigma_sq * (1.0 - ratio), 0.0))
    return ratio * endpoint + std * noise

  @property
  def terminal_var(self) -> float:
    """Var(X_1) at zero control. erf_k MUST be derived from this."""
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
    [-1, 1] before its fourier coder (ctrl_net.py), so the fixed frequency ladder
    reads the same span whatever tau_min/tau_max say."""
    return 2.0 * tau - 1.0

  def euler_step(self, obs, X, i, key):
    """The same controlled transition for sampling and adjoint training."""
    tau = jnp.broadcast_to(self.taus[...][i], (obs.shape[0], 1))
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
