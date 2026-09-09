import math

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.actors.policy_dict import maxent_actor_policy
from barl_jax.components import Actor, Buffer, Critic, frozen
from barl_jax.config.toolkit import ConfigMixin
from barl_jax.hparam import Config, ConfigDict
from barl_jax.networks.diffusion import TimeConditionedField, controlled_sde
from barl_jax.networks.time_embedding import TIME_STYLES
from barl_jax.optimization import ActorOptim
from barl_jax.registration import get_component, register_component


def clip_score(score, max_norm: float, percentile: float):
  norms      = jnp.linalg.norm(score, axis=-1, keepdims=True)
  threshold  = jnp.minimum(jnp.percentile(norms, percentile), max_norm)
  scale      = jnp.where(norms > threshold, threshold / (norms + 1e-6), 1.0)
  return score * scale, jnp.mean(scale < 1.0)


@register_component
class ScoreContinuousActor(ActorOptim, Actor):
  """AMDP's controlled-SDE actor. The net uses GELU in float32 (the chain accumulates
  a log-weight) and the ctrl time coder by default. MLP settings are grouped under
  `mlp`; time_emb selects the time coder.
  Only the SDE and score-clipping knobs are this class's own."""

  POLICY = maxent_actor_policy(policy_delay=3, normalize_obs=False)

  kind = "actor"
  mlp = ConfigDict(activation='gelu', norm='none', norm_stem=False, norm_input=False,
                   norm_after_act=True, dropout=0.0, final_activation=False,
                   residual=False, norm_kwargs={}, dtype='float32')

  time_emb  = Config("ctrl", style=TIME_STYLES)
  squash_cls = Config("ErfSquash")   # `erf_k` is derived from the noise schedule below
  sigma_min    = Config(0.01)
  sigma_max    = Config(2.0)
  tau_min      = Config(0.0)
  tau_max      = Config(1.0)
  max_steps    = Config(32)
  out_clip     = Config(10000.0)   # ctrl_net.py's `outer_clip`

  q_score_max_norm       = Config(5.0)
  q_score_max_percentile = Config(95.0)

  batch_rep = Config(1)

  def __init__(self, env_stats, config: ConfigMixin, *, rngs: nnx.Rngs, **kwargs):
    self.resolve_config(config, env_stats=env_stats)

    self._log_r    = math.log(self.sigma_max / self.sigma_min)
    self._int_base = float(self._raw_int(jnp.asarray(self.tau_min, jnp.float32)))

    # latent_dim/layers are tier-ladder-owned.
    latent_dim, layers = config.get("latent_dim", int), config.get("layers", int)

    self.field = TimeConditionedField(
      self.obs_dim, self.plan_dim, latent_dim=latent_dim, layers=layers,
      time_dim=self.time_dim, time_embed=self.time_embed, time_kwargs=self.static("time_kwargs"),
      rngs=rngs, **self.mlp.todict())

    self.field.zero_init_head()   # u ~ 0 at the beginning

    taus   = jnp.linspace(self.tau_min, self.tau_max, self.max_steps + 1, dtype=jnp.float32)
    sig_sq = self.sigma_sq_int_fn(taus)                        # exact, rebased to tau_min
    self.taus     = Buffer(taus)
    self.step_var = Buffer(jnp.maximum(sig_sq[1:] - sig_sq[:-1], 0.0))
    self.rngs     = rngs

    self._terminal_var = float(self.step_var[...].sum())

    self.squash = get_component(self.squash_cls)(
      env_stats, ConfigMixin(), erf_k=(2 * self._terminal_var) ** -0.5)

  def sigma_fn(self, tau):
    r = self.sigma_max / self.sigma_min
    return self.sigma_min * r ** (1.0 - tau) * math.sqrt(2.0 * self._log_r)

  def _raw_int(self, tau):
    r = self.sigma_max / self.sigma_min
    return self.sigma_max ** 2 * (1.0 - r ** (-2.0 * tau))

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
    return jnp.clip(u, -self.out_clip, self.out_clip)

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
    key = self.rngs.noise() if key is None else key

    X = jnp.zeros((B, self.plan_dim), jnp.float32)
    X, log_w = controlled_sde(lambda x, i, k: self.euler_step(obs, x, i, k),
                              X, self.max_steps, key,
                              aux=jnp.zeros((B, self.chunk_size), jnp.float32))

    log_w = log_w + self.prior_surprisal(X)

    if squash:
      log_w = log_w + self.squash_correction(X)
      X = self.squash(X)

    return X.reshape(B, self.chunk_size, self.action_dim), log_w

  def loss(self, obs, critic: Critic, alpha, *, key=None):
    """Adjoint matching (Adjoint Matching, arXiv:2409.08861, Sec. 5.2)"""

    key = self.rngs.noise() if key is None else key
    rollout_key, tau_key, noise_key = jax.random.split(key, 3)
    alpha = jax.lax.stop_gradient(jnp.asarray(alpha))

    X, _ = self.generate_action(obs, key=rollout_key, squash=False)
    X = jax.lax.stop_gradient(X)

    def q_values(critic, x):
      return critic.value(obs, self.squash(x)).sum()

    x_score = nnx.grad(q_values, argnums=1)(frozen(critic), X)
    x_score, clipped_frac = clip_score(x_score, self.q_score_max_norm, self.q_score_max_percentile)

    def terminal_entropy(actor, x):
      return (alpha * (actor.prior_surprisal(x) + actor.squash_correction(x))).sum()

    entropy_score = nnx.grad(terminal_entropy, argnums=1)(frozen(self), X)

    score = jnp.repeat(x_score + entropy_score, self.batch_rep, axis=0)
    obs_r = jnp.repeat(obs, self.batch_rep, axis=0)
    X_r   = jnp.repeat(X, self.batch_rep, axis=0)

    tau   = jax.random.uniform(tau_key, (X_r.shape[0], 1), minval=self.tau_min, maxval=self.tau_max)
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
