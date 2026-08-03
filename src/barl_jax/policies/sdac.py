import math

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from barl_jax.networks.actor import DiscreteDiffusionActorBase
from barl_jax.networks.critic import QCriticBase
from barl_jax.networks.registry import register_component
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.networks.squash import Squash
from barl_jax.policy import Buffer, Policy, frozen, register_policy


@register_component
class VPDiffusionActor(DiscreteDiffusionActorBase):
  """Variance-preserving cosine ladder + DDPM ancestral sampler. The network
  predicts eps; `to_score` converts. Fixed schedule (Ma et al. Table 4)."""

  def __init__(self, env_stats, latent_dim: int, layers: int, num_steps: int,
               cosine_s: float = 0.008, ab_start: float = 0.999, ab_end: float = 0.001,
               *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, latent_dim, layers, num_steps, rngs=rngs, **kwargs)

    tt = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=jnp.float32)
    f  = jnp.cos((tt + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
    self.alpha_bar = Buffer(ab_end + (ab_start - ab_end) * (f / f[0]))

  def marginal_coef(self, n):
    ab = self.alpha_bar[...][n][..., None]
    return jnp.sqrt(ab), 1.0 - ab

  def to_score(self, raw, x, n):
    return -raw / jnp.sqrt(self.marginal_coef(n)[1])      # raw == eps_theta

  def prior_sample(self, batch: int, key):
    return jax.random.normal(key, (batch, self.action_dim), jnp.float32)

  def reverse_step(self, obs, x, n, key):
    ab_t, ab_prev = self.alpha_bar[...][n], self.alpha_bar[...][n - 1]
    a_t    = ab_t / ab_prev
    beta_t = 1.0 - a_t
    mean   = (x + beta_t * self.predict_score(obs, x, n)) / jnp.sqrt(a_t)
    if n > 1:
      # sigma_t^2 -- Eq. 8 omits the sqrt. `n` is a Python int here (the reverse
      # loop is unrolled), so this branch is resolved at trace time.
      var  = (1.0 - ab_prev) / (1.0 - ab_t) * beta_t
      mean = mean + jnp.sqrt(var) * jax.random.normal(key, x.shape, x.dtype)
    return mean, 0.0


@register_policy
class SDAC(Policy):
  """Soft Diffusion Actor-Critic (Ma et al. 2025).

  The actor is fitted by a *soft* denoising-score-matching loss: K posterior
  candidates are drawn per transition, priced by the critic, and the regression
  target is their softmax(Q/alpha)-weighted score. Acting is best-of-N over
  fresh diffusion samples plus a learned per-dimension exploration noise, which
  is applied in squashed space so its entropy needs no Jacobian correction."""

  # The exploration std is the Policy's own parameter -- no slot owns it.
  policy_params = {"log_std": "explore"}

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    params = self.resolve_params()
    self.entropy_scale  = params["entropy_scale"]
    self.num_candidates = params["num_candidates"]   # K: phi_{0|t} candidates for the softmax weight
    self.num_bon        = params["num_bon"]          # N: best-of-N action selection
    self.explore_init   = params["explore_init"]

    self.actor: VPDiffusionActor = self.build_component("actor", VPDiffusionActor)
    self.critic: QCriticBase     = self.build_component("critic", QCriticBase)
    self.squash: Squash          = self.build_component("squash", Squash)
    self.entropy: AutotuneRegulator = self.build_component(
      "entropy", AutotuneRegulator, target=self.entropy_scale * self.action_dim)

    # Per-dimension exploration noise, applied in SQUASHED space (post-tanh), so
    # no Jacobian correction is needed in the entropy -- unlike DIME's ELBO,
    # which lives in raw space.
    self.log_std = nnx.Param(
      jnp.full((self.action_dim,), math.log(self.explore_init), jnp.float32))

  # ---------------------------------------------------------------- utilities

  @property
  def explore_std(self):
    return jnp.exp(jnp.clip(self.log_std[...], math.log(1e-3), math.log(1.0)))

  def noise_entropy(self):
    """H of the additive exploration noise, the BoN output treated as deterministic."""
    return (0.5 * math.log(2 * math.pi * math.e) + jnp.log(self.explore_std)).sum()

  def best_of_n(self, obs, n: int, key):
    """N diffusion samples, squashed, argmax over Q. Returns (B, action_dim)."""
    B = obs.shape[0]
    obs_rep = jnp.broadcast_to(obs[None], (n, B, obs.shape[-1])).reshape(n * B, -1)
    act = self.squash(self.actor.sample(obs_rep, key))                     # (n*B, D)
    qv  = self.critic.value(obs_rep, act).reshape(n, B, -1).squeeze(-1)    # (n, B)
    idx = jnp.argmax(qv, axis=0)                                          # (B,)
    return jnp.take_along_axis(act.reshape(n, B, -1), idx[None, :, None], axis=0).squeeze(0)

  def explore(self, act, key):
    noise = self.explore_std * jax.random.normal(key, act.shape, act.dtype)
    return jnp.clip(act + noise, -1.0, 1.0)

  def create_optimizer(self) -> dict:
    return {**super().create_optimizer(), "explore": optax.adam(1e-2)}

  def train_step(self, batch):
    obs      = self.normalize_obs(batch["obs"])
    next_obs = self.normalize_obs(batch["next_obs"])
    action   = batch["act"]
    reward   = batch["rew"]
    term     = batch["term"]

    alpha  = self.entropy.value()
    B, K   = obs.shape[0], self.num_candidates
    losses = {}

    keys = jax.random.split(self.rngs.noise(), 6)

    # ---- critic ------------------------------------------------------------
    # Everything feeding the Bellman target is a target: best_of_n prices with
    # the live critic, so it has to be cut off explicitly.
    next_action = jax.lax.stop_gradient(
      self.explore(self.best_of_n(next_obs, self.num_bon, keys[0]), keys[1]))
    entropy_bonus = jnp.broadcast_to(
      jax.lax.stop_gradient(alpha * self.noise_entropy()).reshape(1, 1), (B, 1))

    self.critic.td_loss(obs, action, next_obs, next_action, reward, term,
                        losses, shift=entropy_bonus)

    # ---- actor: soft DSM ---------------------------------------------------
    t   = jax.random.randint(keys[2], (B,), 1, self.actor.num_steps + 1)
    x_t = jax.lax.stop_gradient(self.actor.diffuse(self.squash.inverse(action), t, keys[3]))

    a0_samples = jax.lax.stop_gradient(
      self.actor.sample_phi0(x_t, t, K, keys[4]))                          # (K, B, D), RAW

    qv = frozen(self.critic).value(
      jnp.broadcast_to(obs[None], (K, B, obs.shape[-1])).reshape(K * B, -1),
      self.squash(a0_samples).reshape(K * B, -1),
    ).reshape(K, B, -1).squeeze(-1)                                        # (K, B)

    weight = jax.lax.stop_gradient(
      jax.nn.softmax(qv / alpha, axis=0)[..., None])                       # (K, B, 1), sums to 1 over K
    target = jax.lax.stop_gradient(self.actor.score_target(a0_samples, x_t[None], t))
    dsm_w  = jax.lax.stop_gradient(self.actor.dsm_weight(t))               # (B, 1)

    score_pred = self.actor.predict_score(obs, x_t, t)                     # (B, D)

    sdac_loss = (weight * dsm_w * (score_pred[None] - target) ** 2).sum(axis=-1).sum(axis=0).mean()

    # ---- exploration std ---------------------------------------------------
    a_bon    = jax.lax.stop_gradient(self.best_of_n(obs, self.num_bon, keys[5]))
    a_noisy  = a_bon + self.explore_std * jax.random.normal(keys[5], a_bon.shape, a_bon.dtype)
    std_loss = -(frozen(self.critic).value(obs, a_noisy).mean()
                 + jax.lax.stop_gradient(alpha) * self.noise_entropy())

    self.entropy.loss(self.noise_entropy(), losses)

    losses["actor"]   = sdac_loss
    losses["explore"] = std_loss

    self.metrics.scalar("explore_std", jax.lax.stop_gradient(self.explore_std.mean()))
    self.metrics.scalar("noise_entropy", jax.lax.stop_gradient(self.noise_entropy()))
    # effective sample size of the softmax(Q/alpha) candidate weights, in [1, K]
    self.metrics.dist("weight_ess", 1.0 / (weight.squeeze(-1) ** 2).sum(axis=0))
    self.actor.metrics.vec("score_pred", score_pred)
    self.actor.metrics.vec("score_target", target.reshape(-1, target.shape[-1]))
    self.actor.metrics.vec("action", next_action, limit=1.0)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_obs(observation)
    key = self.rngs.noise() if key is None else key
    k_bon, k_explore = jax.random.split(key)
    return self.explore(self.best_of_n(obs, self.num_bon, k_bon), k_explore), {}
