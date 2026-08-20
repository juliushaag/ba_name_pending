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


def _inv_sigmoid(y: float) -> float:
  return math.log(y / (1.0 - y))


@register_policy
class SDAC(Policy):
  """Soft Diffusion Actor-Critic (Ma et al. 2025).

  The actor is fitted by a *soft* denoising-score-matching loss: K posterior
  candidates are drawn per transition, priced by the critic, and the regression
  target is their softmax-weighted score, at a temperature carried in units of
  the critic's own within-state spread. Acting is best-of-N over
  fresh diffusion samples plus a learned per-dimension exploration noise, which
  is applied in squashed space so its entropy needs no Jacobian correction."""

  # The exploration std is the Policy's own parameter -- no slot owns it.
  policy_params = {"explore_raw": "explore"}

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    params = self.resolve_params()
    self.entropy_scale  = params["entropy_scale"]
    self.num_candidates = params["num_candidates"]   # K: phi_{0|t} candidates for the softmax weight
    self.num_bon        = params["num_bon"]          # N: best-of-N action selection
    self.softmax_temp   = float(params["softmax_temp"])
    self.x0_data_frac   = float(params["x0_data_frac"])
    self.explore_init   = params["explore_init"]
    self.explore_min    = float(params["explore_min"])
    self.explore_max    = float(params["explore_max"])

    self.actor: VPDiffusionActor = self.build_component("actor", VPDiffusionActor)
    self.critic: QCriticBase     = self.build_component("critic", QCriticBase)
    self.squash: Squash          = self.build_component("squash", Squash)
    self.entropy: AutotuneRegulator = self.build_component(
      "entropy", AutotuneRegulator, target=self.entropy_scale * self.action_dim)

    lo, hi = self.explore_min, self.explore_max
    self.explore_raw = nnx.Param(jnp.full(
      (self.action_dim,), _inv_sigmoid((self.explore_init - lo) / (hi - lo)), jnp.float32))

  EXPLORE_RAW_LIMIT = 10.0

  # Keeps arctanh of a replay action that `explore` clipped to exactly +-1 finite.
  ACTION_CLIP = 1.0 - 1e-3

  @property
  def explore_std(self):
    lo, hi = self.explore_min, self.explore_max
    return lo + (hi - lo) * jax.nn.sigmoid(self.explore_raw[...])

  def after_step(self):
    super().after_step()
    self.explore_raw[...] = jnp.clip(self.explore_raw[...],
                                     -self.EXPLORE_RAW_LIMIT, self.EXPLORE_RAW_LIMIT)

  def noise_entropy(self):
    """H of the additive exploration noise, the BoN output treated as deterministic."""
    return (0.5 * math.log(2 * math.pi * math.e) + jnp.log(self.explore_std)).sum()

  @staticmethod
  def _tile(obs, n: int):
    return jnp.broadcast_to(obs[None], (n, *obs.shape)).reshape(n * obs.shape[0], -1)

  def pick_best(self, obs_rep, raw, n: int):
    B   = raw.shape[0] // n
    qv  = frozen(self.critic).value(obs_rep, self.squash(raw)).reshape(n, B, -1).squeeze(-1)
    idx = jnp.argmax(qv, axis=0)                                                 # (B,)
    return jnp.take_along_axis(raw.reshape(n, B, -1), idx[None, :, None], axis=0).squeeze(0)

  def best_of_n(self, obs, n: int, key):
    obs_rep = self._tile(obs, n)
    return self.pick_best(obs_rep, self.actor.sample(obs_rep, key), n)

  def explore(self, act, key):
    noise = self.explore_std * jax.random.normal(key, act.shape, act.dtype)
    return jnp.clip(act + noise, -1.0, 1.0)

  def act(self, obs, key):
    """The played action: best-of-N, squashed, plus exploration noise. Also what
    the Bellman target bootstraps from."""
    k_bon, k_explore = jax.random.split(key)
    return self.explore(self.squash(self.best_of_n(obs, self.num_bon, k_bon)), k_explore)

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

    keys = jax.random.split(self.rngs.noise(), 7)

    
    obs_rep = self._tile(obs, self.num_bon)
    joint   = jax.lax.stop_gradient(
      self.actor.sample(jnp.concatenate([next_obs, obs_rep], axis=0), keys[0]))
    next_raw  = joint[:B]
    policy_x0 = self.pick_best(obs_rep, joint[B:], self.num_bon)

    # ---- critic ------------------------------------------------------------
    next_action = jax.lax.stop_gradient(self.explore(self.squash(next_raw), keys[1]))
    self.critic.td_loss(obs, action, next_obs, next_action, reward, term, losses)

    # ---- actor: soft DSM ---------------------------------------------------
    data_x0 = self.squash.inverse(jnp.clip(action, -self.ACTION_CLIP, self.ACTION_CLIP))
    x_0 = jnp.where(jax.random.uniform(keys[6], (B, 1)) < self.x0_data_frac,
                    data_x0, policy_x0)

    t   = jax.random.randint(keys[2], (B,), 1, self.actor.num_steps + 1)
    x_t = jax.lax.stop_gradient(self.actor.diffuse(x_0, t, keys[3]))

    a0_samples = jax.lax.stop_gradient(
      self.actor.sample_phi0(x_t, t, K, keys[4]))                          # (K, B, D), RAW

    candidates = self.squash(a0_samples)                                   # (K, B, D)
    qv = frozen(self.critic).value(
      jnp.broadcast_to(obs[None], (K, B, obs.shape[-1])).reshape(K * B, -1),
      candidates.reshape(K * B, -1),
    ).reshape(K, B, -1).squeeze(-1)                                        # (K, B)

    q_scale = jax.lax.stop_gradient(qv.std(axis=0).mean()) + 1e-6
    weight  = jax.lax.stop_gradient(
      jax.nn.softmax(qv / (self.softmax_temp * q_scale), axis=0)[..., None])  # (K, B, 1), sums to 1 over K
    target = jax.lax.stop_gradient(self.actor.score_target(a0_samples, x_t[None], t))
    dsm_w  = jax.lax.stop_gradient(self.actor.dsm_weight(t))               # (B, 1)

    score_pred = self.actor.predict_score(obs, x_t, t)                     # (B, D)

    sdac_loss = (weight * dsm_w * (score_pred[None] - target) ** 2).sum(axis=-1).sum(axis=0).mean()

    # ---- exploration std ---------------------------------------------------
    a_bon    = self.squash(policy_x0)
    a_noisy  = a_bon + self.explore_std * jax.random.normal(keys[5], a_bon.shape, a_bon.dtype)
    std_loss = -(frozen(self.critic).value(obs, a_noisy).mean()
                 + jax.lax.stop_gradient(alpha) * self.noise_entropy())

    self.entropy.loss(self.noise_entropy(), losses)

    losses["actor"]   = sdac_loss
    losses["explore"] = std_loss

    self.metrics.scalar("explore_std", jax.lax.stop_gradient(self.explore_std.mean()))
    self.metrics.scalar("noise_entropy", jax.lax.stop_gradient(self.noise_entropy()))
    self.metrics.scalar("q_scale", q_scale)
    self.metrics.dist("weight_ess", 1.0 / (weight.squeeze(-1) ** 2).sum(axis=0))
   
    norm = lambda x: jnp.linalg.norm(x, axis=-1).mean()
    weighted = (weight * target).sum(axis=0)
    self.metrics.scalar("signal_ratio", norm(weighted) / (norm(target) + 1e-8))
    self.metrics.scalar("tilt_ratio",
                        norm(weighted - target.mean(axis=0)) / (norm(target.std(axis=0)) + 1e-8))
    self.metrics.scalar("cand_saturation", (jnp.abs(candidates) > 0.99).mean())
    self.actor.metrics.vec("score_pred", score_pred)
    self.actor.metrics.vec("score_target", target.reshape(-1, target.shape[-1]))
    self.actor.metrics.vec("action", next_action, limit=1.0)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_obs(observation)
    key = self.rngs.noise() if key is None else key
    return self.act(obs, key), {}
