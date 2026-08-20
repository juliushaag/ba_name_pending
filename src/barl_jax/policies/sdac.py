import math

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from barl_jax.networks.actor import DiscreteDiffusionActorBase
from barl_jax.networks.critic import QCriticBase
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.networks.squash import Squash
from barl_jax.policy import Policy, frozen, register_policy


def _inv_sigmoid(y: float) -> float:
  return math.log(y / (1.0 - y))


@register_policy
class SDAC(Policy):
  """Soft Diffusion Actor-Critic (Ma et al. 2025).

  The actor is fitted by a *soft* denoising loss: K noises are drawn per
  transition, the x0 each implies is reconstructed from x_t and priced by the
  critic, and the denoiser is regressed back onto those same noises under the
  resulting softmax weights. Acting is best-of-N over fresh diffusion samples
  plus a learned per-dimension exploration noise, which is applied in squashed
  space so its entropy needs no Jacobian correction.

  Two deliberate departures from the reference remain, both stated so the
  comparison against its numbers stays falsifiable:

  * **Temperature.** The reference divides Q by a constant (its initial alpha);
    this divides by `softmax_temp` times the critic's own within-state spread
    across the K candidates, which makes the weighting invariant to the scale Q
    happens to have reached. `sdac/q_scale` logs that spread.
  * **Bounding.** The reference clips the reconstruction into the action box;
    this squashes it. Both bound, but a clip has zero gradient outside the box
    where tanh merely saturates -- watch `sdac/cand_saturation`, and
    `sdac/weight_ess` for the failure it causes (every candidate on the same
    corner prices identically and the weights go uniform)."""

  # The exploration std is the Policy's own parameter -- no slot owns it.
  policy_params = {"explore_raw": "explore"}  # noqa: RUF012

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    params = self.resolve_params()
    self.entropy_scale  = params["entropy_scale"]
    self.num_candidates = params["num_candidates"]   # K: phi_{0|t} candidates for the softmax weight
    self.num_bon        = params["num_bon"]          # N: best-of-N action selection
    self.softmax_temp   = float(params["softmax_temp"])
    self.explore_init   = params["explore_init"]
    self.explore_min    = float(params["explore_min"])
    self.explore_max    = float(params["explore_max"])
    # Gated on the base, not VPDiffusionActor: everything SDAC calls
    # (diffuse/x0_from_eps/predict_eps/sample) is part of the shared skeleton,
    # so the noise process is swappable -- which is what makes "is the cosine
    # ladder or the reference's linear one responsible" testable as a `class:`
    # override rather than an edit.
    self.actor: DiscreteDiffusionActorBase = self.build_component(
      "actor", DiscreteDiffusionActorBase)
    self.critic: QCriticBase     = self.build_component("critic", QCriticBase)
    self.squash: Squash          = self.build_component("squash", Squash)
    self.entropy: AutotuneRegulator = self.build_component(
      "entropy", AutotuneRegulator, target=self.entropy_scale * self.action_dim)

    lo, hi = self.explore_min, self.explore_max
    self.explore_raw = nnx.Param(jnp.full(
      (self.action_dim,), _inv_sigmoid((self.explore_init - lo) / (hi - lo)), jnp.float32))

  EXPLORE_RAW_LIMIT = 10.0

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

    keys = jax.random.split(self.rngs.noise(), 6)

    
    # Both ends are best-of-N, and the bootstrap especially: acting takes the
    # argmax over `num_bon` samples, so a target built from a single sample
    # prices a strictly worse policy than the one collecting the data -- an
    # underestimate that grows as the actor improves, and that the critic's
    # `pessimistic_target` then compounds. The reference selects its target
    # action with the same Q-scored best-of-N it acts with.
    obs_rep  = self._tile(obs, self.num_bon)
    next_rep = self._tile(next_obs, self.num_bon)
    joint    = jax.lax.stop_gradient(
      self.actor.sample(jnp.concatenate([next_rep, obs_rep], axis=0), keys[0]))

    split     = self.num_bon * B
    next_raw  = self.pick_best(next_rep, joint[:split], self.num_bon)
    policy_x0 = self.pick_best(obs_rep, joint[split:], self.num_bon)

    # ---- critic ------------------------------------------------------------
    next_action = jax.lax.stop_gradient(self.explore(self.squash(next_raw), keys[1]))
    self.critic.td_loss(obs, action, next_obs, next_action, reward, term, losses)

    # ---- actor: soft DSM ---------------------------------------------------
    t   = jax.random.randint(keys[2], (B,), 1, self.actor.num_steps + 1)
    x_t = jax.lax.stop_gradient(self.actor.diffuse(policy_x0, t, keys[3]))

    # The reference's candidate mechanism: draw K noises, reconstruct the x0 each
    # one implies from x_t, price those, and regress the denoiser back onto the
    # *same* noises under the resulting weights. Candidate k and target k are
    # therefore the same draw -- which is what makes the softmax a reweighting of
    # the denoising target rather than a separate scoring step.
    noise      = jax.random.normal(keys[4], (K, B, self.action_dim), x_t.dtype)
    a0_raw     = jax.lax.stop_gradient(self.actor.x0_from_eps(x_t[None], noise, t))
    candidates = self.squash(a0_raw)                                       # (K, B, D)

    qv = frozen(self.critic).value(
      jnp.broadcast_to(obs[None], (K, B, obs.shape[-1])).reshape(K * B, -1),
      candidates.reshape(K * B, -1),
    ).reshape(K, B, -1).squeeze(-1)                                        # (K, B)

    q_scale = jax.lax.stop_gradient(qv.std(axis=0).mean()) + 1e-6
    weight  = jax.lax.stop_gradient(
      jax.nn.softmax(qv / (self.softmax_temp * q_scale), axis=0)[..., None])  # (K, B, 1), sums to 1 over K

    # eps space rather than score space: the two are the same objective, since
    # score = -eps/sqrt(nu) makes the score loss's `dsm_weight` factor exactly
    # the nu that cancels. Stated as the reference states it, one less conversion.
    eps_pred  = self.actor.predict_eps(obs, x_t, t)                        # (B, D)
    sdac_loss = (weight * (eps_pred[None] - noise) ** 2).sum(axis=-1).sum(axis=0).mean()

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
    weighted = (weight * noise).sum(axis=0)
    self.metrics.scalar("signal_ratio", norm(weighted) / (norm(noise) + 1e-8))
    self.metrics.scalar("tilt_ratio",
                        norm(weighted - noise.mean(axis=0)) / (norm(noise.std(axis=0)) + 1e-8))
    self.metrics.scalar("cand_saturation", (jnp.abs(candidates) > 0.99).mean())
    self.actor.metrics.vec("eps_pred", eps_pred)
    self.actor.metrics.vec("eps_target", noise.reshape(-1, noise.shape[-1]))
    self.actor.metrics.vec("action", next_action, limit=1.0)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_obs(observation)
    key = self.rngs.noise() if key is None else key
    return self.act(obs, key), {}
