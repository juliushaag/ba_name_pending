import math

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from barl_jax.networks.critic import QCriticBase
from barl_jax.networks.diffusion import TimeConditionedField, reverse_chain
from barl_jax.networks.optim import adam
from barl_jax.networks.registry import get_component, register_component
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.networks.schedule import Schedule
from barl_jax.policy import Actor, Buffer, Critic, Policy, frozen, register_policy


def _inv_sigmoid(y: float) -> float:
  return math.log(y / (1.0 - y))


def alpha_bar_from_betas(betas):
  """A beta ladder as the alpha_bar it implies, with the clean end prepended.

  The one thing a `Schedule` cannot state, because alpha_bar_n is a product over
  the whole prefix rather than a function of n alone -- so the curve comes from a
  schedule class and the accumulation happens here."""
  return jnp.concatenate([jnp.ones((1,), jnp.float32),
                          jnp.cumprod(1.0 - jnp.clip(betas, 0.0, 0.999))])


@register_component
class DDPMActor(Actor):
  """Fixed-step DDPM actor over an alpha_bar ladder, with the standard ancestral
  sampler -- SDAC's actor, and the only diffusion actor here.

  One class covers every variance-preserving discrete diffusion actor, because
  they differ only in three things and none of them is the update rule: the
  ladder (`ladder_cls`, see `_build_ladder`), the prior's scale, and whether the
  reconstructed x0 is clipped back into the canonical action box each reverse
  step. The posterior mean itself is shared -- see `reverse_step` for why the
  clip is the only reason its two branches are written differently.

  Every method that draws noise takes an explicit `key`: the reverse loop used to
  be differentiated through, so the draws have to be ordinary values in the graph
  rather than a side effect on the module."""

  def __init__(self, env_stats, *, latent_dim: int, layers: int, num_steps: int,
               squash_cls: str,
               ladder_cls: str, ladder_start: float, ladder_end: float,
               ladder_kwargs: dict, ladder_kind: str,
               clip_denoised: bool, prior_scale: float,
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
    self.lr        = float(lr)
    self.adam_b1   = float(adam_b1)
    self.grad_clip = float(grad_clip)
    self.grad_clip_mode = str(grad_clip_mode)
    self.clip_denoised  = bool(clip_denoised)
    self.prior_scale    = float(prior_scale)

    self.field = TimeConditionedField(
      self.obs_dim, self.action_dim, latent_dim=latent_dim, layers=layers,
      time_dim=self.time_dim, time_embed=time_embed, time_kwargs=time_kwargs,
      activation=activation, norm=norm, norm_input=norm_input, norm_stem=norm_stem,
      norm_after_act=norm_after_act, dropout=dropout, final_activation=final_activation,
      residual=residual, norm_kwargs=norm_kwargs, dtype=dtype, rngs=rngs)

    self.tau_table = Buffer(jnp.arange(num_steps + 1, dtype=jnp.float32) / num_steps)
    self.alpha_bar = Buffer(self._build_ladder(
      ladder_cls, ladder_start, ladder_end, ladder_kwargs, ladder_kind, num_steps))
    self.squash = get_component(squash_cls)(env_stats)
    self.rngs = rngs

  @staticmethod
  def _build_ladder(ladder_cls: str, start: float, end: float,
                    ladder_kwargs: dict, kind: str, num_steps: int):
    """The alpha_bar ladder over n = 0..num_steps, index 0 being the clean end.

    The *shape* of the curve is an ordinary `Schedule` named by class, so it is a
    one-word edit and a `model.components.actor.ladder_cls` sweep axis rather
    than a branch in here. `kind` says which space that curve lives in, which is
    the only thing a schedule cannot say about itself:

    * `alpha_bar` -- the curve *is* alpha_bar, `num_steps + 1` points running
      clean -> noisy. `CosineAlphaBarSchedule(start=0.999, end=0.001, s=0.008)`
      is Ma et al.'s VP-cosine ladder exactly, including its deliberate refusal
      to start at 1.
    * `beta` -- the curve is a per-step beta, `num_steps` points, accumulated
      into alpha_bar here. `LinearSchedule` over betas is the DDPM original."""
    cls = get_component(ladder_cls)
    assert issubclass(cls, Schedule), (
      f"DDPMActor: ladder_cls names {cls.__name__}, which is not a Schedule.")

    points = num_steps + 1 if kind == "alpha_bar" else num_steps
    curve  = cls(start=start, end=end, steps=points, warmup=0,
                 **(ladder_kwargs or {})).ladder()

    if kind == "alpha_bar":
      return curve
    if kind == "beta":
      return alpha_bar_from_betas(curve)
    raise ValueError(f"ladder_kind must be alpha_bar or beta, got {kind!r}")

  # ---- the noise process ---------------------------------------------------
  def marginal_coef(self, n):
    """(gamma, nu): the mean coefficient and variance of q(x_n | x_0)."""
    ab = self.alpha_bar[...][n][..., None]
    return jnp.sqrt(ab), 1.0 - ab

  def prior_sample(self, batch: int, key):
    noise = jax.random.normal(key, (batch, self.action_dim), jnp.float32)
    return self.prior_scale * noise if self.prior_scale != 1.0 else noise

  def diffuse(self, x0, n, key):
    g, nu = self.marginal_coef(n)
    return g * x0 + jnp.sqrt(nu) * jax.random.normal(key, x0.shape, x0.dtype)

  # ---- the network ---------------------------------------------------------
  def _tau(self, n, batch: int):
    tau = self.tau_table[...][n]
    tau = jnp.broadcast_to(tau, (batch,)) if jnp.ndim(tau) == 0 else tau
    return tau[:, None]

  def predict_eps(self, obs, x, n):
    return self.field(obs, x, self._tau(n, x.shape[0])).astype(jnp.float32)

  # The DDPM literature states its losses in eps space where this codebase
  # states them in score space; since this ladder's score is -eps/sqrt(nu), the
  # two are one division apart rather than a second parameterization to keep in
  # sync. `eps_from_score` is the inverse, and tests/test_actors.py pins the pair.
  def predict_score(self, obs, x, n):
    return -self.predict_eps(obs, x, n) / jnp.sqrt(self.marginal_coef(n)[1])

  def eps_from_score(self, score, n):
    return -score * jnp.sqrt(self.marginal_coef(n)[1])

  def x0_from_eps(self, x_n, eps, n):
    g, nu = self.marginal_coef(n)
    return (x_n - jnp.sqrt(nu) * eps) / g

  def eps_loss(self, obs, x0, n, key):
    """The plain DDPM regression onto the noise that produced x_n -- the
    unweighted baseline `actor_loss` below reweights. Draws its own eps rather
    than calling `diffuse`, which discards the draw the target needs. Reduction
    is sum-over-action-dims then mean; the references mean over both, which is
    this divided by `action_dim`."""
    eps = jax.random.normal(key, x0.shape, x0.dtype)
    g, nu = self.marginal_coef(n)
    x_n = g * x0 + jnp.sqrt(nu) * eps
    return ((self.predict_eps(obs, x_n, n) - eps) ** 2).sum(axis=-1).mean()

  # ---- the reverse chain ---------------------------------------------------
  def reverse_step(self, obs, x, n, key):
    ab_t, ab_prev = self.alpha_bar[...][n], self.alpha_bar[...][n - 1]
    a_t    = ab_t / ab_prev
    beta_t = 1.0 - a_t

    if self.clip_denoised:
      # x0 form, because the clip is a projection onto the action box that has
      # no expression in terms of the score alone.
      x0   = jnp.clip(self.x0_from_eps(x, self.predict_eps(obs, x, n), n), -1.0, 1.0)
      mean = ((jnp.sqrt(ab_prev) * beta_t) * x0
              + (jnp.sqrt(a_t) * (1.0 - ab_prev)) * x) / (1.0 - ab_t)
    else:
      # Algebraically the same posterior mean in eps form, in fewer ops. Kept
      # verbatim rather than folded into the branch above: it is the expression
      # SDAC's published numbers were produced with, and in float32 "equal" and
      # "identical" are not the same claim.
      mean = (x + beta_t * self.predict_score(obs, x, n)) / jnp.sqrt(a_t)

    if n > 1:
      # sigma_t^2 -- Eq. 8 omits the sqrt. `n` is a Python int here (the reverse
      # loop is unrolled), so this branch is resolved at trace time.
      var  = (1.0 - ab_prev) / (1.0 - ab_t) * beta_t
      mean = mean + jnp.sqrt(var) * jax.random.normal(key, x.shape, x.dtype)

    return mean

  def sample(self, s, key=None, critic: Critic = None):
    key = self.rngs.noise() if key is None else key
    key, prior_key = jax.random.split(key)

    x = self.prior_sample(s.shape[0], prior_key)
    return reverse_chain(lambda x, n, k: self.reverse_step(s, x, n, k),
                         x, self.num_steps, key)

  def actor_loss(self, obs, policy_x0, critic: Critic, losses: dict,
                 *, num_candidates: int, softmax_temp: float, key=None):
 
    key = self.rngs.noise() if key is None else key
    keys = jax.random.split(key, 3)
    B, K = obs.shape[0], num_candidates

    t   = jax.random.randint(keys[0], (B,), 1, self.num_steps + 1)
    x_t = jax.lax.stop_gradient(self.diffuse(policy_x0, t, keys[1]))

    noise      = jax.random.normal(keys[2], (K, B, self.action_dim), x_t.dtype)
    a0_raw     = jax.lax.stop_gradient(self.x0_from_eps(x_t[None], noise, t))
    candidates = self.squash(a0_raw)                                          # (K, B, D)

    qv = frozen(critic).value(
      jnp.broadcast_to(obs[None], (K, B, obs.shape[-1])).reshape(K * B, -1),
      candidates.reshape(K * B, -1),
    ).reshape(K, B, -1).squeeze(-1)                                      # (K, B)

    q_scale = jax.lax.stop_gradient(qv.std(axis=0).mean()) + 1e-6
    weight  = jax.lax.stop_gradient(
      jax.nn.softmax(qv / (softmax_temp * q_scale), axis=0)[..., None])  # (K, B, 1), sums to 1 over K

    # eps space rather than score space: the two are the same objective, since
    # score = -eps/sqrt(nu) makes the score loss's `dsm_weight` factor exactly
    # the nu that cancels. Stated as the reference states it, one less conversion.
    eps_pred = self.predict_eps(obs, x_t, t)                             # (B, D)
    losses["actor"] = (weight * (eps_pred[None] - noise) ** 2).sum(axis=-1).sum(axis=0).mean()

    self.metrics.scalar("q_scale", q_scale)
    self.metrics.dist("weight_ess", 1.0 / (weight.squeeze(-1) ** 2).sum(axis=0))

    norm = lambda x: jnp.linalg.norm(x, axis=-1).mean()
    weighted = (weight * noise).sum(axis=0)
    self.metrics.scalar("signal_ratio", norm(weighted) / (norm(noise) + 1e-8))
    self.metrics.scalar("tilt_ratio",
                        norm(weighted - noise.mean(axis=0)) / (norm(noise.std(axis=0)) + 1e-8))
    self.metrics.scalar("cand_saturation", (jnp.abs(candidates) > 0.99).mean())
    self.metrics.vec("eps_pred", eps_pred)
    self.metrics.vec("eps_target", noise.reshape(-1, noise.shape[-1]))

  def create_optimizer(self) -> dict:
    return {"actor": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}


@register_policy
class SDAC(Policy):
  """Soft Diffusion Actor-Critic (Ma et al. 2025).

  Reference implementation https://github.com/mahaitongdae/diffusion_policy_online_rl/

  The actor is fitted by a *soft* denoising loss -- see `DDPMActor.actor_loss`
  above, which owns it. Acting is best-of-N over fresh diffusion samples plus a
  learned per-dimension exploration noise, which is applied in squashed space so
  its entropy needs no Jacobian correction.

  Two deliberate departures from the reference remain, both stated so the
  comparison against its numbers stays falsifiable:

  * **Temperature.** The reference scales Q by `alpha_init / alpha` -- the
    constant 5.0 over the *current* learned alpha (`q_min * 5. /
    jnp.exp(log_alpha)`, relax/algorithm/sdac.py) -- so its weighting already
    tracks the entropy dual as that dual anneals. This divides by
    `softmax_temp` times the critic's own within-state spread across the K
    candidates instead, which makes the weighting invariant to the scale Q
    happens to have reached rather than tied to alpha. `actor/q_scale` logs
    that spread.
  * **Bounding.** The reference clips the reconstruction into the action box;
    this squashes it. Both bound, but a clip has zero gradient outside the box
    where tanh merely saturates -- watch `actor/cand_saturation`, and
    `actor/weight_ess` for the failure it causes (every candidate on the same
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
    self.explore_raw_limit = float(params["explore_raw_limit"])
    self.explore_lr        = float(params["explore_lr"])
   
    self.actor: DDPMActor        = self.build_component("actor", DDPMActor)
    self.critic: QCriticBase     = self.build_component("critic", QCriticBase)
    self.entropy: AutotuneRegulator = self.build_component(
      "entropy", AutotuneRegulator, target=self.entropy_scale * self.action_dim)

    lo, hi = self.explore_min, self.explore_max
    self.explore_raw = nnx.Param(jnp.full(
      (self.action_dim,), _inv_sigmoid((self.explore_init - lo) / (hi - lo)), jnp.float32))

  @property
  def explore_std(self):
    lo, hi = self.explore_min, self.explore_max
    return lo + (hi - lo) * jax.nn.sigmoid(self.explore_raw[...])

  def after_step(self):
    super().after_step()
    self.explore_raw[...] = jnp.clip(self.explore_raw[...],
                                     -self.explore_raw_limit, self.explore_raw_limit)

  def noise_entropy(self):
    """H of the additive exploration noise, the BoN output treated as deterministic."""
    return (0.5 * math.log(2 * math.pi * math.e) + jnp.log(self.explore_std)).sum()

  @staticmethod
  def _tile(obs, n: int):
    return jnp.broadcast_to(obs[None], (n, *obs.shape)).reshape(n * obs.shape[0], -1)

  def score(self, obs_rep, raw):
    return frozen(self.critic).value(obs_rep, self.actor.squash(raw)).squeeze(-1)

  def pick_best_from(self, qv, raw, n: int):
    B   = raw.shape[0] // n
    idx = jnp.argmax(qv.reshape(n, B), axis=0)                                   # (B,)
    return jnp.take_along_axis(raw.reshape(n, B, -1), idx[None, :, None], axis=0).squeeze(0)

  def pick_best(self, obs_rep, raw, n: int):
    return self.pick_best_from(self.score(obs_rep, raw), raw, n)

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
    return self.explore(self.actor.squash(self.best_of_n(obs, self.num_bon, k_bon)), k_explore)

  def create_optimizer(self) -> dict:
    return {**super().create_optimizer(), "explore": optax.adam(self.explore_lr)}

  def train_step(self, batch):
    obs      = self.normalize_obs(batch["obs"])
    next_obs = self.normalize_obs(batch["next_obs"])
    action   = batch["act"]
    reward   = batch["rew"]
    term     = batch["term"]

    alpha  = self.entropy.value()
    B      = obs.shape[0]
    losses = {}

    keys = jax.random.split(self.rngs.noise(), 4)

    # Both ends are best-of-N, and the bootstrap especially: acting takes the
    # argmax over `num_bon` samples, so a target built from a single sample
    # prices a strictly worse policy than the one collecting the data -- an
    # underestimate that grows as the actor improves, and that the critic's
    # `pessimistic_target` then compounds. The reference selects its target
    # action with the same Q-scored best-of-N it acts with.
    joint_obs = jnp.concatenate([self._tile(next_obs, self.num_bon),
                                 self._tile(obs, self.num_bon)], axis=0)
    joint     = jax.lax.stop_gradient(self.actor.sample(joint_obs, keys[0]))

    # Scored in one critic pass for the same reason both halves are sampled in
    # one actor pass: a frozen critic prices on running statistics, so the two
    # halves cannot influence each other's normalization by sharing a batch.
    split     = self.num_bon * B
    qv        = self.score(joint_obs, joint)
    next_raw  = self.pick_best_from(qv[:split], joint[:split], self.num_bon)
    policy_x0 = self.pick_best_from(qv[split:], joint[split:], self.num_bon)

    # ---- critic ------------------------------------------------------------
    next_action = jax.lax.stop_gradient(self.explore(self.actor.squash(next_raw), keys[1]))
    self.critic.td_loss(obs, action, next_obs, next_action, reward, term, losses)

    # ---- actor: soft DSM ---------------------------------------------------
    self.actor.actor_loss(obs, policy_x0, self.critic, losses,
                          num_candidates=self.num_candidates,
                          softmax_temp=self.softmax_temp, key=keys[2])

    # ---- exploration std ---------------------------------------------------
    a_bon    = self.actor.squash(policy_x0)
    a_noisy  = a_bon + self.explore_std * jax.random.normal(keys[3], a_bon.shape, a_bon.dtype)
    losses["explore"] = -(frozen(self.critic).value(obs, a_noisy).mean()
                          + jax.lax.stop_gradient(alpha) * self.noise_entropy())

    self.entropy.loss(self.noise_entropy(), losses)

    self.metrics.scalar("explore_std", jax.lax.stop_gradient(self.explore_std.mean()))
    self.metrics.scalar("noise_entropy", jax.lax.stop_gradient(self.noise_entropy()))
    self.actor.metrics.vec("action", next_action, limit=1.0)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_obs(observation)
    key = self.rngs.noise() if key is None else key
    return self.act(obs, key), {}
