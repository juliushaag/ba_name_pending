import math

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from barl_jax.components import Actor, Buffer, Critic, frozen
from barl_jax.config.toolkit import ConfigMixin
from barl_jax.hparam import Config, HParam
from barl_jax.networks.diffusion import TimeConditionedField, reverse_chain
from barl_jax.networks.flow import TIME_STYLES
from barl_jax.networks.mlp import MLP_PRESETS
from barl_jax.networks.schedule import Schedule
from barl_jax.optimization import ActorOptim, adam
from barl_jax.registration import get_component, register_component


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
class DDPMActor(Actor, ActorOptim):
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

  kind = "actor"
  # gelu over the plain net and a 20-frequency sinusoidal time coder, by
  # default -- both are Config selectors, so a gelu/silu run is a
  # `mlp_preset`/`--set` choice rather than an inheritance one.
  mlp_preset = Config("gelu", style=MLP_PRESETS)
  time_emb  = Config("sin_20", style=TIME_STYLES)
  squash_cls = Config("TanhSquash")
  num_steps    = Config(20)
  clip_denoised = Config(True)
  prior_scale  = HParam(0.5)
  # The noise ladder's *shape* is a Schedule named by class, with its endpoints
  # searchable; the selector and the kwargs dict are structural.
  ladder_cls   = Config("LinearSchedule")

  ladder_kind  = Config("beta")
  ladder_start = HParam(3.0e-5)
  ladder_end   = HParam(0.2997)
  ladder_kwargs = Config({})

  # Best-of-N and softmax weighting for the soft-DSM loss (SDAC), and the
  # additive exploration noise on top of it -- all of it lives here rather
  # than on the owning Policy, since it is how *this* actor acts and is
  # trained, not algorithm logic. See `best_of_n`/`generate_action`/`loss`
  # below.
  num_bon      = Config(8)
  num_candidates = Config(24)
  softmax_temp = HParam(0.25)
  explore_init = HParam(0.1)
  explore_min  = HParam(1.0e-3)
  explore_max  = HParam(1.0)
  explore_raw_limit = Config(10.0)
  explore_lr   = Config(1.0e-2)   # its own optimizer, separate from the actor's

  def __init__(self, env_stats, config: ConfigMixin, *, rngs: nnx.Rngs, **kwargs):
    self.resolve_hparams(config, env_stats=env_stats)

    # latent_dim/layers are tier-ladder-owned.
    latent_dim, layers = config.get("latent_dim", int), config.get("layers", int)

    self.field = TimeConditionedField(
      self.obs_dim, self.plan_dim, latent_dim=latent_dim, layers=layers,
      time_dim=self.time_dim, time_embed=self.time_embed, time_kwargs=self.hparam("time_kwargs"),
      activation=self.activation, norm=self.norm, norm_input=self.norm_input, norm_stem=self.norm_stem,
      norm_after_act=self.norm_after_act, dropout=self.dropout, final_activation=self.final_activation,
      residual=self.residual, norm_kwargs=self.hparam("norm_kwargs"), dtype=self.dtype, rngs=rngs)

    self.tau_table = Buffer(jnp.arange(self.num_steps + 1, dtype=jnp.float32) / self.num_steps)
    self.alpha_bar = Buffer(self._build_ladder(
      self.ladder_cls, self.ladder_start, self.ladder_end, self.hparam("ladder_kwargs"),
      self.ladder_kind, self.num_steps))
    self.squash = get_component(self.squash_cls)(env_stats, ConfigMixin())
    self.rngs = rngs

    # The exploration std lives here, not on the owning Policy: it is trained
    # by its own optimizer key (see create_optimizer/param_labels) but it is
    # part of how this actor plays an action, not algorithm-level state.
    lo, hi = self.explore_min, self.explore_max
    self.explore_raw = nnx.Param(jnp.full(
      (self.plan_dim,), _inv_sigmoid((self.explore_init - lo) / (hi - lo)), jnp.float32))

  def _plan(self, flat):
    return flat.reshape(flat.shape[0], self.chunk_size, self.action_dim)

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
    ladder_config = ConfigMixin(
      "", {"start": start, "end": end, "steps": points, "warmup": 0, **(ladder_kwargs or {})})
    curve = cls(config=ladder_config).ladder()

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
    noise = jax.random.normal(key, (batch, self.plan_dim), jnp.float32)
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
    is sum-over-plan-dims then mean; the references mean over both, which is
    this divided by `plan_dim`."""
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

  def sample_raw(self, obs, key=None):
    """The reverse chain alone: a raw, unsquashed x0 draw. Renamed from
    `sample` -- the ABC's `sample`/`generate_action` are the *played* action
    (best-of-N, squashed, exploration noise), which this is only one
    ingredient of."""
    key = self.rngs.noise() if key is None else key
    key, prior_key = jax.random.split(key)

    x = self.prior_sample(obs.shape[0], prior_key)
    return reverse_chain(lambda x, n, k: self.reverse_step(obs, x, n, k),
                         x, self.num_steps, key)

  @staticmethod
  def _tile(obs, n: int):
    return jnp.broadcast_to(obs[None], (n, *obs.shape)).reshape(n * obs.shape[0], -1)

  def score(self, obs_rep, raw, critic: Critic):
    return frozen(critic).value(obs_rep, self.squash(raw)).squeeze(-1)

  @staticmethod
  def pick_best_from(qv, raw, n: int):
    B   = raw.shape[0] // n
    idx = jnp.argmax(qv.reshape(n, B), axis=0)                                   # (B,)
    return jnp.take_along_axis(raw.reshape(n, B, -1), idx[None, :, None], axis=0).squeeze(0)

  def best_of_n(self, obs, n: int, *, critic: Critic = None, key=None):
    """`n` fresh chain draws at `obs`, scored by `critic` and reduced to the
    argmax -- the raw (pre-squash) x0. With no critic (or n<=1) this degenerates
    to one draw and no scoring, which is what keeps a critic-free caller (a
    contract test over every actor, say) able to call it at all."""
    key = self.rngs.noise() if key is None else key
    if critic is None or n <= 1:
      return self.sample_raw(obs, key)
    obs_rep = self._tile(obs, n)
    raw = self.sample_raw(obs_rep, key)
    return self.pick_best_from(self.score(obs_rep, raw, critic), raw, n)

  @property
  def explore_std(self):
    lo, hi = self.explore_min, self.explore_max
    return lo + (hi - lo) * jax.nn.sigmoid(self.explore_raw[...])

  def explore(self, action, key):
    noise = self.explore_std * jax.random.normal(key, action.shape, action.dtype)
    return jnp.clip(action + noise, -1.0, 1.0)

  def noise_entropy(self):
    """H of the additive exploration noise, the BoN output treated as
    deterministic. A scalar, unlike the other actors' per-sample `(B, 1)`
    entropy -- the entropy of a parameter, not of a draw."""
    return (0.5 * math.log(2 * math.pi * math.e) + jnp.log(self.explore_std)).sum()

  def after_step(self):
    super().after_step()
    self.explore_raw[...] = jnp.clip(self.explore_raw[...],
                                     -self.explore_raw_limit, self.explore_raw_limit)

  def generate_action(self, obs, *, critic: Critic = None, key=None):
    """The played action plan: best-of-N, squashed, plus exploration noise.
    Also what the owning Policy's Bellman target bootstraps from."""
    key = self.rngs.noise() if key is None else key
    k_bon, k_explore = jax.random.split(key)
    raw = self.best_of_n(obs, self.num_bon, critic=critic, key=k_bon)
    return self._plan(self.explore(self.squash(raw), k_explore)), self.noise_entropy()

  def loss(self, obs, critic: Critic, alpha, *, key=None):
    """The soft-DSM objective plus the exploration-noise objective, sharing one
    best-of-N draw: `policy_x0` (this actor's own best-of-N at `obs`, not the
    caller's) is diffused forward and regressed onto its noise, reweighted by
    a softmax over K freshly-decoded candidates' critic value -- and the same
    draw, squashed and noised, prices the `explore_raw` update on its own
    optimizer key (see `create_optimizer`/`param_labels`).

    This is a second best-of-N draw, independent of `generate_action`'s -- the
    two used to share one fused pass over `concat([tile(next_obs), tile(obs)])`
    scored in a single critic call (SDAC.train_step, pre-refactor); splitting
    them onto separate ABC methods costs the extra kernel launches of a second
    chain and a second critic invocation for identical total FLOPs. See
    REFACTOR.md's fusion measurement before assuming this is free."""
    key = self.rngs.noise() if key is None else key
    keys = jax.random.split(key, 5)
    policy_x0 = self.best_of_n(obs, self.num_bon, critic=critic, key=keys[0])

    B, K = obs.shape[0], self.num_candidates
    t   = jax.random.randint(keys[1], (B,), 1, self.num_steps + 1)
    x_t = jax.lax.stop_gradient(self.diffuse(policy_x0, t, keys[2]))

    noise      = jax.random.normal(keys[3], (K, B, self.plan_dim), x_t.dtype)
    a0_raw     = jax.lax.stop_gradient(self.x0_from_eps(x_t[None], noise, t))
    candidates = self.squash(a0_raw)                                          # (K, B, P)

    qv = frozen(critic).value(
      jnp.broadcast_to(obs[None], (K, B, obs.shape[-1])).reshape(K * B, -1),
      candidates.reshape(K * B, -1),
    ).reshape(K, B, -1).squeeze(-1)                                      # (K, B)

    q_scale = jax.lax.stop_gradient(qv.std(axis=0).mean()) + 1e-6
    weight  = jax.lax.stop_gradient(
      jax.nn.softmax(qv / (self.softmax_temp * q_scale), axis=0)[..., None])  # (K, B, 1), sums to 1 over K

    # eps space rather than score space: the two are the same objective, since
    # score = -eps/sqrt(nu) makes the score loss's `dsm_weight` factor exactly
    # the nu that cancels. Stated as the reference states it, one less conversion.
    eps_pred = self.predict_eps(obs, x_t, t)                             # (B, P)
    loss = (weight * (eps_pred[None] - noise) ** 2).sum(axis=-1).sum(axis=0).mean()

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

    a_bon   = self.squash(policy_x0)
    a_noisy = a_bon + self.explore_std * jax.random.normal(keys[4], a_bon.shape, a_bon.dtype)
    explore_loss = -(frozen(critic).value(obs, a_noisy).mean()
                     + jax.lax.stop_gradient(alpha) * self.noise_entropy())
    # Named "explore_noise_std", not "explore_std" -- a component's own
    # `self.metrics` carries no prefix (unlike a Policy's), so the metric key
    # is a bare attribute name and would collide with the `explore_std`
    # property itself (`setattr` onto a property with no setter raises).
    self.metrics.scalar("explore_noise_std", jax.lax.stop_gradient(self.explore_std.mean()))
    # Same reason as explore_noise_std above: "noise_entropy" would overwrite
    # the noise_entropy() method itself via setattr, which doesn't raise (a
    # method is a plain attribute, unlike explore_std's property) but silently
    # breaks any later call to it in the same trace.
    self.metrics.scalar("noise_entropy_value", jax.lax.stop_gradient(self.noise_entropy()))
    return {"actor": loss, "explore": explore_loss}

  def create_optimizer(self) -> dict:
    return {"actor": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode),
            "explore": optax.adam(self.explore_lr)}

  def param_labels(self, subpath) -> str:
    """`explore_raw` trains on its own optimizer key; every other parameter
    stays on the main actor optimizer."""
    return "explore" if subpath and str(subpath[0]) == "explore_raw" else "actor"
