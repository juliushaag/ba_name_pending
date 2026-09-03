import math

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.configs import ConfigMixin
from barl_jax.hparam import Config, HParam
from barl_jax.networks.ensemble import Ensemble
from barl_jax.networks.mlp import MLP, MLP_STYLES
from barl_jax.networks.optim import adam, CriticOptim
from barl_jax.networks.registry import register_component
from barl_jax.policy import TARGET_SUFFIX, Buffer, Critic, frozen


# ---------------------------------------------------------------------------
# Critic config mixins -- the algorithm geometry a critic owns, so a concrete
# critic class states only what its representation adds (a distribution's
# atoms, an ensemble's rho) and inherits the rest.
# ---------------------------------------------------------------------------


class CriticLossConfig:
  """The Bellman geometry every critic shares: the discount, whether to price
  the heads' disagreement into the price and/or the target, and the Polyak
  target net."""

  gamma             = HParam(0.99)
  pessimistic_target = HParam(False)
  polyak_tau        = HParam(0.0)
  target_delay      = HParam(1)


class CriticDistConfig:
  """The distributional critics' geometry (C51 / HLGauss): how many atoms / bins
  the support carries and the smoothing around it. All are HParams -- the return
  representation a sweep may vary."""

  num_atoms          = HParam(101)
  num_bins           = HParam(151)
  sigma_to_bin_ratio = HParam(0.75)
  entropy_coef       = HParam(0.0)
  rho                = HParam(0.0)


class Polyak:
  """Opts the inheriting critic into a Polyak target net at the NC-LQL tau.
  *Must* be used with a non-BatchRenorm MLP style -- `QCriticBase` raises on the
  polyak+batchrenorm pairing."""

  polyak_tau = HParam(0.005)


class QCriticBase(CriticLossConfig, CriticOptim, Critic):
  """`polyak_tau > 0` opts this critic into a Polyak-averaged target network.

  At the default `0.0` nothing is cloned: no target modules, no extra state, no
  work in `after_step()`, and `loss` keeps the single fused forward over
  `[(s,a), (s',a')]` that CrossQ's shared BatchRenorm pass exists to exploit.
  That is why the two are mutually exclusive rather than merely discouraged --
  they are alternative answers to the same overestimation problem, and running
  both gives two half-cancelling mechanisms and a plausible-looking curve.

  Declarations come from the `CriticLossConfig`/`CriticOptim` mixins (the Bellman
  geometry and the critic optimizer); a concrete class adds its `net_style`
  selector and its own reduction geometry."""

  # False on the plain contract; `NoiseConditionedCritic` flips it because its
  # `loss` is a wider signature (TD at the clean level plus smoothing over the
  # noisier ones) that the RLPolicy template's one-backward call cannot drive.
  noise_conditioned = False

  def __init__(self, env_stats, config: ConfigMixin, q_function, **kwargs):
    super().__init__(env_stats, config, **kwargs)

    self.q_function = q_function

    # HParam values (gamma/lr/.../norm) are already resolved onto self by
    # Component.__init__; the polyak check and the clone are the rest.
    if self.polyak_tau > 0.0:
      if self.norm in ("batch", "batchrenorm"):
        raise ValueError(
          f"{type(self).__name__}: polyak_tau={self.polyak_tau} with norm={self.norm!r}. A target net "
          f"and CrossQ-style running-statistic normalization are alternative answers to the "
          f"same problem -- the target's separate forward pass removes the shared "
          f"[(s,a), (s',a')] batch that makes BatchRenorm worth having. Pick one: either "
          f"polyak_tau: 0.0, or norm: layer/none.")
      for name, module in self.target_modules().items():
        setattr(self, f"{name}{TARGET_SUFFIX}", nnx.clone(module))
      self.target_steps = Buffer(jnp.zeros((), jnp.int32))

  # ---- target network ------------------------------------------------------

  @property
  def has_target(self) -> bool:
    return self.polyak_tau > 0.0

  def target_modules(self) -> dict:
    """Live modules to keep a Polyak copy of, keyed by the name their copy takes
    (`critic` -> `self.critic_target`). The `_target` suffix keeps the clone out of
    TRAINABLE, so a subclass adding an entry gets that for free. Called from
    `__init__`, so anything it names must already be assigned."""
    return {"critic": self.q_function}

  def target_raw(self, s, a):
    """The bootstrap's un-reduced prediction: from the target copy if there is
    one, else from a frozen view of the live net -- so `td_loss` has one path."""
    net = self.critic_target if self.has_target else frozen(self.q_function)
    return jax.lax.stop_gradient(net(s, a))

  def after_step(self):
    if not self.has_target:
      return

    self.target_steps[...] += 1
    # `target_delay` is a Python int, so the cadence is resolved at trace time;
    # only the counter comparison is traced.
    due = ((self.target_steps[...] % self.target_delay) == 0 if self.target_delay > 1
           else True)

    for name, live_module in self.target_modules().items():
      target_module = getattr(self, f"{name}{TARGET_SUFFIX}")
      live = nnx.state(live_module, nnx.Param)
      tgt  = nnx.state(target_module, nnx.Param)

      blend = lambda t, l: t + self.polyak_tau * (l - t)
      step  = (jax.tree.map(lambda t, l: jnp.where(due, blend(t, l), t), tgt, live)
               if self.target_delay > 1 else jax.tree.map(blend, tgt, live))
      nnx.update(target_module, step)

      self.metrics.gap(f"{name}{TARGET_SUFFIX}", live, tgt)

  # ---- pricing -------------------------------------------------------------

  def raw(self, s, a):
    return self.q_function(s, a)

  def raw_batches(self, *pairs):
    sizes = [s.shape[0] for s, _ in pairs]
    joint = self.raw(jnp.concatenate([s for s, _ in pairs], axis=0),
                     jnp.concatenate([a for _, a in pairs], axis=0))
    out, start = [], 0
    for n in sizes:
      out.append(joint[..., start:start + n, :])
      start += n
    return out

  def reduce_value(self, raw):
    raise NotImplementedError

  def value(self, s, a):
    return self.reduce_value(self.raw(s, a))

  def expectation_from_raw(self, raw):
    """E[Z] per head, `(n_models, B, 1)` -- the identity for a head that already
    predicts a scalar, an integral over the support for one that does not."""
    return raw

  def mean_expectation(self, raw):
    """E[Z] averaged over the heads -- `reduce_value`'s risk-neutral
    counterpart, and what the two differ by is `pessimism_gap`."""
    return self.expectation_from_raw(raw).astype(jnp.float32).mean(axis=0)

  def pessimism_gap(self, raw):
    """How far below E[Z] the price `value()` returns sits, whatever reduction
    produced it. Defined as the difference rather than reconstructed from a
    coefficient so it stays exact when a subclass changes how it reduces."""
    return self.mean_expectation(raw) - self.reduce_value(raw).astype(jnp.float32)

  def expectation(self, s, a):
    """E[Z] averaged over the heads, where `value()` is the pessimistic price.
    A Policy whose improvement direction *is* the critic's action-gradient field
    (NC-LQL's Langevin chain) differentiates this one: a spread term would move
    the distribution it samples from, not merely bound the price."""
    return self.mean_expectation(self.raw(s, a))

  def create_optimizer(self) -> dict:
    return {"critic": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}

  # ---- regression ----------------------------------------------------------

  def bootstrap_from_raw(self, next_raw):
    """`rho` prices the heads' disagreement into `value()`; this decides whether
    it is priced into the *target* as well. The two are independent because they
    answer to different things -- dime_am puts pessimism in neither (both its
    reductions are the plain mean), SAC-style clipped double-Q in both, and
    NC-LQL needs it in the target because it has no actor to keep the bootstrap
    action on-distribution."""
    reduce = self.reduce_value if self.pessimistic_target else self.mean_expectation
    return jax.lax.stop_gradient(reduce(next_raw))

  def bellman_target(self, reward, term, bootstrap, shift=0.0):
    reward = reward.reshape(-1, 1)
    term   = term.reshape(-1, 1).astype(reward.dtype)
    return reward + (1.0 - term) * self.gamma * (bootstrap + shift)

  def target_from_raw(self, raw):
    """One of this critic's own predictions lifted into a regression target --
    head-averaged and detached, in whatever representation `loss()` regresses
    against. What a Policy needs to pull one forward pass onto another (NC-LQL's
    smoothing loss pulls the noisy levels onto the clean one) without knowing
    whether that means a scalar or a distribution over atoms."""
    return jax.lax.stop_gradient(raw.mean(axis=0))

  def regression_loss(self, raw, target):
    raise NotImplementedError

  def record(self, raw, target) -> None:
    """The diagnostics half of `loss_from_raw`, separate so a Policy that fits
    the same heads to two targets (NC-LQL) pays the `critic/q/*` keys once,
    against whichever pass it means, instead of the second call overwriting the
    first."""
    raise NotImplementedError

  def loss_from_raw(self, raw, target, losses: dict) -> None:
    losses["critic"] = self.regression_loss(raw, target)
    self.record(raw, target)

  def loss(self, obs, action, next_obs, next_action, reward, term,
              losses: dict, shift=0.0) -> None:
    if self.has_target:
      # The bootstrap comes from a different network, so there is no shared
      # forward to fuse -- two separate passes is the whole point of a target.
      pred_raw = self.raw(obs, action)
      next_raw = self.target_raw(next_obs, next_action)
    else:
      pred_raw, next_raw = self.raw_batches((obs, action), (next_obs, next_action))
    target = self.bellman_target(reward, term, self.bootstrap_from_raw(next_raw), shift)
    self.loss_from_raw(pred_raw, target, losses)


class ConcatQ(nnx.Module):
  """`q_function(s, a)` over concatenated inputs: `n_models` vmapped MLP heads
  with a leading model axis. `n_models: 1` is the single-head case -- it keeps
  the axis, so every critic reduces over heads the same way."""

  def __init__(self, in_dim: int, out_dim: int, latent_dim: int, layers: int, *, n_models: int,
               activation: str, norm: str,
               norm_input: bool, norm_stem: bool, norm_after_act: bool,
               dropout: float, final_activation: bool,
               residual: bool, norm_kwargs: dict, dtype: str,
               rngs: nnx.Rngs):
    self.net = Ensemble(n_models, MLP, input_dim=in_dim, output_dim=out_dim,
                        latent_dim=latent_dim, hidden_layers=layers,
                        activation=activation, norm=norm, norm_input=norm_input,
                         norm_stem=norm_stem, norm_after_act=norm_after_act,
                        dropout=dropout, final_activation=final_activation,
                        residual=residual, dtype=dtype, rngs=rngs, **(norm_kwargs or {}))

  def __call__(self, s, a):
    return self.net(jnp.concatenate([s, a], axis=-1))


# ----------------------------------------------------------------------
# Scalar critic
# ----------------------------------------------------------------------


@register_component
class EnsembleCritic(QCriticBase):
  """`n_models` MLP heads predicting a scalar Q, reduced by mean - `rho`*std.

  The non-distributional counterpart to C51EnsembleCritic, on the same
  scaffolding and with the same slot kwargs, so swapping one for the other is a
  one-word `class:` edit -- which is what makes "does the distributional part
  earn its keep here" an A/B rather than a rewrite. The spread it reduces with
  is *epistemic*, disagreement between independently initialized heads, where a
  distributional critic's is the return's own.

  `rho` subsumes the usual reductions rather than needing a class each: 0.0 is
  the plain mean, and at `n_models: 2` the population std is |q1 - q2|/2, soy
  `rho: 1.0` is exactly the min-over-heads that C51EnsembleCritic takes."""

  kind = "critic"
  # CrossQ net: BatchRenorm over the shared [(s,a),(s',a')] forward.
  net_style = Config("crossq", style=MLP_STYLES)
  rho                = HParam(1.0)
  pessimistic_target = HParam(True)

  def __init__(self, env_stats, config: ConfigMixin, *, rngs: nnx.Rngs, **kwargs):
    self.resolve_hparams(config, env_stats=env_stats)
    # latent_dim/layers/n_models are size-ladder-owned -- not declared on any
    # class, so they are read structurally from the resolved layer.
    latent_dim, layers, n_models = (config.get(k, int) for k in ("latent_dim", "layers", "n_models"))

    q_function = ConcatQ(env_stats["obs_dim"] + env_stats["act_dim"], 1,
                         latent_dim, layers, n_models=n_models,
                         activation=self.activation, norm=self.norm, norm_input=self.norm_input,
                         norm_stem=self.norm_stem, norm_after_act=self.norm_after_act,
                         dropout=self.dropout, final_activation=self.final_activation,
                         residual=self.residual, norm_kwargs=self.hparam("norm_kwargs"), dtype=self.dtype,
                         rngs=rngs)

    super().__init__(env_stats, config, q_function, **kwargs)

    self.n_models = n_models

  def reduce_value(self, raw):
    raw = raw.astype(jnp.float32)
    std = jnp.sqrt(jnp.maximum(raw.var(axis=0), 1e-8))
    return raw.mean(axis=0) - self.rho * std

  def regression_loss(self, raw, target):
    pred = raw.astype(jnp.float32)                             # (n_models, B, 1)
    tgt  = jax.lax.stop_gradient(target).astype(jnp.float32)   # (B, 1), shared by every head
    return ((pred - tgt) ** 2).mean()

  def record(self, raw, target) -> None:
    pred = raw.astype(jnp.float32)

    self.metrics.dist("critic", pred.mean(axis=0))
    self.metrics.dist("q_target", jax.lax.stop_gradient(target).astype(jnp.float32))
    self.metrics.ensemble("ensemble", pred, gap=self.pessimism_gap(raw))


@register_component
class TargetEnsembleCritic(Polyak, EnsembleCritic):
  """`EnsembleCritic` that bootstraps from a Polyak target instead of leaning on
  CrossQ's shared-forward BatchRenorm.

  It exists as a class rather than as two keys in a slot because the two
  mechanisms are mutually exclusive (`QCriticBase` raises if both are on) and
  setting `polyak_tau` while leaving the crossq `net_style` is the easy mistake
  -- it does not fail, it just half-cancels and produces a plausible curve.
  `net_style` needs the layer variant and the `Polyak` marker the tau; the swap
  is then one word: `class: EnsembleCritic` -> `class: TargetEnsembleCritic`."""

  kind = "critic"
  net_style = Config("layer_relu", style=MLP_STYLES)


# ----------------------------------------------------------------------
# Classification-as-regression critics
# ----------------------------------------------------------------------


class CategoricalCriticBase(QCriticBase, CriticDistConfig):
  """A head emitting logits over a **fixed support**, fitted by cross-entropy
  against a target distribution.

  Both concrete critics below are this; they differ only in how a target
  becomes that distribution. C51 transports the whole categorical through a
  projection, HL-Gauss discretizes a scalar onto a Gaussian-smoothed histogram
  -- so `bellman_target` is the entire difference, and everything about pricing
  and the loss is shared, including the `n_models` heads and their reduction."""

  net_style = Config("crossq", style=MLP_STYLES)

  def __init__(self, env_stats, config: ConfigMixin, q_function, support, **kwargs):
    super().__init__(env_stats, config, q_function, **kwargs)

    self.support = Buffer(support)
    # n_models is size-ladder-owned and read structurally (see EnsembleCritic).
    self.n_models = config.get("n_models", int)

  def probs(self, raw):
    return jax.nn.softmax(raw.astype(jnp.float32), axis=-1)

  def expectation_from_raw(self, raw):
    return (self.probs(raw) * self.support[...]).sum(axis=-1, keepdims=True)

  def reduce_value(self, raw):
    """`rho` means what it means on EnsembleCritic -- the spread reduced with is
    disagreement between heads, epistemic, whatever each head predicts. Sharing
    the reduction is what makes swapping this family for that one an A/B over
    the return representation alone."""
    exp = self.expectation_from_raw(raw).astype(jnp.float32)
    std = jnp.sqrt(jnp.maximum(exp.var(axis=0), 1e-8))
    return exp.mean(axis=0) - self.rho * std

  def target_from_raw(self, raw):
    """The heads' mean *distribution*, not their mean expectation -- the target
    has to live on the support that `regression_loss` cross-entropies against."""
    return jax.lax.stop_gradient(self.probs(raw).mean(axis=0))

  def regression_loss(self, raw, target):
    logp = jax.nn.log_softmax(raw.astype(jnp.float32), axis=-1)
    tgt  = jax.lax.stop_gradient(target).astype(jnp.float32)

    # sum(-1) is the per-sample cross-entropy, mean(-1) averages the batch, and
    # sum() adds up the heads -- a no-op on the scalar a single head leaves.
    reduce = lambda x: x.sum(axis=-1).mean(axis=-1).sum()

    loss = reduce(-(tgt * logp))
    if self.entropy_coef:
      loss += self.entropy_coef * reduce(jnp.exp(logp) * logp)
    return loss

  def record(self, raw, target) -> None:
    tgt     = jax.lax.stop_gradient(target).astype(jnp.float32)
    support = self.support[...]
    probs   = self.probs(raw)                                    # (n_models, B, atoms)
    exp     = (probs * support).sum(axis=-1, keepdims=True)      # expectation_from_raw

    self.metrics.dist("critic", exp.mean(axis=0))
    self.metrics.dist("q_target", (tgt * support).sum(axis=-1, keepdims=True))
    self.metrics.ensemble("ensemble", exp, gap=self.pessimism_gap(raw))

    var = (probs * (support - exp) ** 2).sum(axis=-1)
    self.metrics.dist("return_std", jnp.sqrt(var).mean(axis=0))


def categorical_projection(next_probs, atoms, reward, term, gamma, shift, v_min, v_max):
  num_atoms = atoms.shape[0]
  delta_z   = (v_max - v_min) / (num_atoms - 1)

  reward = reward.reshape(-1, 1)
  term   = term.reshape(-1, 1).astype(reward.dtype)
  # `shift` rides inside the discount exactly as in QCriticBase.bellman_target,
  # so an entropy bonus means the same thing here as for a scalar critic -- it
  # moves the *support*, not the probabilities.
  tz = jnp.clip(reward + (1.0 - term) * gamma * (atoms.reshape(1, -1) + shift), v_min, v_max)

  b = (tz - v_min) / delta_z
  l = jnp.clip(jnp.floor(b), 0, num_atoms - 1).astype(jnp.int32)
  u = jnp.clip(jnp.ceil(b),  0, num_atoms - 1).astype(jnp.int32)

  # An atom landing exactly on a support point must not have its mass split
  # between l and u (they are equal there) or it would be counted twice.
  on_atom = (l == u)
  m_l = jnp.where(on_atom, next_probs, next_probs * (u.astype(b.dtype) - b))
  m_u = jnp.where(on_atom, 0.0,        next_probs * (b - l.astype(b.dtype)))

  proj = jnp.zeros_like(next_probs)
  batch_idx = jnp.arange(next_probs.shape[-2])[:, None]
  proj = proj.at[..., batch_idx, l].add(m_l)
  proj = proj.at[..., batch_idx, u].add(m_u)
  return proj


@register_component
class C51Critic(CategoricalCriticBase):
  """`n_models` C51 heads over a fixed categorical support, trained by the
  distributional Bellman operator (Bellemare, Dabney & Munos, 2017).

  Genuinely distributional: it transports the whole next-state distribution
  rather than a scalar, so its spread is the return's own. Both head reductions
  follow dime_am, and both are the plain mean at the shipped `rho: 0.0`: the
  target averages the heads' projected distributions so both regress onto one
  well-posed target, and the price a Policy improves against averages their
  expectations (`policy_q_reduce_fn = jnp.mean` there -- the `min_qf_pi` its
  critic loss computes is a metric, not a learning signal). `rho` and
  `pessimistic_target` are the two independent ways to put the heads'
  disagreement back, in the price and in the target respectively; NC-LQL's
  subclass turns the first on, having no actor to brake over-estimation.

  Deviations from dime_am worth knowing: its net emits post-softmax
  probabilities and takes log(p + 1e-15), where this keeps logits and uses
  log_softmax -- identical in exact arithmetic, stable in float32; and this MLP
  has a stem the reference lacks, so `layers` here is one Linear deeper than the
  reference's `hs` of the same length (`layers: 1` + `norm_stem: true` is its
  `hs: [w, w]`)."""

  # The C51 defaults that differ from CriticDistConfig's: entropy_coef on,
  # rho/pessimistic off (both already CriticDistConfig's defaults -- stated so
  # the swapped-versus-EnsembleCritic story stays a single `class:` edit).
  kind = "critic"
  entropy_coef = HParam(0.005)

  def __init__(self, env_stats, config: ConfigMixin, *, rngs: nnx.Rngs, **kwargs):
    self.resolve_hparams(config, env_stats=env_stats)
    # n_models is size-ladder-owned; v_min/v_max fall back to the env's declared
    # return range -- a property of the env, not a class default.
    n_models = config.get("n_models", int)
    v_min = config.get("v_min", float, default=env_stats["return_min"])
    v_max = config.get("v_max", float, default=env_stats["return_max"])

    q_function = ConcatQ(env_stats["obs_dim"] + env_stats["act_dim"], self.num_atoms,
                         config.get("latent_dim", int), config.get("layers", int),
                         n_models=n_models,
                         activation=self.activation, norm=self.norm, norm_input=self.norm_input,
                         norm_stem=self.norm_stem, norm_after_act=self.norm_after_act,
                         dropout=self.dropout, final_activation=self.final_activation,
                         residual=self.residual, norm_kwargs=self.hparam("norm_kwargs"),
                         dtype=self.dtype, rngs=rngs)

    super().__init__(env_stats, config, q_function,
                     jnp.linspace(v_min, v_max, self.num_atoms, dtype=jnp.float32), **kwargs)

    self.v_min, self.v_max = v_min, v_max

  def derived(self) -> dict:
    """The support is the env's return range unless a kwarg pins it, so it is
    not readable off this critic's own config."""
    return {"v_min": self.v_min, "v_max": self.v_max,
            "bin_width": (self.v_max - self.v_min) / (self.num_atoms - 1)}

  def bootstrap_from_raw(self, next_raw):
    """The next-state *distribution*, per head -- not a scalar, so the base's
    `pessimistic_target` reduction cannot apply here; `bellman_target` below
    applies the flag in the only form that keeps a distribution."""
    return jax.lax.stop_gradient(self.probs(next_raw))

  def bellman_target(self, reward, term, bootstrap, shift=0.0):
    if self.pessimistic_target:
      # Transport the whole distribution of the head with the lower E[Z], per
      # sample. A pointwise min over probabilities is not a distribution;
      # picking a head keeps one, and the mean below is then a no-op.
      exp = (bootstrap * self.support[...]).sum(axis=-1, keepdims=True)   # (n, B, 1)
      bootstrap = jnp.take_along_axis(bootstrap, jnp.argmin(exp, axis=0, keepdims=True),
                                      axis=0)                             # (1, B, atoms)
    projected = categorical_projection(bootstrap, self.support[...], reward, term,
                                       self.gamma, shift, self.v_min, self.v_max)
    return projected.mean(axis=0)


@register_component
class HLGaussCritic(CategoricalCriticBase):
  """`n_models` heads over a fixed histogram support, fitted by cross-entropy
  against a Gaussian-smoothed discretization of the scalar Bellman target
  (Farebrother et al., 2024).

  Unlike C51 this is *not* distributional: it bootstraps a scalar E[Z] and only
  discretizes at the very end, so `sigma` is label smoothing, not a return
  spread -- and the spread its heads disagree by is epistemic, as in
  EnsembleCritic. The ensemble is this codebase's, not the paper's, and exists
  so all three critics differ only in the return representation."""

  kind = "critic"

  def __init__(self, env_stats, config: ConfigMixin, *, rngs: nnx.Rngs, **kwargs):
    self.resolve_hparams(config, env_stats=env_stats)
    n_models = config.get("n_models", int)

    # Same kwarg-over-EnvStats fallback as C51Critic -- see its __init__.
    v_min = config.get("v_min", float, default=env_stats["return_min"])
    v_max = config.get("v_max", float, default=env_stats["return_max"])
    edges = jnp.linspace(v_min, v_max, self.num_bins + 1, dtype=jnp.float32)

    q_function = ConcatQ(env_stats["obs_dim"] + env_stats["act_dim"], self.num_bins,
                         config.get("latent_dim", int), config.get("layers", int),
                         n_models=n_models,
                         activation=self.activation, norm=self.norm, norm_input=self.norm_input,
                         norm_stem=self.norm_stem, norm_after_act=self.norm_after_act,
                         dropout=self.dropout, final_activation=self.final_activation,
                         residual=self.residual, norm_kwargs=self.hparam("norm_kwargs"),
                         dtype=self.dtype, rngs=rngs)

    super().__init__(env_stats, config, q_function, (edges[:-1] + edges[1:]) / 2, **kwargs)

    self.v_min, self.v_max = v_min, v_max
    self.sigma    = self.sigma_to_bin_ratio * (self.v_max - self.v_min) / self.num_bins
    self.edges    = Buffer(edges)

  def derived(self) -> dict:
    """`sigma` is the label smoothing in *return units*, derived from the support
    and `num_bins` -- so changing either silently rescales it. Logged for exactly
    that reason."""
    return {"v_min": self.v_min, "v_max": self.v_max, "sigma": float(self.sigma),
            "bin_width": (self.v_max - self.v_min) / self.num_bins}

  def _to_histogram(self, value):
    value = jnp.clip(value.reshape(-1, 1), self.v_min, self.v_max)
    z     = (self.edges[...] - value) / (self.sigma * math.sqrt(2.0))
    cdf   = 0.5 * (1.0 + jax.scipy.special.erf(z))
    mass  = cdf[:, 1:] - cdf[:, :-1]
    return mass / jnp.maximum(mass.sum(axis=-1, keepdims=True), 1e-8)

  def bellman_target(self, reward, term, bootstrap, shift=0.0):
    return self._to_histogram(super().bellman_target(reward, term, bootstrap, shift))
