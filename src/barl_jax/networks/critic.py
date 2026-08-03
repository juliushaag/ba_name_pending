import math

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from barl_jax.networks.ensemble import Ensemble
from barl_jax.networks.mlp import MLP
from barl_jax.networks.registry import register_component
from barl_jax.policy import Buffer, Critic


class Bootstrap:
  """What `bellman_target()` consumes: a target-net prediction in the critic's
  own representation, plus a per-sample value shift. Keeping the shift separate
  is what lets a Policy write `critic.bootstrap(...) + alpha * entropy` no
  matter which critic it holds -- for a fixed categorical support a value shift
  moves the support, not the probabilities, so plain addition would silently
  corrupt it."""

  def __init__(self, pred, shift=0.0):
    self.pred  = pred
    self.shift = shift

  def __add__(self, other) -> "Bootstrap":
    return Bootstrap(self.pred, self.shift + other)

  __radd__ = __add__


class QCriticBase(Critic):
  """Shared scaffolding for every concrete critic below: the bootstrap ->
  bellman_target -> loss regression path, and the target network -- which is
  optional. Not instantiated directly.

  Two accessors, both meaningful for every representation: `expectation()` is
  E[Z(s,a)], `value()` is the price a Policy improves against -- the same thing
  unless a subclass reduces with a spread.

  A subclass supplies three hooks and gets everything else from here:
  `reduce_expectation(raw)` and (optionally) `reduce_value(raw)` turn an
  un-reduced prediction into a scalar, and `loss_from_raw(raw, target, losses)`
  regresses one. Keeping those separate from the forward pass is what lets
  `td_loss()` price both batches in a single call when there is no target net."""

  def __init__(self, env_stats, q_function, *, polyak_tau: float, gamma: float,
               lr: float = 3e-4, polyak_ramp: bool = True, use_target: bool = True,
               **kwargs):
    super().__init__(env_stats, **kwargs)

    self.polyak_tau  = float(polyak_tau)
    self.gamma       = float(gamma)
    self.lr          = float(lr)
    self.polyak_ramp = bool(polyak_ramp)
    self.use_target  = bool(use_target)

    self.q_function = q_function
    if self.use_target:
      # Named `*_target` so the TRAINABLE filter excludes it from every optimizer --
      # nnx.update() preserves a Variable's original type, so the clone's leaves
      # stay nnx.Param and cannot be retyped into something the filter would skip.
      self.q_target = nnx.clone(q_function)
      self.polyak_step = Buffer(jnp.zeros((), jnp.float32))

  # ------------------------------------------------------------------
  # Target network
  # ------------------------------------------------------------------

  # `polyak_ramp` reproduces a quirk of the torch implementation rather than
  # its config: QCriticBase there built `EMA(q_function)` with EMA's *default*
  # decay and never passed `polyak_tau` on, so the effective interpolation
  # weight ramped 0.1 -> 0.01 over 5000 steps and the configured
  # `polyak_tau: 0.005` was dead for every critic. Keeping it the default here
  # is what makes a torch-vs-JAX learning-curve comparison mean something;
  # set `polyak_ramp: false` in config/components.yaml to honour `polyak_tau`.
  _RAMP_START, _RAMP_END, _RAMP_STEPS = 0.9, 0.99, 5000

  def polyak_weight(self):
    assert self.use_target, "no target network to Polyak towards (use_target=False)"
    if not self.polyak_ramp:
      return self.polyak_tau
    t = jnp.minimum(self.polyak_step[...] / self._RAMP_STEPS, 1.0)
    decay = self._RAMP_START + t * (self._RAMP_END - self._RAMP_START)
    return 1.0 - decay

  def before_step(self):
    """Polyak update, driven by the Policy's lifecycle fan-out rather than by
    each train_step. A no-op without a target network."""
    if not self.use_target:
      return
    self.polyak_step[...] = self.polyak_step[...] + 1
    online = nnx.state(self.q_function, nnx.Param)
    target = nnx.state(self.q_target, nnx.Param)
    nnx.update(self.q_target, optax.incremental_update(online, target, self.polyak_weight()))

  def target_gap(self, name: str = "ema"):
    """How far the Polyak target has drifted from the live net."""
    if not self.use_target:
      return
    self.metrics.gap(name,
                     nnx.state(self.q_function, nnx.Param),
                     nnx.state(self.q_target, nnx.Param))

  # ------------------------------------------------------------------
  # Pricing
  # ------------------------------------------------------------------

  def raw(self, s, a, target: bool = False):
    """Un-reduced prediction: (n_models, B, 1) ensemble heads, or (B, num_bins)
    logits. A critic's own loss() and nothing else -- a Policy prices with
    value()/expectation(). `target=True` falls back to the live net when there
    is no target network, so callers never have to ask."""
    net = self.q_target if (target and self.use_target) else self.q_function
    return net(s, a)

  def raw_pair(self, s, a, next_s, next_a):
    """Both batches through the network in **one** forward, returned as the two
    un-reduced halves. This is CrossQ's trick (Bhatt et al., ICLR 2024): with a
    batch-statistic normalization inside, computing them separately lets the
    bootstrap drift off-distribution from the prediction it trains -- which is
    the instability target networks exist to damp. Sharing the moments removes
    the need for the target net rather than papering over its absence."""
    batch = s.shape[0]
    joint = self.raw(jnp.concatenate([s, next_s], axis=0),
                     jnp.concatenate([a, next_a], axis=0))
    return joint[..., :batch, :], joint[..., batch:, :]

  # ---- reductions: raw -> scalar, kept separate from the forward pass -------

  def reduce_expectation(self, raw):
    """E[Z] from an un-reduced prediction."""
    raise NotImplementedError

  def reduce_value(self, raw):
    """The price a Policy improves against -- the same as the expectation
    unless the subclass reduces with a spread."""
    return self.reduce_expectation(raw)

  def expectation(self, s, a, target: bool = False):
    return self.reduce_expectation(self.raw(s, a, target=target))

  def value(self, s, a, target: bool = False):
    return self.reduce_value(self.raw(s, a, target=target))

  def create_optimizer(self) -> dict:
    return {"q": optax.adam(self.lr)}

  # ------------------------------------------------------------------
  # Regression
  # ------------------------------------------------------------------

  def bootstrap(self, next_obs, next_act) -> Bootstrap:
    """The target net's next-state prediction, for a Policy to add its entropy
    bonus to before handing it to bellman_target()."""
    return self.bootstrap_from_raw(self.raw(next_obs, next_act, target=True))

  def bootstrap_from_raw(self, next_raw) -> Bootstrap:
    """A Bootstrap from an already-computed next-state prediction. Overridden by
    a critic that transports its own representation rather than a scalar."""
    return Bootstrap(jax.lax.stop_gradient(self.reduce_value(next_raw)))

  def bellman_target(self, reward, term, bootstrap: Bootstrap):
    reward = reward.reshape(-1, 1)
    term   = term.reshape(-1, 1).astype(reward.dtype)
    return reward + (1.0 - term) * self.gamma * (bootstrap.pred + bootstrap.shift)

  def terminal_target(self, value):
    """`value` itself, lifted into this critic's target representation -- the
    Bellman target of a transition that doesn't bootstrap."""
    return value

  def loss_from_raw(self, raw, target, losses: dict) -> None:
    """Regress an already-computed prediction onto `target`, in whatever
    representation this critic uses."""
    raise NotImplementedError

  def loss(self, obs, action, target, losses: dict) -> None:
    self.loss_from_raw(self.raw(obs, action), target, losses)

  def td_loss(self, obs, action, next_obs, next_action, reward, term,
              losses: dict, shift=0.0) -> None:
    """The whole TD update: price both batches, transport one step, regress.

    Which of the two paths runs is the critic's business, not the Policy's --
    that is the point of having it here. With a target network the prediction
    and the bootstrap are separate forwards through separate weights; without
    one they are a single forward through the live weights (see `raw_pair`), so
    any batch-statistic normalization sees both batches as one population.

    `shift` is the Policy's per-sample value shift, e.g. an entropy bonus."""
    if self.use_target:
      bootstrap = self.bootstrap(next_obs, next_action) + shift
      self.loss(obs, action, self.bellman_target(reward, term, bootstrap), losses)
      return

    pred_raw, next_raw = self.raw_pair(obs, action, next_obs, next_action)
    bootstrap = self.bootstrap_from_raw(next_raw) + shift
    self.loss_from_raw(pred_raw, self.bellman_target(reward, term, bootstrap), losses)


# ----------------------------------------------------------------------
# Ensemble critics
# ----------------------------------------------------------------------


class EnsembleCriticBase(QCriticBase):
  """Shared __init__/loss for the reduction variants below -- `n_models`
  independent MLP heads, all regressed to the same scalar Bellman target
  (epistemic disagreement across heads, not a return distribution). Only
  `value()`'s reduction differs between them. Not instantiated directly."""

  def __init__(self, env_stats, hidden_dim: int, layers: int, n_models: int,
               norm: str = None, norm_kwargs: dict = None, dropout: float = 0.0, *,
               rngs: nnx.Rngs, **kwargs):
    q_function = _EnsembleQ(n_models,
                            env_stats["obs_dim"] + env_stats["act_dim"],
                            hidden_dim, layers, norm=norm, dropout=dropout,
                            norm_kwargs=norm_kwargs, rngs=rngs)
    super().__init__(env_stats, q_function, rngs=rngs, **kwargs)
    self.n_models = n_models

  def reduce_expectation(self, raw):
    return raw.astype(jnp.float32).mean(axis=0)

  def loss_from_raw(self, raw, target, losses: dict) -> None:
    pred = raw.astype(jnp.float32)
    tgt  = jax.lax.stop_gradient(target).astype(jnp.float32)
    losses["q"] = ((pred - jnp.broadcast_to(tgt, pred.shape)) ** 2).mean()

    self.metrics.dist("q", pred.mean(axis=0))
    self.metrics.dist("q_target", tgt)
    self.metrics.ensemble("ensemble", pred, rho=getattr(self, "rho", 0.0))
    self.target_gap()


class _EnsembleQ(nnx.Module):
  """The ensemble itself, wrapped so `q_function(s, a)` reads the same as a
  single-head critic's does."""

  def __init__(self, n_models: int, in_dim: int, hidden_dim: int, layers: int,
               norm: str = None, norm_kwargs: dict = None, dropout: float = 0.0,
               *, rngs: nnx.Rngs):
    self.ensemble = Ensemble(n_models, MLP, input_dim=in_dim, output_dim=1,
                             latent_dim=hidden_dim, hidden_layers=layers,
                             norm=norm, dropout=dropout, rngs=rngs, **(norm_kwargs or {}))

  def __call__(self, s, a):
    return self.ensemble(jnp.concatenate([s, a], axis=-1))


@register_component
class MeanStdEnsembleCritic(EnsembleCriticBase):
  """Q(s,a) critic backed by `n_models` independent MLP heads; spread =
  epistemic uncertainty, pessimism = mean - rho*std."""

  def __init__(self, env_stats, hidden_dim: int, layers: int, n_models: int, *,
               rho: float, **kwargs):
    super().__init__(env_stats, hidden_dim, layers, n_models, **kwargs)
    self.rho = float(rho)

  def reduce_value(self, raw):
    raw = raw.astype(jnp.float32)
    var = raw.var(axis=0)
    return raw.mean(axis=0) - self.rho * jnp.sqrt(jnp.maximum(var, 1e-8))


@register_component
class MeanEnsembleCritic(EnsembleCriticBase):
  """Same ensemble, reduced by plain mean -- no pessimism, so value() is the
  ensemble expectation and there is no `rho` to set."""


@register_component
class MinEnsembleCritic(EnsembleCriticBase):
  """Same ensemble, reduced by TD3-style min-over-heads -- its pessimism has no
  scale to tune, so there is no `rho` here either."""

  def reduce_value(self, raw):
    return raw.astype(jnp.float32).min(axis=0)


# ----------------------------------------------------------------------
# HL-Gauss
# ----------------------------------------------------------------------


@register_component
class HLGaussCritic(QCriticBase):
  """HL-Gauss (Farebrother et al., 2024, "Stop Regressing"): scalar regression
  recast as classification. A single MLP head predicts logits over `num_bins`
  equal-width bins spanning [v_min, v_max]; the *scalar* Bellman target is
  spread over them by integrating N(target, sigma^2) across each bin, and the
  loss is cross-entropy against that.

  Unlike a distributional critic this is not a distributional Bellman operator
  -- it bootstraps a scalar E[Z] and only discretizes at the very end -- so it
  has no return distribution to be risk-averse about and takes no spread
  coefficient. `sigma_to_bin_ratio` is the paper's parameterization: sigma is
  that multiple of the bin width, and ~0.75 is what it recommends."""

  def __init__(self, env_stats, hidden_dim: int, layers: int, num_bins: int,
               sigma_to_bin_ratio: float, v_min: float = None, v_max: float = None,
               norm: str = None, norm_kwargs: dict = None, dropout: float = 0.0,
               *, rngs: nnx.Rngs, **kwargs):
    # Read off the argument rather than self.env_stats: the support has to
    # exist before super().__init__().
    v_min = (env_stats or {}).get("return_min") if v_min is None else v_min
    v_max = (env_stats or {}).get("return_max") if v_max is None else v_max
    if v_min is None or v_max is None:
      raise ValueError(
        "HLGaussCritic's support is fixed and must be known at construction time, but neither "
        f"config/models.yaml (v_min/v_max on the slot) nor this env's EnvStats "
        f"(return_min/return_max, declared via Environment.stats) supplies it -- got "
        f"v_min={v_min!r}, v_max={v_max!r}."
      )

    q_function = _SingleHeadQ(env_stats["obs_dim"] + env_stats["act_dim"],
                              num_bins, hidden_dim, layers, norm=norm, dropout=dropout,
                              norm_kwargs=norm_kwargs, rngs=rngs)
    super().__init__(env_stats, q_function, rngs=rngs, **kwargs)

    self.v_min, self.v_max = float(v_min), float(v_max)
    self.num_bins  = int(num_bins)
    self.bin_width = (self.v_max - self.v_min) / num_bins
    self.sigma     = sigma_to_bin_ratio * self.bin_width

    edges = jnp.linspace(self.v_min, self.v_max, num_bins + 1, dtype=jnp.float32)
    self.edges   = Buffer(edges)
    self.centres = Buffer((edges[:-1] + edges[1:]) / 2)

  def _to_histogram(self, value):
    """A scalar target -> the categorical it is regressed onto: the mass
    N(value, sigma^2) puts in each bin, renormalized by the total that landed
    in-support so the tails clipped at either edge don't shrink the target.

    The clip is load-bearing, not cosmetic. sigma is a fraction of one bin, so
    a target even a few bins outside the support puts *no* representable mass
    inside it -- every bin underflows to 0, and an all-zero target is a
    cross-entropy term with no gradient, i.e. a transition silently dropped
    from the loss. Clipping first pins it to the edge bin instead."""
    value = jnp.clip(value.reshape(-1, 1), self.v_min, self.v_max)
    z     = (self.edges[...] - value) / (self.sigma * math.sqrt(2.0))
    cdf   = 0.5 * (1.0 + jax.scipy.special.erf(z))
    mass  = cdf[:, 1:] - cdf[:, :-1]
    return mass / jnp.maximum(mass.sum(axis=-1, keepdims=True), 1e-8)

  def probs(self, s, a, target: bool = False):
    return jax.nn.softmax(self.raw(s, a, target=target).astype(jnp.float32), axis=-1)

  def reduce_expectation(self, raw):
    probs = jax.nn.softmax(raw.astype(jnp.float32), axis=-1)
    return (probs * self.centres[...]).sum(axis=-1, keepdims=True)

  def bellman_target(self, reward, term, bootstrap: Bootstrap):
    return self._to_histogram(super().bellman_target(reward, term, bootstrap))

  def terminal_target(self, value):
    return self._to_histogram(value)

  def loss_from_raw(self, raw, target, losses: dict) -> None:
    pred_logits = raw.astype(jnp.float32)
    tgt = jax.lax.stop_gradient(target)
    losses["q"] = -(tgt * jax.nn.log_softmax(pred_logits, axis=-1)).sum(axis=-1).mean()

    probs = jax.nn.softmax(pred_logits, axis=-1)
    self.metrics.dist("q", (probs * self.centres[...]).sum(axis=-1, keepdims=True))
    self.metrics.dist("q_target", (tgt * self.centres[...]).sum(axis=-1, keepdims=True))
    self.target_gap()


class _SingleHeadQ(nnx.Module):
  def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, layers: int,
               norm: str = None, norm_kwargs: dict = None, dropout: float = 0.0,
               *, rngs: nnx.Rngs):
    self.net = MLP(in_dim, out_dim, hidden_dim, layers, layernorm=False, norm=norm,
                   dropout=dropout, rngs=rngs, **(norm_kwargs or {}))

  def __call__(self, s, a):
    return self.net(jnp.concatenate([s, a], axis=-1))
