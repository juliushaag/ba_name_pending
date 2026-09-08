
import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import TARGET_SUFFIX, Buffer, Critic, frozen
from barl_jax.config.toolkit import ConfigMixin
from barl_jax.hparam import Config
from barl_jax.networks.ensemble import Ensemble
from barl_jax.networks.mlp import MLP
from barl_jax.optimization import CriticOptim, adam


class CriticLossConfig:
  gamma             = Config(0.99)
  pessimistic_target = Config(False)
  polyak_tau        = Config(0.0)
  target_delay      = Config(1)


class Polyak:
  polyak_tau = Config(0.005)


def n_step_discount(gamma, horizon, terminated, dtype):
  """bootstrap_discount = (1 - terminated) * gamma**horizon, broadcast to (B, 1).

  The one piece of arithmetic `QCriticBase.bellman_target` and
  `categorical_projection` share -- everything downstream of it (what the
  discount multiplies into: a scalar bootstrap vs. a per-atom support) is
  representation-specific and stays at each call site.

  `horizon` is required: the executed plan length h, with no implicit
  fallback to 1. A chunk transition's horizon must always be known by the
  time it reaches this function -- K=1 is passed explicitly, not inferred
  from an absent argument."""
  horizon = jnp.asarray(horizon, dtype).reshape(-1, 1)
  terminated = terminated.reshape(-1, 1).astype(dtype)
  return (1.0 - terminated) * jnp.power(jnp.asarray(gamma, dtype), horizon)


class QCriticBase(CriticLossConfig, CriticOptim, Critic):
  learning_objective = "td"

  def __init__(self, env_stats, config: ConfigMixin, q_function, **kwargs):
    super().__init__(env_stats, config, **kwargs)

    self.q_function = q_function

    if self.polyak_tau > 0.0:
      if self.mlp.norm in ("batch", "batchrenorm"):
        raise ValueError(
          f"{type(self).__name__}: polyak_tau={self.polyak_tau} with norm={self.mlp.norm!r}. A target net "
          f"and CrossQ-style running-statistic normalization are alternative answers to the "
          f"same problem -- the target's separate forward pass removes the shared "
          f"[(s,a), (s',a')] batch that makes BatchRenorm worth having. Pick one: either "
          f"polyak_tau: 0.0, or norm: layer/none.")
      for name, module in self.target_modules().items():
        setattr(self, f"{name}{TARGET_SUFFIX}", nnx.clone(module))
      self.target_steps = Buffer(jnp.zeros((), jnp.int32))

  @property
  def has_target(self) -> bool:
    return self.polyak_tau > 0.0

  def target_modules(self) -> dict:
    return {"critic": self.q_function}

  def target_raw(self, s, a):
    net = self.critic_target if self.has_target else frozen(self.q_function)
    return jax.lax.stop_gradient(net(s, a))

  def after_step(self):
    if not self.has_target:
      return

    self.target_steps[...] += 1
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
    return raw

  def mean_expectation(self, raw):
    return self.expectation_from_raw(raw).astype(jnp.float32).mean(axis=0)

  def pessimism_gap(self, raw):
    return self.mean_expectation(raw) - self.reduce_value(raw).astype(jnp.float32)

  def expectation(self, s, a):
    return self.mean_expectation(self.raw(s, a))

  def create_optimizer(self) -> dict:
    return {"critic": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}

  def bootstrap_from_raw(self, next_raw):
    reduce = self.reduce_value if self.pessimistic_target else self.mean_expectation
    return jax.lax.stop_gradient(reduce(next_raw))

  def td_loss(self, obs, action, next_obs, next_action, chunk_return, terminated, horizon,
              entropy_bonus=0.0, *, key=None) -> dict:
    """chunk_return: (B,) accumulated discounted return over the executed
    horizon (already gamma-weighted per primitive step by the collector --
    this applies exactly one more discount, from t+h to t+0, to the
    bootstrap). terminated: (B,) true-termination flag (never truncation).
    horizon: (B,) executed plan length h, required. entropy_bonus: (B,1) or
    0.0, alpha*next_chunk_entropy added to the bootstrap before discounting;
    0.0 for variants with no target entropy (TD3-shaped / SDAC)."""
    if self.has_target:
      pred_raw = self.raw(obs, action)
      next_raw = self.target_raw(next_obs, next_action)
    else:
      pred_raw, next_raw = self.raw_batches((obs, action), (next_obs, next_action))
    target = self.bellman_target(chunk_return, terminated, horizon,
                                 self.bootstrap_from_raw(next_raw), entropy_bonus)
    loss = self.regression_loss(pred_raw, target)
    self.record(pred_raw, target)
    return {"critic": loss}

  def bellman_target(self, chunk_return, terminated, horizon, bootstrap, entropy_bonus=0.0):
    """td_target = chunk_return + bootstrap_discount * (bootstrap + entropy_bonus),
    bootstrap_discount = (1 - terminated) * gamma**horizon (n_step_discount)."""
    chunk_return = chunk_return.reshape(-1, 1)
    discount = n_step_discount(self.gamma, horizon, terminated, chunk_return.dtype)
    return chunk_return + discount * (bootstrap + entropy_bonus)

  def target_from_raw(self, raw):
    return jax.lax.stop_gradient(raw.mean(axis=0))

  def regression_loss(self, raw, target):
    raise NotImplementedError

  def record(self, raw, target) -> None:
    raise NotImplementedError


class ConcatQ(nnx.Module):
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
