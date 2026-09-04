import math

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.config.toolkit import ConfigMixin
from barl_jax.hparam import Config, HParam
from barl_jax.networks.ensemble import Ensemble
from barl_jax.networks.mlp import MLP, MLP_PRESETS
from barl_jax.networks.optim import adam, CriticOptim
from barl_jax.policy import TARGET_SUFFIX, Buffer, Critic, frozen


class CriticLossConfig:
  gamma             = HParam(0.99)
  pessimistic_target = Config(False)
  polyak_tau        = HParam(0.0)
  target_delay      = Config(1)


class CriticDistConfig:
  num_atoms          = Config(101)
  num_bins           = Config(151)
  sigma_to_bin_ratio = Config(0.75)
  entropy_coef       = HParam(0.0)
  rho                = HParam(0.0)


class Polyak:
  polyak_tau = HParam(0.005)


class QCriticBase(CriticLossConfig, CriticOptim, Critic):
  noise_conditioned = False

  def __init__(self, env_stats, config: ConfigMixin, q_function, **kwargs):
    super().__init__(env_stats, config, **kwargs)

    self.q_function = q_function

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

  def bellman_target(self, reward, term, bootstrap, shift=0.0, horizon=None):
    reward = reward.reshape(-1, 1)
    term   = term.reshape(-1, 1).astype(reward.dtype)
    if horizon is None:
      horizon = jnp.ones_like(reward)
    else:
      horizon = jnp.asarray(horizon, reward.dtype).reshape(-1, 1)
    discount = jnp.power(jnp.asarray(self.gamma, reward.dtype), horizon)
    return reward + (1.0 - term) * discount * (bootstrap + shift)

  def target_from_raw(self, raw):
    return jax.lax.stop_gradient(raw.mean(axis=0))

  def regression_loss(self, raw, target):
    raise NotImplementedError

  def record(self, raw, target) -> None:
    raise NotImplementedError

  def loss_from_raw(self, raw, target, losses: dict) -> None:
    losses["critic"] = self.regression_loss(raw, target)
    self.record(raw, target)

  def loss(self, obs, action, next_obs, next_action, reward, term,
              losses: dict, shift=0.0, horizon=None) -> None:
    if self.has_target:
      pred_raw = self.raw(obs, action)
      next_raw = self.target_raw(next_obs, next_action)
    else:
      pred_raw, next_raw = self.raw_batches((obs, action), (next_obs, next_action))
    target = self.bellman_target(reward, term, self.bootstrap_from_raw(next_raw), shift,
                                 horizon=horizon)
    self.loss_from_raw(pred_raw, target, losses)


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


class CategoricalCriticBase(QCriticBase, CriticDistConfig):
  mlp_preset = Config("crossq", style=MLP_PRESETS)

  def __init__(self, env_stats, config: ConfigMixin, q_function, support, **kwargs):
    super().__init__(env_stats, config, q_function, **kwargs)

    self.support = Buffer(support)
    self.n_models = config.get("n_models", int)

  def probs(self, raw):
    return jax.nn.softmax(raw.astype(jnp.float32), axis=-1)

  def expectation_from_raw(self, raw):
    return (self.probs(raw) * self.support[...]).sum(axis=-1, keepdims=True)

  def reduce_value(self, raw):
    exp = self.expectation_from_raw(raw).astype(jnp.float32)
    std = jnp.sqrt(jnp.maximum(exp.var(axis=0), 1e-8))
    return exp.mean(axis=0) - self.rho * std

  def target_from_raw(self, raw):
    return jax.lax.stop_gradient(self.probs(raw).mean(axis=0))

  def regression_loss(self, raw, target):
    logp = jax.nn.log_softmax(raw.astype(jnp.float32), axis=-1)
    tgt  = jax.lax.stop_gradient(target).astype(jnp.float32)

    reduce = lambda x: x.sum(axis=-1).mean(axis=-1).sum()

    loss = reduce(-(tgt * logp))
    if self.entropy_coef:
      loss += self.entropy_coef * reduce(jnp.exp(logp) * logp)
    return loss

  def record(self, raw, target) -> None:
    tgt     = jax.lax.stop_gradient(target).astype(jnp.float32)
    support = self.support[...]
    probs   = self.probs(raw)
    exp     = (probs * support).sum(axis=-1, keepdims=True)

    self.metrics.dist("critic", exp.mean(axis=0))
    self.metrics.dist("q_target", (tgt * support).sum(axis=-1, keepdims=True))
    self.metrics.ensemble("ensemble", exp, gap=self.pessimism_gap(raw))

    var = (probs * (support - exp) ** 2).sum(axis=-1)
    self.metrics.dist("return_std", jnp.sqrt(var).mean(axis=0))


def categorical_projection(next_probs, atoms, reward, term, gamma, shift, v_min, v_max,
                           horizon=None):
  num_atoms = atoms.shape[0]
  delta_z   = (v_max - v_min) / (num_atoms - 1)

  reward = reward.reshape(-1, 1)
  term   = term.reshape(-1, 1).astype(reward.dtype)
  if horizon is None:
    horizon = jnp.ones_like(reward)
  else:
    horizon = jnp.asarray(horizon, reward.dtype).reshape(-1, 1)
  discount = jnp.power(jnp.asarray(gamma, reward.dtype), horizon)
  tz = jnp.clip(reward + (1.0 - term) * discount * (atoms.reshape(1, -1) + shift),
                v_min, v_max)

  b = (tz - v_min) / delta_z
  l = jnp.clip(jnp.floor(b), 0, num_atoms - 1).astype(jnp.int32)
  u = jnp.clip(jnp.ceil(b),  0, num_atoms - 1).astype(jnp.int32)

  on_atom = (l == u)
  m_l = jnp.where(on_atom, next_probs, next_probs * (u.astype(b.dtype) - b))
  m_u = jnp.where(on_atom, 0.0,        next_probs * (b - l.astype(b.dtype)))

  proj = jnp.zeros_like(next_probs)
  batch_idx = jnp.arange(next_probs.shape[-2])[:, None]
  proj = proj.at[..., batch_idx, l].add(m_l)
  proj = proj.at[..., batch_idx, u].add(m_u)
  return proj
