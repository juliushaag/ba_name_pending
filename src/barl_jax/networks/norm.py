"""Normalization layers for MLP, including the BatchRenorm that CrossQ relies on.

`layer` is the framework default and the only one with no train/eval split.
`batch` and `batchrenorm` carry running statistics as `nnx.BatchStat`, mutated
in place during training and used in place of the batch moments at eval --
`nnx.Module.eval()` / `.train()` flip that recursively via the
`use_running_average` attribute, which is why it is spelled exactly that way
here."""

import jax
import jax.numpy as jnp
from flax import nnx


class BatchRenorm(nnx.Module):
  """Batch Renormalization (Ioffe, 2017) over a (B, C) activation.

  Plain BatchNorm normalizes by the *minibatch* moments at train time and by
  the running ones at eval time, so the function the critic learns is not the
  function it is later evaluated with -- which is what makes BatchNorm
  notoriously unstable in off-policy RL, where the same network is queried on a
  fresh single-transition batch every env step. BatchRenorm closes that gap: it
  still normalizes with the batch moments, then applies an affine correction
  (r, d) that maps them onto the running ones, with r/d clipped to a window
  that opens over `warmup_steps`. Training starts as BatchNorm and converges to
  the eval-time normalization, so train and eval agree.

  This is the normalization CrossQ (Bhatt et al., ICLR 2024) relies on to drop
  target networks entirely.

  `momentum` is the weight given to the *batch* moment when updating the
  running one (torch's `lerp` convention), so 0.01 means a slow running
  average -- note flax's own BatchNorm takes the opposite convention, where
  0.99 means the same thing."""

  # epsilon is dime_am's 1e-3, not flax's 1e-5: it enters the r/d ratios as well
  # as the denominator, so the two are not interchangeable here.
  def __init__(self, num_features: int, *, momentum: float = 0.01, epsilon: float = 1e-3,
               r_max: float = 3.0, d_max: float = 5.0, warmup_steps: int = 100_000,
               use_running_average: bool = False, dtype=jnp.float32, rngs: nnx.Rngs = None):
    self.num_features = num_features
    self.dtype        = dtype
    self.momentum     = float(momentum)
    self.epsilon      = float(epsilon)
    self.r_max        = float(r_max)
    self.d_max        = float(d_max)
    self.warmup_steps = int(warmup_steps)
    # The name matters: nnx.Module.eval()/.train() set exactly this attribute.
    self.use_running_average = use_running_average

    self.scale = nnx.Param(jnp.ones((num_features,), jnp.float32))
    self.bias  = nnx.Param(jnp.zeros((num_features,), jnp.float32))

    self.mean  = nnx.BatchStat(jnp.zeros((num_features,), jnp.float32))
    self.var   = nnx.BatchStat(jnp.ones((num_features,), jnp.float32))
    self.steps = nnx.BatchStat(jnp.zeros((), jnp.float32))

  def _limits(self):
    """r and d open linearly from (1, 0) -- i.e. exactly BatchNorm -- to
    (r_max, d_max) over warmup_steps, so early training isn't corrected towards
    running statistics that are still meaningless."""
    t = jnp.minimum(self.steps[...] / self.warmup_steps, 1.0)
    return 1.0 + t * (self.r_max - 1.0), t * self.d_max

  def _apply(self, x, a, b):
    """The whole normalization folded into one per-channel affine.

    Algebraically identical to `((x - mean)/std * r + d) * scale + bias`, but
    `a` and `b` are `(C,)` and computed in float32, so under a reduced compute
    dtype the statistics never round-trip a full (B, C) activation through
    float32 -- and it is one fused multiply-add over the big tensor instead of
    four elementwise passes."""
    return x * a.astype(x.dtype) + b.astype(x.dtype)

  def __call__(self, x, use_running_average: bool = None):
    use_running = self.use_running_average if use_running_average is None else use_running_average

    if use_running:
      a = self.scale[...] * jax.lax.rsqrt(self.var[...] + self.epsilon)
      return self._apply(x, a, self.bias[...] - self.mean[...] * a)

    # Accumulated in float32 whatever `x` is: a bf16 sum over a batch this wide
    # loses the moments outright, and the running statistics are float32.
    batch_mean = x.mean(axis=0, dtype=jnp.float32)
    batch_var  = x.var(axis=0, dtype=jnp.float32)
    batch_std  = jnp.sqrt(batch_var + self.epsilon)

    running_std = jnp.sqrt(self.var[...] + self.epsilon)
    r_max, d_max = self._limits()
    # stop_gradient: r and d are a correction *towards* the running statistics,
    # not a path the loss should push back through.
    r = jax.lax.stop_gradient(jnp.clip(batch_std / running_std, 1.0 / r_max, r_max))
    d = jax.lax.stop_gradient(
      jnp.clip((batch_mean - self.mean[...]) / running_std, -d_max, d_max))

    self.mean[...]  = self.mean[...] + self.momentum * (jax.lax.stop_gradient(batch_mean) - self.mean[...])
    self.var[...]   = self.var[...] + self.momentum * (jax.lax.stop_gradient(batch_var) - self.var[...])
    self.steps[...] = self.steps[...] + 1

    a = r * self.scale[...] / batch_std
    return self._apply(x, a, self.bias[...] + d * self.scale[...] - batch_mean * a)


def BatchNorm(num_features: int, *, momentum: float = 0.01, epsilon: float = 1e-5,
              use_running_average: bool = False, dtype=jnp.float32,
              rngs: nnx.Rngs, **_) -> nnx.BatchNorm:
  """flax's BatchNorm, with `momentum` in the same sense as BatchRenorm above
  (weight on the batch moment) rather than flax's own decay convention."""
  return nnx.BatchNorm(num_features, momentum=1.0 - momentum, epsilon=epsilon,
                       use_running_average=use_running_average,
                       dtype=dtype, param_dtype=jnp.float32, rngs=rngs)


def LayerNorm(num_features: int, *, epsilon: float = 1e-5, dtype=jnp.float32,
              rngs: nnx.Rngs, **_) -> nnx.LayerNorm:
  # torch's LayerNorm eps default is 1e-5; flax's is 1e-6.
  return nnx.LayerNorm(num_features, epsilon=epsilon,
                       dtype=dtype, param_dtype=jnp.float32, rngs=rngs)


NORMS = {
  "none":        None,
  "layer":       LayerNorm,
  "batch":       BatchNorm,
  "batchrenorm": BatchRenorm,
}


def make_norm(kind: str, num_features: int, *, rngs: nnx.Rngs, **kwargs):
  """`kind` -> a norm module, or None for "none"."""
  if kind not in NORMS:
    raise KeyError(f"{kind!r} is not a known normalization: {sorted(NORMS)}")
  factory = NORMS[kind]
  return None if factory is None else factory(num_features, rngs=rngs, **kwargs)
