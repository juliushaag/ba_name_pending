from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from barl_jax.policy import Buffer


CLIP_MODES = ("global_norm", "element", "none")


def grad_clip_transform(grad_clip: float, grad_clip_mode: str):
  """The clip a component puts ahead of its Adam.

  The two modes are different operations, not two strengths of one, which is why
  this is a component's choice and not a run-level number. `global_norm`
  rescales the whole gradient to `grad_clip` and so preserves its direction
  exactly -- and since Adam normalizes per coordinate anyway, that leaves an
  outlier sample free to set the update's direction while the length it removed
  never mattered. `element` bounds each coordinate at +-`grad_clip`, which is
  what actually attenuates one sample's contribution. dime_am clips its control
  net the element-wise way (`cont_time_policy.py`: `chain(zero_nans(), clip(1.0),
  adam(lr))`) and leaves critic and dual unclipped."""
  if grad_clip_mode not in CLIP_MODES:
    raise KeyError(f"grad_clip_mode {grad_clip_mode!r} is not one of {', '.join(CLIP_MODES)}")

  return {"global_norm": lambda: optax.clip_by_global_norm(float(grad_clip)),
          "element":     lambda: optax.clip(float(grad_clip)),
          "none":        optax.identity}[grad_clip_mode]()


def adam(lr: float, adam_b1: float, grad_clip: float, grad_clip_mode: str):
  """This component's optimizer: NaN guard, its own gradient clip, then Adam.

  `zero_nans` goes first so an infinity cannot drag a global norm to inf and
  scale the whole update to nothing. It hides nothing that was being reported:
  the `*/grad_frac_nonfinite` metric is taken from the raw gradient."""
  return optax.chain(optax.zero_nans(),
                     grad_clip_transform(grad_clip, grad_clip_mode),
                     optax.adam(float(lr), b1=float(adam_b1)))


class EveryKState(NamedTuple):
  count: jax.Array
  inner: optax.OptState


def every_k(inner: optax.GradientTransformation, k: int) -> optax.GradientTransformation:
  """Run `inner` on every k-th call and emit zero updates in between.

  Neither of the two obvious alternatives is equivalent. `optax.MultiSteps`
  *accumulates* the skipped gradients and applies their average, where a delayed
  update simply drops them -- the parameter does not move on an off step and the
  gradient computed there is discarded. Zeroing the loss instead would still let
  Adam move on its accumulated momentum. So the inner state is held frozen on off
  steps too, making this exactly "don't call the optimizer this step"."""
  k = int(k)
  if k <= 1:
    return inner

  def init_fn(params):
    return EveryKState(jnp.zeros((), jnp.int32), inner.init(params))

  def update_fn(updates, state, params=None):
    stepped, inner_state = inner.update(updates, state.inner, params)
    count = state.count + 1
    due   = (count % k) == 0

    select = lambda new, old: jnp.where(due, new, old)
    return (jax.tree.map(lambda u: jnp.where(due, u, jnp.zeros_like(u)), stepped),
            EveryKState(count, jax.tree.map(select, inner_state, state.inner)))

  return optax.GradientTransformation(init_fn, update_fn)


class RunningMoments(nnx.Module):
  """EMA mean/std of a scalar stream, as non-trainable state.

  For algorithms that weight a denoising loss by a normalized Q: the scale of Q
  drifts over training, so a fixed temperature would mean something different at
  every point in the run."""

  def __init__(self, decay: float = 0.001, eps: float = 1e-6):
    self.decay = float(decay)
    self.eps   = float(eps)
    self.mean  = Buffer(jnp.zeros((), jnp.float32))
    self.var   = Buffer(jnp.ones((), jnp.float32))

  def update(self, value):
    value = jax.lax.stop_gradient(value)
    mean  = value.mean()
    self.mean[...] = self.mean[...] + self.decay * (mean - self.mean[...])
    self.var[...]  = self.var[...] + self.decay * (value.var() - self.var[...])

  def normalize(self, value):
    return (value - self.mean[...]) / (jnp.sqrt(jnp.maximum(self.var[...], 0.0)) + self.eps)
