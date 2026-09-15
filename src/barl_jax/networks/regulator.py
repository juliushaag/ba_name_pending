import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from barl_jax.components import Component
from barl_jax.optimization import DualOptim, DualOptimConfig, adam
from barl_jax.registration import register_component


@dataclass(frozen=True)
class AutotuneRegulatorConfig(DualOptimConfig):
  class_name: str = "AutotuneRegulator"
  alpha_init: float = 1.0
  tune: bool = True
  min_alpha: float = 1.0e-7
  max_alpha: float = 500.0


@register_component
class AutotuneRegulator(DualOptim, Component, nnx.Module):
  """A Lagrange multiplier tuned to hold some measured quantity at `target`
  (SAC's entropy temperature, and the same machinery for any other dual). The
  loss is `log_alpha * (value - target)`, whose gradient wrt log_alpha is
  `value - target`, so under descent alpha *falls* while the measured value sits
  above target and rises while it sits below -- the sign canonical SAC needs, and
  the opposite of what this docstring claimed before.

  `target` is always the owning Policy's own computed value, never a
  config/tunable -- so it stays a real keyword argument and is not a `Config`
  declaration. It is either a plain float (one alpha for the whole chunked plan -- SAC/AMDP/
  DIME/SDAC's default, `target=self.chunk_size * self.action_dim *
  self.entropy_scale`, entropy being joint over the whole plan) or a
  `(chunk_size,)` array (one alpha per chunk position -- `MaxEntRLPolicy`'s
  `alpha_granularity: per_action`, `target=self.action_dim * self.entropy_scale`
  per position). `log_alpha` just takes `target`'s shape, so this class carries
  no granularity switch of its own -- the owning Policy decides by what shape
  of `target` it passes in, and `value()`/`loss()` broadcast either way.

  The optimizer knobs come from the `DualOptim` mixin (a higher LR, unclipped)."""

  config_type = AutotuneRegulatorConfig

  # False pins alpha at `alpha_init` -- `components.<slot>.tune: false` in
  # config. `create_optimizer()` still declares this dual's key (train_step's
  # loss dict must match `create_optimizer()` exactly regardless), but backed
  # by `optax.set_to_zero()` instead of Adam, so `log_alpha` still gets a
  # gradient computed and logged, it is just never applied -- alpha is a
  # config-set constant, not a silently-frozen-at-whatever-init-happened-to-be
  # one.

  # A dual has no tier ladder -- `kind` exists so the resolver can tell a sized
  # component from a non-sized one.
  kind = "dual"

  # max_alpha/min_alpha are accepted because the config layers set them, but
  # value() does not clamp -- dime_am's dual is unbounded too, and pinning
  # alpha at a floor injects a constant entropy bonus into the Bellman target
  # that a near-zero-reward env cannot outvote. Declared as hparams, not dead
  # state, so a bound can be reinstated by an override without a code change.

  def __init__(self, env_stats=None, config=None, *, target,
               rngs: nnx.Rngs = None, **kwargs):
    super().__init__(env_stats, config, **kwargs)

    target = jnp.asarray(target, jnp.float32)
    # 0-dim, not shape (1,), for a scalar target: the torch original carried a
    # singleton dimension that made every product with alpha non-scalar, so a
    # loss which did not happen to end in .mean() came out shaped (1,) and
    # jax.grad rejected it. A (chunk_size,) target gives log_alpha the same
    # shape -- one Lagrange multiplier per chunk position instead of one for
    # the whole plan.
    self.log_alpha = nnx.Param(jnp.full(target.shape, math.log(config.alpha_init), jnp.float32))

    # A bare Python float for the scalar case (unchanged from before -- kept
    # out of the pytree entirely), a plain jnp array for the vector case.
    self.target_value = float(target) if target.ndim == 0 else target

  def value(self):
    return jnp.exp(self.log_alpha[...])

  def loss(self, value) -> dict:
    value = jax.lax.stop_gradient(value)
    alpha = self.value()
    target = self.target_value
    # Elementwise product before the mean, not `alpha * (value - target).mean()`:
    # identical for a scalar alpha (a scalar factors out of a mean either way),
    # but for a (chunk_size,) alpha this is what gives log_alpha_i a gradient
    # from position i's own discrepancy rather than from the discrepancy
    # averaged across every position.
    loss = (alpha * (value - target)).mean()

    alpha_sg = jax.lax.stop_gradient(alpha)
    log = self.metrics.dist if alpha_sg.ndim else self.metrics.scalar
    log("alpha", alpha_sg)
    log("target", target)
    self.metrics.dist("value", value)
    self.metrics.dist("value_scaled", value * alpha_sg)
    return {f"{self.slot}_alpha": loss}

  def create_optimizer(self) -> dict:
    # Keyed by slot, so a policy's two regulators stay distinct without a
    # `name` kwarg.
    c = self.config
    tx = (adam(c.lr, c.adam_b1, c.grad_clip, c.grad_clip_mode)
          if c.tune else optax.set_to_zero())
    return {f"{self.slot}_alpha": tx}
