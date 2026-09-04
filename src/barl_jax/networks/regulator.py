import math

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.configs import ConfigMixin
from barl_jax.hparam import Config
from barl_jax.networks.optim import adam, DualOptim
from barl_jax.networks.registry import register_component
from barl_jax.policy import Component


@register_component
class AutotuneRegulator(DualOptim, Component, nnx.Module):
  """A Lagrange multiplier tuned to hold some measured quantity at `target`
  (SAC's entropy temperature, and the same machinery for any other dual). The
  loss is `log_alpha * (value - target)`, whose gradient wrt log_alpha is
  `value - target`, so under descent alpha *falls* while the measured value sits
  above target and rises while it sits below -- the sign canonical SAC needs, and
  the opposite of what this docstring claimed before.

  `target` is always the owning Policy's own computed value (SAC/AMDP/DIME/SDAC
  all pass `target=self.action_dim * self.entropy_scale`), never a config/
  tunable -- so it stays a real keyword argument and is not an HParam.

  The optimizer knobs come from the `DualOptim` mixin (a higher LR, unclipped)."""

  alpha_init   = Config(1.0)

  # A dual has no size ladder -- `kind` exists so the resolver can tell a sized
  # component from a non-sized one.
  kind = "dual"

  # max_alpha/min_alpha are accepted because the config layers set them, but
  # value() does not clamp -- dime_am's dual is unbounded too, and pinning
  # alpha at a floor injects a constant entropy bonus into the Bellman target
  # that a near-zero-reward env cannot outvote. Declared as hparams, not dead
  # state, so a bound can be reinstated by an override without a code change.
  min_alpha    = Config(1.0e-7)
  max_alpha    = Config(500.0)

  def __init__(self, env_stats=None, config=None, *, target: float,
               rngs: nnx.Rngs = None, **kwargs):
    super().__init__(env_stats, config, **kwargs)

    # 0-dim, not shape (1,): the torch original carried a singleton dimension
    # that made every product with alpha non-scalar, so a loss which did not
    # happen to end in .mean() came out shaped (1,) and jax.grad rejected it.
    self.log_alpha = nnx.Param(jnp.asarray(math.log(self.alpha_init), jnp.float32))

    self.target_value = float(target)

  def value(self):
    return jnp.exp(self.log_alpha[...])

  def loss(self, value, losses: dict) -> None:
    value = jax.lax.stop_gradient(value)
    losses[f"{self.slot}_alpha"] = self.value() * (value - self.target_value).mean()

    alpha = jax.lax.stop_gradient(self.value())
    self.metrics.scalar("alpha", alpha)
    self.metrics.scalar("target", self.target_value)
    self.metrics.dist("value", value)
    self.metrics.dist("value_scaled", value * alpha)

  def create_optimizer(self) -> dict:
    # Keyed by slot, so a policy's two regulators stay distinct without a
    # `name` kwarg.
    return {f"{self.slot}_alpha": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}
