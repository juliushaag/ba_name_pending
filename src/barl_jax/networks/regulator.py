import math

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.optim import adam
from barl_jax.networks.registry import register_component
from barl_jax.policy import Component


@register_component
class AutotuneRegulator(Component, nnx.Module):
  """A Lagrange multiplier tuned to hold some measured quantity at `target`
  (SAC's entropy temperature, and the same machinery for any other dual). The
  loss is `log_alpha * (value - target)`, whose gradient wrt log_alpha is
  `value - target`, so under descent alpha *falls* while the measured value sits
  above target and rises while it sits below -- the sign canonical SAC needs, and
  the opposite of what this docstring claimed before."""

  def __init__(self, env_stats=None, *, lr: float, adam_b1: float,
               grad_clip: float, grad_clip_mode: str, alpha_init: float,
               max_alpha: float, min_alpha: float, target: float,
               rngs: nnx.Rngs = None, **kwargs):

    super().__init__(env_stats, **kwargs)

    self.lr, self.adam_b1 = float(lr), float(adam_b1)
    self.grad_clip, self.grad_clip_mode = float(grad_clip), str(grad_clip_mode)
    # 0-dim, not shape (1,): the torch original carried a singleton dimension
    # that made every product with alpha non-scalar, so a loss which did not
    # happen to end in .mean() came out shaped (1,) and jax.grad rejected it.
    self.log_alpha = nnx.Param(jnp.asarray(math.log(alpha_init), jnp.float32))

    self.target_value = float(target)
    # max_alpha/min_alpha are accepted because config/components/ sets them,
    # but value() does not clamp -- dime_am's dual is unbounded too, and pinning
    # alpha at a floor injects a constant entropy bonus into the Bellman target
    # that a near-zero-reward env cannot outvote. Kept as config, not as dead
    # state, so a bound can be reinstated without a config migration.
    self.min_alpha = float(min_alpha)
    self.max_alpha = float(max_alpha)

  def value(self):
    return jnp.exp(self.log_alpha[...])

  def loss(self, value, losses: dict) -> None:
    value = jax.lax.stop_gradient(value)
    losses[f"{self.slot}_alpha"] = self.log_alpha[...] * (value - self.target_value).mean()

    alpha = jax.lax.stop_gradient(self.value())
    self.metrics.scalar("alpha", alpha)
    self.metrics.scalar("target", self.target_value)
    self.metrics.dist("value", value)
    self.metrics.dist("value_scaled", value * alpha)

  def create_optimizer(self) -> dict:
    # Keyed by slot, so a policy's two regulators stay distinct without a
    # `name` kwarg.
    return {f"{self.slot}_alpha": adam(self.lr, self.adam_b1, self.grad_clip, self.grad_clip_mode)}
