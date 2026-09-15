import math
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Component
from barl_jax.registration import register_component


class Squash(Component, nnx.Module, ABC):
  """Maps an actor's unconstrained output to a canonical [-1, 1]-per-dimension
  action and back (SAC's tanh, AMDP's erf, ...). An actor selects one with its
  own `squash_cls` setting (`ErfSquash` sets its `erf_k` from the noise
  schedule)."""

  kind = "squash"

  def __init__(self, env_stats=None, config=None, **kwargs):
    """Absorbs kwargs a concrete Squash doesn't need -- e.g. AMDP always
    passes `erf_k=...` regardless of which class the squash slot resolves to."""
    super().__init__(env_stats, config)

  @abstractmethod
  def __call__(self, x):
    """Raw actor output -> canonical [-1, 1]-per-dimension action."""
    ...

  @abstractmethod
  def inverse(self, y):
    """Canonical [-1, 1]-per-dimension action -> raw actor output."""
    ...

  @abstractmethod
  def log_abs_det_jacobian(self, x, y=None):
    """log|dy/dx|, elementwise; `y` accepted to avoid a redundant forward() if
    already computed."""
    ...


@register_component
class TanhSquash(Squash):
  """Standard tanh squash (Haarnoja et al., 2018 SAC appendix C)."""

  def __call__(self, x):
    return jnp.tanh(x)

  def inverse(self, y, eps: float = 1e-6):
    return jnp.arctanh(jnp.clip(y, -1 + eps, 1 - eps))

  def log_abs_det_jacobian(self, x, y=None, eps: float = 1e-6):
    # The stable form. log(1 - tanh(x)^2) saturates to -inf for |x| >~ 9 in
    # float32, which is what makes naive tanh log-probs NaN out in SAC.
    return 2.0 * (math.log(2.0) - x - jax.nn.softplus(-2.0 * x))


@register_component
class ErfSquash(Squash):
  """erf squash (Thilges et al. 2026 Eq. 14), scaled by `erf_k` -- a fixed
  (not learned) function of the owning actor's noise schedule, chosen so
  this squash's log-Jacobian cancels out of AMDP's direct-score regression."""

  def __init__(self, env_stats=None, config=None, *, erf_k: float, **kwargs):
    super().__init__(env_stats, config, **kwargs)
    # A plain Python float, not a Variable: it is derived from the actor's
    # noise schedule at construction time and never trained or checkpointed.
    self.erf_k = float(erf_k)

  def derived(self) -> dict:
    return {"erf_k": self.erf_k}

  def __call__(self, x):
    return jnp.clip(jax.scipy.special.erf(self.erf_k * x), -1.0, 1.0)

  def inverse(self, y, eps: float = 1e-6):
    return jax.scipy.special.erfinv(jnp.clip(y, -1 + eps, 1 - eps)) / self.erf_k

  def log_abs_det_jacobian(self, x, y=None):
    return (math.log(2 / math.sqrt(math.pi)) + math.log(self.erf_k) - (self.erf_k * x) ** 2)
