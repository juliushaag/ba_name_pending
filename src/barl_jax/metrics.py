"""Diagnostics, as sown nnx intermediates.

`train_step` returns *only* losses; everything else is set as an attribute on the
module that computed it and collected with `nnx.pop` at the jit boundary.

- **Keys come from the module hierarchy.** A critic writing
  `self.metrics.dist("critic", pred)` lands at `critic/q/*` because the critic *is*
  at `policy.critic`; nothing names itself.
- **Nothing leaks a tracer.** `nnx.pop` hands the values back as a State the trainer
  returns from the traced step, which is how values leave a transform.

Collection is `nnx.pop`, not `nnx.capture`: `capture` instruments the module while
the function runs, and every nnx operation that flattens module state then dies
sorting the injected `__captures__` keys. That rules out any nnx transform inside
the traced region, which NC-LQL needs (`nnx.grad` through an ensemble critic) and
AMDP needs to freeze a critic. Recording is a plain attribute assignment rather
than `Module.sow()`: a metric never wants sow's `reduce_fn`, and sow warns outside
`nnx.capture()`.

Tiers are a *trace-time* decision: `recording()` is entered by the trainer around
the traced step, so an unarmed family is compiled away rather than skipped, and
there are at most three compiled variants (none / primary / primary+debug).

Values stay 0-dim device arrays; crossing to the host is the trainer's job -- one
`jax.device_get` per logged step, never a `float()` per key."""

import types
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

PRIMARY = "primary"
DEBUG   = "debug"


class Metric(nnx.Intermediate):
  """A sown primary-tier diagnostic. Its logged key is `<module path>/<name>`."""


class DebugMetric(nnx.Intermediate):
  """The debug tier. A separate Variable type rather than a flag, so
  `nnx.capture(fn, Metric, DebugMetric)` hands the two back already separated
  and `log_quiet` gets its own dict without any filtering."""


_TIER_TYPE = {PRIMARY: Metric, DEBUG: DebugMetric}


_ACTIVE: frozenset = frozenset()


@contextmanager
def recording(*tiers: str):
  """Arms the given tiers for the duration. Read at trace time, so an unarmed
  family costs nothing in the compiled step."""
  global _ACTIVE
  previous = _ACTIVE
  _ACTIVE = frozenset(tiers)
  try:
    yield
  finally:
    _ACTIVE = previous


def armed(*tiers: str) -> bool:
  """Whether any of `tiers` is recording -- a **trace-time** bool, so a caller
  can branch on it in python. With no argument: any tier at all."""
  return bool(_ACTIVE & (frozenset(tiers) if tiers else frozenset(_TIER_TYPE)))


def collect(state) -> dict:
  """A popped Metric State -> a flat `owner/quantity/reduction` dict, still on
  the device. The owner is the sowing module's path, so it needs no separate
  bookkeeping."""
  out = {}
  for path, variable in nnx.to_flat_state(state):
    key = "/".join(str(p) for p in path)
    out[key] = variable[...] if hasattr(variable, "__getitem__") else variable
  return out


def to_host(packed) -> dict:
  """`(dict_of_device_scalars, stacked_array)` from a jitted step -> a plain
  host dict. One transfer for the whole set: the dict supplies the key order,
  the array the values, and the dict's own device arrays are never touched."""
  metrics, stacked = packed
  if not metrics:
    return {}

  values = np.asarray(jax.device_get(stacked))
  # `_pack` stacks along axis 0 *inside* the step, so an enclosing vmap prepends
  # the seed axis and the array arrives as (seeds, metrics). Zipping the keys
  # against axis 0 without this would hand key 0 a whole row of unrelated
  # metrics and silently drop the rest -- which is exactly what it did.
  if values.ndim > 1:
    values = np.moveaxis(values, 0, -1)
  if len(values) != len(metrics):
    raise ValueError(f"packed {len(values)} metric rows for {len(metrics)} keys")
  # sorted(), matching train._pack -- see its docstring.
  return dict(zip(sorted(metrics), values))


class Recorder:
  """The `self.metrics` facade. Bound to a module rather than stored on it --
  it holds no state of its own, so there is nothing to thread through jit and
  nothing to keep in sync; `Component.metrics` rebuilds one per access.

  `prefix` is only for a module that owns several namespaces at the same point
  in the tree (a Policy's own `<algo>/` and `batch/`). A component leaves it
  unset and lets its path do the naming."""

  __slots__ = ("_owner", "_prefix")

  def __init__(self, owner: nnx.Module, prefix: str = None):
    self._owner  = owner
    self._prefix = prefix

  # Recording

  def _put(self, name: str, value, tier: str = PRIMARY):
    if tier not in _ACTIVE:
      return
    value = jnp.asarray(value)
    value = value.reshape(-1)[0] if value.size > 1 else value.reshape(())
    key = f"{self._prefix}/{name}" if self._prefix else name
    # A metric is sown as an attribute of its owning module, so its name must
    # not collide with that module's own API: `setattr` onto a property with no
    # setter raises (DDPMActor.explore_std used to), and onto a method it
    # silently shadows the method for the rest of the trace. Both are config-
    # name bugs; say so here rather than as whatever nnx or the next call makes
    # of them.
    member = type(self._owner).__dict__.get(key)
    if isinstance(member, (property, types.FunctionType, staticmethod, classmethod)):
      raise ValueError(
        f"metric {key!r} collides with {type(self._owner).__name__}.{key} -- sow it "
        f"under a name that is not already a method or property of the component")
    setattr(self._owner, key, _TIER_TYPE[tier](value))

  def scalar(self, name: str, value, tier: str = PRIMARY):
    self._put(name, value, tier)

  # Families -- one primary key per quantity, the expansion on the debug tier

  def dist(self, name: str, x):
    """Any per-sample scalar quantity: q values, advantages, rewards, entropies."""
    if not _ACTIVE:
      return
    x = jnp.asarray(x, dtype=jnp.float32)
    self._put(f"{name}/mean", x.mean())
    self._put(f"{name}/std", x.std() if x.size > 1 else jnp.zeros(()))
    self._put(f"{name}/min", x.min())
    self._put(f"{name}/max", x.max())

    # `or`: a value cannot be both above max-1e-3 and below min+1e-3.
    lo, hi = x.min() + 1e-3, x.max() - 1e-3
    self._put(f"{name}/edge_sat",
              jnp.logical_or(x > hi, x < lo).astype(jnp.float32).mean())

  def vec(self, name: str, x, limit: float = None):
    """Per-sample vectors: actions, score fields, flow velocities."""
    if not _ACTIVE:
      return
    x = jnp.asarray(x, dtype=jnp.float32)
    self._put(f"{name}/norm_mean", jnp.linalg.norm(x, axis=-1).mean())
    self._put(f"{name}/dim_std_mean",
              x.std(axis=0).mean() if x.shape[0] > 1 else jnp.zeros(()))
    if limit is not None:
      self._put(f"{name}/frac_at_limit",
                (jnp.abs(x) >= limit - 1e-4).astype(jnp.float32).mean())

  def ensemble(self, name: str, raw, gap=None):
    """Epistemic disagreement across an ensemble critic's `(n_models, B, 1)`
    heads -- distinct from a distributional critic's return spread, which goes
    through `dist("return_std")`, and from `q/std`, which is the batch spread
    of E[Z].

    `gap` is how far below E[Z] the price a Policy improves against sits, passed
    in rather than derived from a coefficient here: `mean - rho*std` is only one
    of the reductions in this codebase, and a `min` over heads has no rho to
    multiply -- deriving it that way logged a constant 0 for every categorical
    critic. See QCritic.pessimism_gap."""
    if not _ACTIVE:
      return
    std = jnp.asarray(raw, dtype=jnp.float32).std(axis=0)
    self._put(f"{name}/disagreement", std.mean())
    if gap is not None:
      self._put(f"{name}/pessimism_gap", jnp.asarray(gap, dtype=jnp.float32).mean())

  def gap(self, name: str, online_state, target_state):
    """Target-net drift, ||online - target|| / ||online||, over two
    `nnx.state(...)` trees. Accumulated per-leaf rather than over a flattened
    concatenation -- this runs on the primary tier and an ensemble critic's
    parameters are large enough that the copy would dominate the reduction."""
    if not _ACTIVE:
      return
    src = [jnp.asarray(v, dtype=jnp.float32) for v in jax.tree.leaves(online_state)]
    tgt = [jnp.asarray(v, dtype=jnp.float32) for v in jax.tree.leaves(target_state)]
    if not src:
      return
    num = sum(((p - q) ** 2).sum() for p, q in zip(src, tgt))
    den = sum((p ** 2).sum() for p in src)
    self._put(f"{name}/gap", jnp.sqrt(num / jnp.maximum(den, 1e-24)))
