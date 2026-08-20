"""The jitted gradient step, shared by the single-seed and multi-seed trainers.

It is a pure function of `(graphdef, state, batch, tiers)`, which is what makes
`jax.vmap` over a leading seed axis work: N independently-initialized policies
whose states are stacked step together in one call. See barl_jax/multiseed.py.
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from barl_jax.metrics import DEBUG, PRIMARY, DebugMetric, Metric, collect, recording
from barl_jax.policy import TRAINABLE


# ----------------------------------------------------------------------
# The jitted step
# ----------------------------------------------------------------------

# One backward pass over the whole parameter tree, partitioned by
# optax.multi_transform. The torch loop instead took one autograd.grad per loss
# key with retain_graph=True, restricting each to its own optimizer's params and
# silently discarding whatever leaked into the others; here that discarding is
# explicit (jax.lax.stop_gradient inside each train_step) and the backward runs
# once. `tiers` is static so an unarmed metric family is compiled away entirely
# -- at most three variants get built (none / primary / primary+debug).

@partial(jax.jit, static_argnums=(0, 3))
def train_step(graphdef, state, batch, tiers):
  policy, opt = nnx.merge(graphdef, state)

  def scalar_loss(policy):
    policy.record_batch(batch)
    losses = policy.train_step(batch)
    return sum(losses.values()), losses

  def loss_fn(policy):
    total, losses = scalar_loss(policy)
    sown, sown_debug = nnx.pop(policy, Metric, DebugMetric)
    return total, (losses, sown, sown_debug)

  with recording(*tiers):
    policy.before_step()
    (_, (losses, sown, sown_debug)), grads = nnx.value_and_grad(
      loss_fn, argnums=nnx.DiffState(0, TRAINABLE), has_aux=True)(policy)

  primary = collect(sown)
  debug   = collect(sown_debug)

  if tiers:
    labels = dict(nnx.to_flat_state(policy.optimizer_labels()))
    flat_grads = nnx.to_flat_state(grads)
    for key in losses:
      partition = [_raw(v).astype(jnp.float32) for path, v in flat_grads
                   if labels.get(path) == key]
      if PRIMARY in tiers:
        primary[policy.metrics_key(key, "loss")] = losses[key]
        primary[policy.metrics_key(key, "grad_norm")] = optax.tree.norm(partition)
      if DEBUG in tiers and partition:

        total   = sum(g.size for g in partition)
        zeros   = sum((g == 0).sum() for g in partition)
        nonfin  = sum((~jnp.isfinite(g)).sum() for g in partition)
        gmax    = jnp.max(jnp.stack([jnp.abs(g).max() for g in partition]))
        debug[policy.metrics_key(key, "grad_max")] = gmax
        debug[policy.metrics_key(key, "grad_frac_zero")] = zeros / total
        debug[policy.metrics_key(key, "grad_frac_nonfinite")] = nonfin / total

  opt.update(policy, grads)
  policy.after_step()

  return nnx.state((policy, opt)), losses, (primary, _pack(primary)), (debug, _pack(debug))


def _pack(metrics: dict):
  """The dict's values as one array, in **sorted key order**.

  Explicitly sorted rather than in insertion order: jit returns a dict rebuilt
  from a pytree treedef, and pytree flattening sorts dict keys -- so the dict
  the caller receives is not ordered the way this one was built. Both sides
  agreeing on `sorted()` is what keeps the keys attached to their values."""
  if not metrics:
    return jnp.zeros((0,))
  return jnp.stack([jnp.asarray(metrics[k]) for k in sorted(metrics)])


def _raw(v):
  """The array behind an nnx Variable (or the array itself)."""
  return v[...] if hasattr(v, "__getitem__") and not isinstance(v, jax.Array) else v


