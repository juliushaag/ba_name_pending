import jax
import jax.numpy as jnp
from flax import nnx


def polyak_step(net, net_target, steps, config):
  """Blend `net_target` towards `net` by `config.tau`, every `config.delay`-th
  call. `steps` is the target's call counter. Returns the live and the
  pre-update target params, for the caller's drift metric."""
  steps[...] += 1
  live = nnx.state(net, nnx.Param)
  tgt = nnx.state(net_target, nnx.Param)
  blend = lambda t, l: t + config.tau * (l - t)
  if config.delay > 1:
    due = (steps[...] % config.delay) == 0
    step = jax.tree.map(lambda t, l: jnp.where(due, blend(t, l), t), tgt, live)
  else:
    step = jax.tree.map(blend, tgt, live)
  nnx.update(net_target, step)
  return live, tgt
