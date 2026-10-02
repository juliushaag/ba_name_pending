"""The parts three otherwise-unrelated samplers genuinely share.

AMDP's controlled SDE, DIME's time-reversed DIS chain and SDAC's ancestral DDPM
loop are different processes with different objectives, and each lives next to
the Policy that owns it. What they have in common is mechanical rather than
mathematical: the same time-conditioned network, and the same unrolled
fixed-step loop over a handful of steps. Nothing here knows about alpha_bar,
eps/x0 conversions or path weights -- those belong to whichever sampler defines
them.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.base import NetworkConfig
from barl_jax.networks.time_embedding import TimeEmbeddingConfig


class TimeConditionedField(nnx.Module):
  """`(obs, x, t) -> a vector over the action dims`, conditioned on a scalar time.

  Diffusion actors default to AdaLNSimBa: the noisy action `x` enters the stem,
  while observation and encoded time modulate every block and the final norm.
  What `t` means is the caller's: a normalized tau, a step index, a noise level.
  The output is left in the net's compute dtype -- a caller that accumulates in
  float32 casts at its own boundary."""

  def __init__(self, obs_dim: int, action_dim: int, *, latent_dim: int, layers: int,
               time: TimeEmbeddingConfig, mlp: NetworkConfig, rngs: nnx.Rngs):
    self.time_emb = time.build(rngs=rngs)
    self.net = mlp.build(action_dim, action_dim, latent_dim, layers,
                         cond_dim=obs_dim + time.time_dim, rngs=rngs)

  def zero_init_head(self):
    """Start the chain uncontrolled -- the field is identically 0 at step 0, so
    the sampler begins as whatever process it is anchored to."""
    self.net.head.kernel[...] = jnp.zeros_like(self.net.head.kernel[...])
    self.net.head.bias[...]   = jnp.zeros_like(self.net.head.bias[...])

  def __call__(self, obs, x, t):
    """`t` is a scalar or `(n,)` shared level, or `(..., 1)` per row. It is embedded
    at its own shape and the embedding broadcast over the batch, so a level shared
    by every row (a step index) costs one embedding rather than one per row."""
    t = jnp.asarray(t, jnp.float32)
    if t.ndim <= 1:
      t = t[..., None]
    emb = self.time_emb(t)
    emb = jnp.broadcast_to(emb, (*x.shape[:-1], emb.shape[-1]))
    return self.net((x,), (obs, emb))



def reverse_chain(step, x, num_steps: int, key):
  """Ancestral / denoising direction: n = num_steps .. 1, `step(x, n, key) -> x`.
  `n` is a traced int32 scalar, so `step` indexes with it rather than branching."""
  keys = jax.random.split(key, num_steps)
  ns = jnp.arange(num_steps, 0, -1, dtype=jnp.int32)
  x, _ = jax.lax.scan(lambda x, nk: (step(x, *nk), None), x, (ns, keys))
  return x


def controlled_sde(step, x, num_steps: int, key, aux=0.0):
  """Forward Euler-Maruyama under a learned control, accumulating a per-sample
  scalar alongside the state: `step(x, i, key) -> (x_next, aux_increment)`.

  The scalar is whatever the sampler's own change of measure is -- a Girsanov
  path weight, a log-ratio of transition kernels -- so `aux` is seeded by the
  caller with the term its process starts from, and read back as (B, 1).
  `i` is a traced int32 scalar."""
  keys = jax.random.split(key, num_steps)
  idx = jnp.arange(num_steps, dtype=jnp.int32)

  def body(carry, ik):
    x, aux = carry
    x, increment = step(x, *ik)
    return (x, aux + increment), None

  # A scalar seed is broadcast to the increment's shape so the carry is fixed.
  increment = jax.eval_shape(lambda: step(x, idx[0], keys[0])[1])
  aux = jnp.broadcast_to(jnp.asarray(aux, increment.dtype), increment.shape)
  (x, aux), _ = jax.lax.scan(body, (x, aux), (idx, keys))
  return x, aux
