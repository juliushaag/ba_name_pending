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

from barl_jax.networks.flow import TimeEmbedding
from barl_jax.networks.mlp import MLP


class TimeConditionedField(nnx.Module):
  """`(obs, x, t) -> a vector over the action dims`, conditioned on a scalar time.

  What `t` means is the caller's: a normalized tau, a step index, a noise level.
  The output is left in the net's compute dtype -- a caller that accumulates in
  float32 casts at its own boundary."""

  def __init__(self, obs_dim: int, action_dim: int, *, latent_dim: int, layers: int,
               time_dim: int, time_embed: str, time_kwargs: dict,
               activation: str, norm: str,
               norm_input: bool, norm_stem: bool, norm_after_act: bool,
               dropout: float, final_activation: bool,
               residual: bool, norm_kwargs: dict, dtype: str,
               rngs: nnx.Rngs):
    self.time_emb = TimeEmbedding(time_dim, kind=time_embed, rngs=rngs, **time_kwargs)
    self.net = MLP(obs_dim + action_dim + time_dim, action_dim, latent_dim, layers,
                   activation=activation, norm=norm, norm_input=norm_input,
                   norm_stem=norm_stem, norm_after_act=norm_after_act,
                   dropout=dropout, final_activation=final_activation,
                   residual=residual, dtype=dtype, rngs=rngs, **(norm_kwargs or {}))

  def zero_init_head(self):
    """Start the chain uncontrolled -- the field is identically 0 at step 0, so
    the sampler begins as whatever process it is anchored to."""
    self.net.head.kernel[...] = jnp.zeros_like(self.net.head.kernel[...])
    self.net.head.bias[...]   = jnp.zeros_like(self.net.head.bias[...])

  def __call__(self, obs, x, t):
    return self.net(obs, x, self.time_emb(t))


# Both loops are unrolled Python rather than a lax.scan, and both split every key
# up front. Scan was measured: it cuts the traced graph by ~15% and compiles ~15%
# faster, and changes SDAC's step time by 0.3% -- so the unrolled form stays,
# since it keeps `n` a Python int and the step algebra readable. The keys are
# split up front because DIME differentiates through its whole chain, so the
# draws have to be ordinary values in the graph rather than module side effects.

def reverse_chain(step, x, num_steps: int, key):
  """Ancestral / denoising direction: n = num_steps .. 1, `step(x, n, key) -> x`."""
  keys = jax.random.split(key, num_steps)
  for i, n in enumerate(range(num_steps, 0, -1)):
    x = step(x, n, keys[i])
  return x


def controlled_sde(step, x, num_steps: int, key, aux=0.0):
  """Forward Euler-Maruyama under a learned control, accumulating a per-sample
  scalar alongside the state: `step(x, i, key) -> (x_next, aux_increment)`.

  The scalar is whatever the sampler's own change of measure is -- a Girsanov
  path weight, a log-ratio of transition kernels -- so `aux` is seeded by the
  caller with the term its process starts from, and read back as (B, 1)."""
  keys = jax.random.split(key, num_steps)
  for i in range(num_steps):
    x, increment = step(x, i, keys[i])
    aux = aux + increment
  return x, aux
