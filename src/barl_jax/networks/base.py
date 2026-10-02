"""What every trunk network shares: a selectable, registered config that builds it.

A component holds `mlp: NetworkConfig` and calls `config.mlp.build(...)`, so a new
network is one `NetworkConfig` subclass registered here plus its module; nothing
that consumes a network changes."""

import math
from dataclasses import dataclass, field
from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.config.serialization import variant_registry
from barl_jax.networks.norm import make_norm

ACTIVATIONS = dict(
  silu=jax.nn.silu,
  relu=jax.nn.relu,
  gelu=jax.nn.gelu,
  tanh=jax.nn.tanh,
  relu6=jax.nn.relu6,
  leakyrelu=jax.nn.leaky_relu,
  mish=jax.nn.mish,
)

# Conventional gains for rectifiers/tanh. Smooth-activation gains approximate
# 1 / sqrt(E[activation(Z)^2]), Z ~ N(0, 1), using JAX's default activations
# (including approximate GELU). They are initialization heuristics, not a
# guarantee of variance preservation through normalization or residual skips.
ACTIVATION_GAINS = dict(
  relu=math.sqrt(2.0),
  relu6=math.sqrt(2.0),  # ReLU approximation before the upper clip is active.
  leakyrelu=math.sqrt(2.0 / (1.0 + 0.01 ** 2)),  # jax.nn.leaky_relu default slope.
  tanh=5.0 / 3.0,
  silu=1.67653247,
  gelu=1.53358047,
  mish=1.48684758,
)

DTYPES = dict(
  float32=jnp.float32,
  bfloat16=jnp.bfloat16,
  float16=jnp.float16,
)

NETWORK_CONFIGS, register_network = variant_registry()


@dataclass(frozen=True)
class NetworkConfig:
  """Settings common to every trunk. `class_name` selects the registered subclass
  (see `config.serialization`); `dtype` is the compute dtype only, parameters stay
  float32."""
  variants: ClassVar[dict[str, type]] = NETWORK_CONFIGS

  class_name: str = ""
  activation: str = "silu"
  norm: str = "layer"
  dropout: float = 0.0
  norm_kwargs: dict = field(default_factory=lambda: {"epsilon": 1e-5})
  dtype: str = "float32"
  head_gain: float = 0.01

  def build(self, input_dim: int, output_dim: int, latent_dim: int, layers: int,
            *, cond_dim: int = 0, rngs: nnx.Rngs) -> nnx.Module:
    """A module `(inputs=(...), conditioning=(...)) -> (..., output_dim)`, each a tuple of
    arrays joined on the last axis, of widths `input_dim` and `cond_dim`. Plain trunks
    concatenate both; adaptive ones modulate their blocks with `conditioning`. It exposes
    `stem`, `hidden_linears` and `head` (what `normalize_weights` projects)."""
    raise NotImplementedError

  def problems(self) -> list[str]:
    from barl_jax.networks.norm import NORMS
    problems = [f"{name}={getattr(self, name)!r}: unknown {name}; available: {sorted(allowed)}"
            for name, allowed in (("activation", ACTIVATIONS), ("dtype", DTYPES), ("norm", NORMS))
            if getattr(self, name) not in allowed]
    if not (math.isfinite(self.head_gain) and self.head_gain > 0):
      problems.append("head_gain must be finite and > 0")
    return problems

  def resolve(self):
    activation = self.activation.lower()
    assert activation in ACTIVATIONS, f"{activation} not in available activations {list(ACTIVATIONS)}"
    assert self.dtype in DTYPES, f"{self.dtype} not in available dtypes {list(DTYPES)}"
    return ACTIVATIONS[activation], DTYPES[self.dtype]

  def make_norm(self, dim: int, *, rngs: nnx.Rngs):
    return make_norm(self.norm, dim, rngs=rngs, dtype=DTYPES[self.dtype], **self.norm_kwargs)


def Linear(in_features: int, out_features: int, *, rngs: nnx.Rngs,
           dtype=jnp.float32, activation: str | None = None, gain: float = 1.0) -> nnx.Linear:
  """Zero-bias Linear initialized for its following activation, if any.

  Rectifiers and smooth activations use fan-in scaling; tanh uses Xavier
  scaling with gain 5/3. Bare projections use LeCun normal. `gain` additionally
  scales the kernel's standard deviation (heads normally use 0.01).
  """
  if not (math.isfinite(gain) and gain > 0):
    raise ValueError("Linear gain must be finite and > 0")
  activation = activation.lower() if activation is not None else None
  act_gain = ACTIVATION_GAINS[activation] if activation is not None else 1.0
  mode = "fan_avg" if activation == "tanh" else "fan_in"
  kernel_init = nnx.initializers.variance_scaling((gain * act_gain) ** 2, mode,
                                                "truncated_normal")
  return nnx.Linear(in_features, out_features,
                    kernel_init=kernel_init,
                    bias_init=nnx.initializers.zeros_init(),
                    dtype=dtype, param_dtype=jnp.float32, rngs=rngs)


def concat(*groups):
  """Every array of every group, joined on the last axis."""
  xs = [x for group in groups for x in group]
  return xs[0] if len(xs) == 1 else jnp.concatenate(xs, axis=-1)


def make_dropout(rate: float, rngs: nnx.Rngs):
  return nnx.Dropout(rate, rngs=rngs) if rate > 0 else None


def act_norm(h, act, norm, norm_after_act: bool):
  """Activation and norm in either order; either may be None. For a BatchRenorm
  the two orders track different populations (post-activation statistics are
  those of a one-sided signal), so the choice is not cosmetic."""
  for f in ((act, norm) if norm_after_act else (norm, act)):
    if f is not None:
      h = f(h)
  return h


def _renorm_dense(lin: nnx.Linear, *, norm_bias: bool) -> None:
  """Unit L2 norm per output unit over axis -2 (the input dim, whatever
  leading ensemble axes precede it), bias folded in when `norm_bias`."""
  kernel = lin.kernel[...]
  w = jnp.concatenate([kernel, lin.bias[...][..., None, :]], axis=-2) if norm_bias else kernel
  norm = jnp.linalg.norm(w, axis=-2, keepdims=True)
  if norm_bias:
    lin.bias[...] = lin.bias[...] / norm[..., 0, :]
  lin.kernel[...] = kernel / norm


def normalize_weights(net) -> None:
  """XQC's weight projection (arXiv:2509.25174): hidden layers with bias,
  the head's kernel only."""
  for lin in net.hidden_linears:
    _renorm_dense(lin, norm_bias=True)
  _renorm_dense(net.head, norm_bias=False)
