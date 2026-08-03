import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.norm import make_norm

_ACTIVATIONS = dict(
  silu=jax.nn.silu,
  relu=jax.nn.relu,
  gelu=jax.nn.gelu,
  tanh=jax.nn.tanh,
  relu6=jax.nn.relu6,
)

# torch's nn.Linear default: kaiming_uniform(a=sqrt(5)) on the weight reduces to
# U(-1/sqrt(fan_in), 1/sqrt(fan_in)), and the bias uses the same bound. Flax
# defaults to lecun_normal + zero bias, so a run started from Flax's init is a
# different experiment -- matching torch here is what keeps the two backends'
# learning curves comparable.
def _uniform_fan_in(fan_in: int):
  bound = float(fan_in) ** -0.5
  def init(key, shape, dtype=jnp.float32):
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
  return init


def Linear(in_features: int, out_features: int, *, rngs: nnx.Rngs) -> nnx.Linear:
  init = _uniform_fan_in(in_features)
  return nnx.Linear(in_features, out_features, kernel_init=init, bias_init=init, rngs=rngs)


def _resolve_norm(norm: str | None, layernorm: bool) -> str:
  """`norm` is the real setting; `layernorm` is the original boolean kept so
  every existing config keeps meaning what it meant."""
  if norm is not None:
    return norm
  return "layer" if layernorm else "none"


class ResidualBlock(nnx.Module):
  """One pre-norm hidden layer: `x + Linear(ACT(LN(x)))`. The norm sits on the
  branch rather than after its Linear, so the skip carries an unnormalized
  identity and a branch whose activation has a one-sided range (ReLU, SiLU)
  cannot walk the stream's mean upward once per layer."""

  def __init__(self, latent_dim: int, act, norm: str, dropout: float = 0.0,
               *, rngs: nnx.Rngs, **norm_kwargs):
    self.norm    = make_norm(norm, latent_dim, rngs=rngs, **norm_kwargs)
    self.act     = act
    self.lin     = Linear(latent_dim, latent_dim, rngs=rngs)
    self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0 else None

  def __call__(self, x):
    h = self.norm(x) if self.norm is not None else x
    h = self.lin(self.act(h))
    if self.dropout is not None:
      h = self.dropout(h)
    return x + h


class MLP(nnx.Module):

  def __init__(self, input_dim: int, output_dim: int, latent_dim: int, hidden_layers: int,
               activation: str = "silu", layernorm: bool = True, norm: str = None,
               norm_input: bool = False, norm_stem: bool = False, dropout: float = 0.0,
               final_activation: bool = False, residual: bool = False,
               *, rngs: nnx.Rngs, **norm_kwargs):
    """`norm` selects the normalization ("none" / "layer" / "batch" /
    "batchrenorm"); leaving it unset falls back to the original `layernorm`
    boolean, so every existing config is unchanged.

    `dropout` is applied to each hidden layer's output, before its norm and
    activation -- the ordering DroQ (Hiraoka et al., 2022) uses to get
    ensemble-like uncertainty out of far fewer heads. It is off at eval, since
    `nnx.Dropout`'s `deterministic` flag is one of the two things
    `nnx.Module.eval()` flips.

    `norm_input` and `norm_stem` are off by default because barl's MLP has no
    input norm and leaves its stem un-normalized. Both are what a CrossQ-style
    network wants: it normalizes the network input and every Linear's output,
    which for `batchrenorm` is the whole point -- the running statistics are
    what let the critic drop its target network. Note the shapes still differ
    slightly: CrossQ's net is `layers` Linear->Norm->Act blocks plus a head,
    where this MLP additionally has a stem, so `layers=2` here is one Linear
    deeper than CrossQ's."""

    assert activation in _ACTIVATIONS, f"{activation} not in available activations {list(_ACTIVATIONS)}"

    act  = _ACTIVATIONS[activation]
    norm = _resolve_norm(norm, layernorm)

    self.input_dim        = input_dim
    self.output_dim       = output_dim
    self.latent_dim       = latent_dim
    self.hidden_layers    = hidden_layers
    self.layernorm        = layernorm
    self.norm             = norm
    self.dropout_rate     = dropout
    self.final_activation = final_activation
    self.residual         = residual
    self.act              = act

    normed = norm != "none"
    mk = lambda dim: make_norm(norm, dim, rngs=rngs, **norm_kwargs)

    self.input_norm = mk(input_dim) if norm_input else None

    # A residual stem is a bare linear embedding of the stream -- the activation
    # the plain path puts here would be immediately followed by each block's own.
    self.stem = Linear(input_dim, latent_dim, rngs=rngs)
    self.stem_norm = mk(latent_dim) if norm_stem else None

    blocks, norms, drops = [], [], []
    for _ in range(hidden_layers):
      if residual:
        # One Linear per block, same as the plain path, so `residual` swaps the
        # wiring rather than the width -- the two differ only by the head norm.
        blocks.append(ResidualBlock(latent_dim, act, norm, dropout, rngs=rngs, **norm_kwargs))
      else:
        blocks.append(Linear(latent_dim, latent_dim, rngs=rngs))
        norms.append(mk(latent_dim))
        drops.append(nnx.Dropout(dropout, rngs=rngs) if dropout > 0 else None)
    self.blocks = nnx.List(blocks)
    self.norms  = nnx.List(norms)
    self.drops  = nnx.List(drops)

    # Nothing has normalized the residual stream itself, only each branch's
    # input, so without this the head sees a scale that grows with depth.
    self.stream_norm = mk(latent_dim) if (residual and normed) else None

    self.head = Linear(latent_dim, output_dim, rngs=rngs)
    self.head_norm = mk(output_dim) if (final_activation and normed) else None

  def __call__(self, *x):
    h = x[0] if len(x) == 1 else jnp.concatenate(x, axis=-1)

    if self.input_norm is not None:
      h = self.input_norm(h)

    h = self.stem(h)
    if self.stem_norm is not None:
      h = self.stem_norm(h)
    if not self.residual:
      h = self.act(h)

    for i, block in enumerate(self.blocks):
      if self.residual:
        h = block(h)
      else:
        h = block(h)
        drop = self.drops[i]
        if drop is not None:
          h = drop(h)
        norm = self.norms[i]
        if norm is not None:
          h = norm(h)
        h = self.act(h)

    if self.stream_norm is not None:
      h = self.stream_norm(h)

    h = self.head(h)

    if self.final_activation:
      if self.head_norm is not None:
        h = self.head_norm(h)
      h = self.act(h)

    return h
