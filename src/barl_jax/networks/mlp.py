import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.hparam import Config
from barl_jax.networks.norm import make_norm

# ---------------------------------------------------------------------------
# MLP_STYLES -- the named MLP surfaces a component *selects* from. A component
# declares `net_style = Config("gelu", style=MLP_STYLES)` and gets the whole
# surface (activation/norm/dtype/...) below its own declarations; switching the
# style is a `--set model.components.<slot>.net_style=silu`, not inheritance.
# ---------------------------------------------------------------------------

_NET_PLAIN = dict(activation="silu", norm="none", norm_stem=False, norm_input=False,
                  norm_after_act=True, dropout=0.0, final_activation=False,
                  residual=False, norm_kwargs={}, dtype="bfloat16")

MLP_STYLES = {
  "plain":    _NET_PLAIN,
  "silu":     _NET_PLAIN,
  "gelu":     {**_NET_PLAIN, "activation": "gelu"},
  "relu":     {**_NET_PLAIN, "activation": "relu"},
  # mish, unnormalized -- the NC-LQL critics' net.
  "mish":     {**_NET_PLAIN, "activation": "mish"},
  # A LayerNorm stem (epsilon 1e-5, torch's) over silu -- the Gaussian actor's net.
  "layer":    {**_NET_PLAIN, "norm": "layer",
               "norm_kwargs": {"epsilon": 1.0e-5}},
  "layer_relu": {**_NET_PLAIN, "activation": "relu", "norm": "layer",
                 "norm_stem": True, "norm_input": True,
                 "norm_kwargs": {"epsilon": 1.0e-5}},
  # BatchRenorm over the shared [(s,a),(s',a')] forward -- what replaces a
  # Polyak target net, and the default critic net.
  "crossq":   {**_NET_PLAIN, "activation": "relu", "norm": "batchrenorm",
               "norm_stem": True, "norm_input": True,
               "norm_kwargs": {"momentum": 0.99, "warmup_steps": 100000,
                               "epsilon": 1.0e-3, "r_max": 3.0, "d_max": 5.0}},
  # fp32 compute -- the score/SDE actors accumulate a log-weight over an fp32
  # chain (they went non-finite in bf16 -- sweep 2026-08-27_13-23-15).
  "silu_fp32": {**_NET_PLAIN, "dtype": "float32"},
  "gelu_fp32": {**_NET_PLAIN, "activation": "gelu", "dtype": "float32"},
}


_ACTIVATIONS = dict(
  silu=jax.nn.silu,
  relu=jax.nn.relu,
  gelu=jax.nn.gelu,
  tanh=jax.nn.tanh,
  relu6=jax.nn.relu6,
  leakyrelu=jax.nn.leaky_relu,
  mish=jax.nn.mish,
)

_DTYPES = dict(
  float32=jnp.float32,
  bfloat16=jnp.bfloat16,
  float16=jnp.float16,
)

def Linear(in_features: int, out_features: int, *, rngs: nnx.Rngs,
           dtype=jnp.float32) -> nnx.Linear:
  return nnx.Linear(in_features, out_features,
                    kernel_init=nnx.initializers.lecun_normal(),
                    bias_init=nnx.initializers.zeros_init(),
                    dtype=dtype, param_dtype=jnp.float32, rngs=rngs)


# The full MLP surface, restated as named parameters by every component that
# builds one (see e.g. StochasticActor). Named rather than passed through a
# **kwargs catch-all on purpose: model_config.validate() reports a kwarg no
# __init__ in the MRO accepts, and a catch-all would swallow every typo in a
# `components:` block silently. Keep this list and those signatures in step.
MLP_KWARGS = ("activation", "norm", "norm_input", "norm_stem", "norm_after_act", "dropout",
              "final_activation", "residual", "norm_kwargs", "dtype")


class ResidualBlock(nnx.Module):
  """One hidden layer with residual connection, matching the main MLP's
  norm_after_act ordering: Linear -> act -> Norm (default) or Linear -> Norm -> act."""

  def __init__(self, latent_dim: int, act, norm: str, *, dropout: float,
               norm_after_act: bool, rngs: nnx.Rngs, dtype=jnp.float32, **norm_kwargs):
    self.norm          = make_norm(norm, latent_dim, rngs=rngs, dtype=dtype, **norm_kwargs)
    self.act           = act
    self.lin           = Linear(latent_dim, latent_dim, rngs=rngs, dtype=dtype)
    self.dropout       = nnx.Dropout(dropout, rngs=rngs) if dropout > 0 else None
    self.norm_after_act = norm_after_act

  def __call__(self, x):
    h = self.lin(x)
    if self.norm_after_act:
      h = self.act(h)
      if self.norm is not None:
        h = self.norm(h)
    else:
      if self.norm is not None:
        h = self.norm(h)
      h = self.act(h)
    if self.dropout is not None:
      h = self.dropout(h)
    return x + h


class MLP(nnx.Module):

  def __init__(self, input_dim: int, output_dim: int, latent_dim: int, hidden_layers: int,
               *, activation: str, norm: str,
               norm_input: bool, norm_stem: bool,
               norm_after_act: bool, dropout: float,
               final_activation: bool, residual: bool,
               dtype: str,
               rngs: nnx.Rngs, **norm_kwargs):
    """`norm` selects the normalization ("none" / "layer" / "batch" /
    "batchrenorm"); leaving it unset gives LayerNorm.

    `dtype` is the compute dtype only -- parameters stay float32 regardless.

    `dropout` is applied to each hidden layer's output, before its norm and
    activation -- the ordering DroQ (Hiraoka et al., 2022) uses to get
    ensemble-like uncertainty out of far fewer heads. It is off at eval, since
    `nnx.Dropout`'s `deterministic` flag is one of the two things
    `nnx.Module.eval()` flips.

    `norm_input` normalizes the raw input, before the stem's Linear -- what
    dime_am's critic does and what a CrossQ-style network wants, since it is
    the input population the shared current+next forward has to agree on. It
    and `norm_stem` are off by default because barl's MLP historically had
    neither. Note the shapes still differ slightly even with both on: CrossQ's
    net is `layers` Linear->Norm->Act blocks plus a head, where this MLP
    additionally has a stem, so `layers=2` here is one Linear deeper."""

    activation = activation.lower()

    assert activation in _ACTIVATIONS, f"{activation} not in available activations {list(_ACTIVATIONS)}"
    assert dtype in _DTYPES, f"{dtype} not in available dtypes {list(_DTYPES)}"

    act   = _ACTIVATIONS[activation]
    dtype = _DTYPES[dtype]

    self.input_dim        = input_dim
    self.output_dim       = output_dim
    self.latent_dim       = latent_dim
    self.hidden_layers    = hidden_layers
    self.norm             = norm
    self.dropout_rate     = dropout
    self.final_activation = final_activation
    self.residual         = residual
    self.act              = act
    self.dtype            = dtype
    self.norm_after_act   = bool(norm_after_act)

    normed = norm != "none"
    mk = lambda dim: make_norm(norm, dim, rngs=rngs, dtype=dtype, **norm_kwargs)

    # A residual stem is a bare linear embedding of the stream -- the activation
    # the plain path puts here would be immediately followed by each block's own.
    self.input_norm = mk(input_dim) if norm_input else None
    self.stem = Linear(input_dim, latent_dim, rngs=rngs, dtype=dtype)
    self.stem_norm = mk(latent_dim) if norm_stem else None

    blocks, norms, drops = [], [], []
    for _ in range(hidden_layers):
      if residual:
        # One Linear per block, same as the plain path, so `residual` swaps the
        # wiring rather than the width -- the two differ only by the head norm.
        blocks.append(ResidualBlock(latent_dim, act, norm, dropout,
                                    norm_after_act=norm_after_act, rngs=rngs,
                                    dtype=dtype, **norm_kwargs))
      else:
        blocks.append(Linear(latent_dim, latent_dim, rngs=rngs, dtype=dtype))
        norms.append(mk(latent_dim))
        drops.append(nnx.Dropout(dropout, rngs=rngs) if dropout > 0 else None)
    self.blocks = nnx.List(blocks)
    self.norms  = nnx.List(norms)
    self.drops  = nnx.List(drops)

    # Nothing has normalized the residual stream itself, only each branch's
    # input, so without this the head sees a scale that grows with depth.
    self.stream_norm = mk(latent_dim) if (residual and normed) else None

    self.head = Linear(latent_dim, output_dim, rngs=rngs, dtype=dtype)
    self.head_norm = mk(output_dim) if (final_activation and normed) else None

  def __call__(self, *x):
    h = x[0] if len(x) == 1 else jnp.concatenate(x, axis=-1)

    # Cast once at the boundary: inputs (observations, actions, time embeddings)
    # are float32 and every consumer of the output casts back.
    h = h.astype(self.dtype)

    if self.input_norm is not None:
      h = self.input_norm(h)

    # `norm_after_act` is dime_am's order (Dense -> act -> Norm); the other is
    # barl's original (Dense -> Norm -> act). For a BatchRenorm the two track
    # different populations -- post-activation statistics are those of a
    # one-sided, half-zero signal -- so this is not cosmetic.
    h = self.stem(h)
    if self.residual:
      if self.stem_norm is not None:
        h = self.stem_norm(h)
    elif self.norm_after_act:
      h = self.act(h)
      if self.stem_norm is not None:
        h = self.stem_norm(h)
    else:
      if self.stem_norm is not None:
        h = self.stem_norm(h)
      h = self.act(h)

    for i, block in enumerate(self.blocks):
      if self.residual:
        h = block(h)
        continue
      h = block(h)
      drop = self.drops[i]
      if drop is not None:
        h = drop(h)
      norm = self.norms[i]
      if self.norm_after_act:
        h = self.act(h)
        if norm is not None:
          h = norm(h)
      else:
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
