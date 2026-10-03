"""The MLP recipes the critic presets compose."""

from barl_jax.networks.mlp import MLPConfig


def crossq_mlp() -> MLPConfig:
  """CrossQ (Bhatt et al., ICLR 2024): post-activation BatchRenorm with input
  and stem normalized, so the joint [(s,a), (s',a')] forward shares statistics."""
  return MLPConfig(activation="gelu", norm="batchrenorm", norm_stem=True, norm_input=True,
                   norm_after_act=True, norm_kwargs={"momentum": 0.99, "warmup_steps": 100000,
                   "epsilon": 0.001, "r_max": 3.0, "d_max": 5.0}, dtype="bfloat16")


def target_mlp() -> MLPConfig:
  """`crossq_mlp` with LayerNorm, for separate target-net forwards."""
  return MLPConfig(activation="gelu", norm="layer", norm_stem=True, norm_input=True,
                   norm_after_act=True, norm_kwargs={"epsilon": 1e-5}, dtype="bfloat16")


def xqc_mlp() -> MLPConfig:
  """XQC (arXiv:2509.25174): pre-activation BatchNorm, input included."""
  return MLPConfig(activation="relu", norm="batch", norm_input=True, norm_stem=True,
                   norm_after_act=False, norm_kwargs={"momentum": 0.99, "epsilon": 0.001},
                   dtype="float32")
