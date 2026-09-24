"""Named combinations of the critic axes, applicable to any head's config."""

from dataclasses import replace

from .base import EnsembleConfig, NoiseConditioningConfig, TargetConfig, target_mlp, xqc_mlp
from .categorical import CategoricalCriticConfig


def target_ensemble(config):
  """Polyak target net with separate forwards and LayerNorm."""
  return replace(config, target=TargetConfig(kind="polyak", joint_forward=False),
                 mlp=target_mlp())


def xqc(config):
  """XQC (Palenicek et al., arXiv:2509.25174): polyak target sharing the joint
  batch, pre-activation BatchNorm, weight normalization, min-member bootstrap."""
  config = replace(config, target=TargetConfig(kind="polyak", joint_forward=True),
                   ensemble=EnsembleConfig(n_models=2, value_reduce="mean", target_reduce="min"),
                   mlp=xqc_mlp(), weight_norm=True, latent_dim=512, layers=3, adam_b1=0.9)
  if isinstance(config, CategoricalCriticConfig):
    config = replace(config, num_bins=101, entropy_coef=0.0)
  return config


def nclql(config):
  """NCLQL: noise-conditioned TD on a polyak/LayerNorm critic, min-reduced."""
  return replace(target_ensemble(config), noise=NoiseConditioningConfig(enabled=True),
                 ensemble=replace(config.ensemble, value_reduce="min", target_reduce="min"))
