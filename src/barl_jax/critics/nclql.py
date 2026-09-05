import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.config.toolkit import ConfigMixin
from barl_jax.hparam import Config, HParam
from barl_jax.networks.flow import TIME_STYLES, TimeEmbedding
from barl_jax.networks.mlp import MLP_PRESETS
from barl_jax.networks.registry import register_component
from barl_jax.components import HParamsOwner

from .base import Polyak
from .categorical import C51Critic, HLGaussCritic
from .ensemble import EnsembleCritic


class NoiseConditionedCritic(HParamsOwner):
  learning_objective = "noise_td"

  time_emb  = Config("fourier", style=TIME_STYLES)
  smoothing_coef = HParam(1.0)

  def __init__(self, env_stats, config: ConfigMixin, *args, rngs: nnx.Rngs, **kwargs):
    self.resolve_hparams(config, env_stats=env_stats)
    self.encoder = TimeEmbedding(time_dim=self.time_dim, kind=self.time_embed,
                                 rngs=rngs, **self.hparam("time_kwargs"))

    super().__init__(env_stats.derive(act_dim=env_stats["act_dim"] + self.time_dim),
                     config, *args, rngs=rngs, **kwargs)

    self.action_dim = env_stats["act_dim"]

  def target_modules(self) -> dict:
    return {**super().target_modules(), "encoder": self.encoder}

  def augment(self, act, level, target: bool = False):
    if not isinstance(level, jax.Array):
      level = jnp.full((act.shape[0],), level, jnp.int32)

    embed = self.encoder_target if target and self.has_target else self.encoder
    return jnp.concatenate([act, embed(level.astype(jnp.float32)[:, None])], axis=-1)




@register_component
class NCLQLCritic(Polyak, NoiseConditionedCritic, EnsembleCritic):
  kind = "critic"
  mlp_preset = Config("mish", style=MLP_PRESETS)
  rho                = HParam(1.0)
  pessimistic_target = Config(True)


@register_component
class NCLQLC51Critic(Polyak, NoiseConditionedCritic, C51Critic):
  kind = "critic"
  mlp_preset = Config("mish", style=MLP_PRESETS)
  entropy_coef    = HParam(0.005)
  rho             = HParam(1.0)
  pessimistic_target = Config(True)


@register_component
class NCLQLHLGaussCritic(Polyak, NoiseConditionedCritic, HLGaussCritic):
  kind = "critic"
  mlp_preset = Config("mish", style=MLP_PRESETS)
