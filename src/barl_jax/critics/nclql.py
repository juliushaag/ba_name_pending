from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Buffer, _Static
from barl_jax.networks.mlp import MLPConfig
from barl_jax.networks.schedule import Schedule, ScheduleConfig
from barl_jax.networks.time_embedding import TimeEmbedding, TimeEmbeddingConfig
from barl_jax.registration import get_component, register_component

from .base import CriticConfig, target_mlp
from .categorical import C51Critic, C51CriticConfig, HLGaussCritic, HLGaussCriticConfig
from .ensemble import EnsembleCritic, EnsembleCriticConfig


@dataclass(frozen=True)
class NoiseConditionedCriticConfig(CriticConfig):
  """Fields shared by every noise-conditioned TD critic (NCLQL's clean+smoothing
  objective on top of a plain critic's value head) -- mixed in ahead of the
  value-head-specific config so each `NCLQL*CriticConfig` below only states
  what actually differs from its non-noise-conditioned counterpart."""
  reward_scale: float = 1.0
  noise_schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
  time_emb: str = "fourier"
  time: TimeEmbeddingConfig = field(default_factory=lambda: TimeEmbeddingConfig(
    kind="fourier", time_dim=16, num_freqs=16, hidden=32, concat_input=True))
  smoothing_coef: float = 1.0
  polyak_tau: float = 0.005
  mlp: MLPConfig = field(default_factory=target_mlp)


@dataclass(frozen=True)
class NCLQLCriticConfig(NoiseConditionedCriticConfig, EnsembleCriticConfig):
  class_name: str = "NCLQLCritic"


@dataclass(frozen=True)
class NCLQLC51CriticConfig(NoiseConditionedCriticConfig, C51CriticConfig):
  class_name: str = "NCLQLC51Critic"
  rho: float = 1.0
  pessimistic_target: bool = True


@dataclass(frozen=True)
class NCLQLHLGaussCriticConfig(NoiseConditionedCriticConfig, HLGaussCriticConfig):
  class_name: str = "NCLQLHLGaussCritic"
  rho: float = 1.0
  pessimistic_target: bool = True


class NoiseConditionedCritic:
  learning_objective = "noise_td"
  supports_bon = False

  def __init__(self, env_stats, config, *args, rngs: nnx.Rngs, **kwargs):
    time_values = config.time.to_runtime_dict()
    self.time_emb = config.time_emb
    self.time_dim = time_values["time_dim"]
    self.time_embed = time_values["time_embed"]
    self._static_config = _Static({"time_kwargs": time_values["time_kwargs"]})

    schedule_config = config.noise_schedule
    schedule_cls = get_component(schedule_config.class_name)
    if not issubclass(schedule_cls, Schedule):
      raise TypeError("noise_schedule must name a Schedule")
    self.sigmas_schedule = schedule_cls(env_stats, schedule_config)
    self.sigmas = Buffer(self.sigmas_schedule.ladder())
    if self.sigmas.shape[0] < 2:
      raise ValueError("Noise-conditioned TD requires at least two noise levels")
    if env_stats.get("return_min") is not None:
      env_stats = env_stats.derive(return_min=env_stats["return_min"] * config.reward_scale,
                                   return_max=env_stats["return_max"] * config.reward_scale)
    self.encoder = TimeEmbedding(time_dim=self.time_dim, kind=self.time_embed,
                                 rngs=rngs, **self.static("time_kwargs"))

    super().__init__(env_stats.derive(act_dim=env_stats["act_dim"] + self.time_dim),
                     config, *args, rngs=rngs, **kwargs)

    self.action_dim = env_stats["act_dim"]

  def target_modules(self) -> dict:
    return {**super().target_modules(), "encoder": self.encoder}

  def td_loss(self, obs, action, next_obs, next_action, chunk_return, terminated, horizon,
              entropy_bonus=0.0, *, key=None) -> dict:
    """Clean TD plus smoothing, sharing the sampler's noise ladder."""
    if key is None:
      raise ValueError("Noise-conditioned TD requires an explicit random key")
    return self.noise_td_loss(obs, action, next_obs, next_action,
                              chunk_return * self.config.reward_scale, terminated, horizon,
                              self.sigmas[...], key, entropy_bonus=entropy_bonus)

  def noise_td_loss(self, obs, action, next_obs, next_action, chunk_return, terminated, horizon,
                    sigmas, key, entropy_bonus=0.0) -> dict:
    """NCLQL has no entropy dual -- `entropy_bonus` stays at its 0.0 default at
    every call site. See `QCriticBase.td_loss` for the chunk_return/terminated/
    horizon contract this shares."""
    clean_level = sigmas.shape[0] - 1

    level_key, noise_key = jax.random.split(key)
    levels = jax.random.randint(level_key, (action.shape[0],), 0, clean_level)
    noisy  = action + sigmas[levels][:, None] * jax.random.normal(
      noise_key, action.shape, action.dtype)

    pred_clean, pred_noisy = self.raw_batches(
      (obs, self.augment(action, clean_level)),
      (obs, self.augment(noisy, levels)),
    )

    next_raw  = self.target_raw(next_obs, self.augment(next_action, clean_level, target=True))
    td_target = self.bellman_target(chunk_return, terminated, horizon,
                                    self.bootstrap_from_raw(next_raw), entropy_bonus)

    td_loss = self.regression_loss(pred_clean, td_target)
    smoothing_loss = self.regression_loss(pred_noisy, self.target_from_raw(pred_clean))

    self.record(pred_clean, td_target)
    self.metrics.scalar("td_loss", td_loss)
    self.metrics.scalar("smoothing_loss", smoothing_loss)
    return {"critic": td_loss + self.config.smoothing_coef * smoothing_loss}

  def augment(self, act, level, target: bool = False):
    if not isinstance(level, jax.Array):
      level = jnp.full((act.shape[0],), level, jnp.int32)

    embed = self.encoder_target if target and self.has_target else self.encoder
    return jnp.concatenate([act, embed(level.astype(jnp.float32)[:, None])], axis=-1)


@register_component
class NCLQLCritic(NoiseConditionedCritic, EnsembleCritic):
  kind = "critic"
  config_type = NCLQLCriticConfig


@register_component
class NCLQLC51Critic(NoiseConditionedCritic, C51Critic):
  kind = "critic"
  config_type = NCLQLC51CriticConfig


@register_component
class NCLQLHLGaussCritic(NoiseConditionedCritic, HLGaussCritic):
  kind = "critic"
  config_type = NCLQLHLGaussCriticConfig
