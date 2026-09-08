import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.components import Buffer, ConfigOwner
from barl_jax.config.toolkit import ConfigMixin
from barl_jax.hparam import Config, ConfigDict
from barl_jax.networks.schedule import Schedule
from barl_jax.networks.time_embedding import TIME_STYLES, TimeEmbedding
from barl_jax.registration import get_component, register_component

from .base import Polyak
from .categorical import C51Critic, HLGaussCritic
from .ensemble import EnsembleCritic


class NoiseConditionedCritic(ConfigOwner):
  learning_objective = "noise_td"
  supports_bon = False
  reward_scale = Config(5.0)
  noise_schedule = Config({"class": "ExponentialSchedule", "start": 0.1,
                           "end": 0.001, "steps": 10})

  time_emb  = Config("fourier", style=TIME_STYLES)
  smoothing_coef = Config(1.0)

  def __init__(self, env_stats, config: ConfigMixin, *args, rngs: nnx.Rngs, **kwargs):
    self.resolve_config(config, env_stats=env_stats)
    schedule_config = dict(self.static("noise_schedule"))
    schedule_cls = get_component(schedule_config.pop("class"))
    if not issubclass(schedule_cls, Schedule):
      raise TypeError("noise_schedule must name a Schedule")
    schedule_config = ConfigMixin("", schedule_config)
    self.sigmas_schedule = schedule_cls(env_stats, schedule_config)
    schedule_config.assert_all_consumed(owner="noise_schedule")
    self.sigmas = Buffer(self.sigmas_schedule.ladder())
    if self.sigmas.shape[0] < 2:
      raise ValueError("Noise-conditioned TD requires at least two noise levels")
    if env_stats.get("return_min") is not None:
      env_stats = env_stats.derive(return_min=env_stats["return_min"] * self.reward_scale,
                                   return_max=env_stats["return_max"] * self.reward_scale)
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
                              chunk_return * self.reward_scale, terminated, horizon,
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
    return {"critic": td_loss + self.smoothing_coef * smoothing_loss}

  def augment(self, act, level, target: bool = False):
    if not isinstance(level, jax.Array):
      level = jnp.full((act.shape[0],), level, jnp.int32)

    embed = self.encoder_target if target and self.has_target else self.encoder
    return jnp.concatenate([act, embed(level.astype(jnp.float32)[:, None])], axis=-1)




_NCLQL_MLP = dict(activation='mish', norm='none', norm_stem=False, norm_input=False,
                  norm_after_act=True, dropout=0.0, final_activation=False,
                  residual=False, norm_kwargs={}, dtype='bfloat16')


@register_component
class NCLQLCritic(Polyak, NoiseConditionedCritic, EnsembleCritic):
  kind = "critic"
  mlp = ConfigDict(**_NCLQL_MLP)
  rho                = Config(1.0)
  pessimistic_target = Config(True)


@register_component
class NCLQLC51Critic(Polyak, NoiseConditionedCritic, C51Critic):
  kind = "critic"
  mlp = ConfigDict(**_NCLQL_MLP)
  entropy_coef    = Config(0.005)
  rho             = Config(1.0)
  pessimistic_target = Config(True)


@register_component
class NCLQLHLGaussCritic(Polyak, NoiseConditionedCritic, HLGaussCritic):
  kind = "critic"
  mlp = ConfigDict(**_NCLQL_MLP)
