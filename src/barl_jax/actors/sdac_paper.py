import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from barl_jax.components import Actor, Buffer, frozen
from barl_jax.config.schemas import ComponentsConfig, ModelConfig, register_model
from barl_jax.critics.categorical import C51CriticConfig
from barl_jax.networks.base import NetworkConfig
from barl_jax.networks.diffusion import TimeConditionedField
from barl_jax.networks.mlp import MLPConfig
from barl_jax.networks.simba import SimbaConfig
from barl_jax.networks.time_embedding import SinusoidalTimeConfig, TimeEmbeddingConfig
from barl_jax.optimization import ActorOptimConfig, adam
from barl_jax.policies.rl import PolicyConfig, RLPolicy
from barl_jax.registration import register_component, register_policy

# Rows of `PaperDDPMActor.coef`, one column per diffusion step.
SQRT_ALPHA_BAR, SQRT_ONE_MINUS, SQRT_RECIP, SQRT_RECIPM1, POST_MEAN_1, POST_MEAN_2, POST_LOGVAR = range(7)

# `sinusoidal_emb` multiplies its input by 1000 (it expects t in [0, 1]); the
# reference feeds the raw step index, so the index is pre-divided to cancel it.
TIME_PRE_SCALE = 1.0e-3


@dataclass(frozen=True)
class PaperDDPMActorConfig(ActorOptimConfig):
  """The official SDAC release (mahaitongdae/diffusion_policy_online_rl,
  `relax/algorithm/sdac.py` with `scripts/train_mujoco.py` defaults)."""
  class_name: str = "PaperDDPMActor"
  grad_clip_mode: str = "none"
  latent_dim: int = 128
  layers: int = 3
  mlp: NetworkConfig = field(default_factory=lambda: SimbaConfig(
    activation="silu", norm="layer", norm_kwargs={"epsilon": 1e-5}, dtype="bfloat16"))
  time: TimeEmbeddingConfig = field(default_factory=lambda: SinusoidalTimeConfig(time_dim=16))
  num_steps: int = 20
  beta_scale: float = 0.8
  beta_start: float = 1.0e-4
  beta_end: float = 0.999
  prior_scale: float = 0.5
  num_particles: int = 32
  noise_scale: float = 0.1
  alpha_init: float = 5.0
  alpha_lr: float = 7.0e-3
  target_entropy_scale: float = 0.9
  weight_scale: float = 5.0
  num_candidates: int = 64

  def problems(self):
    problems = super().problems()
    for name in ("num_steps", "num_particles", "num_candidates"):
      if getattr(self, name) < 1:
        problems.append(f"{name} must be >= 1")
    if not (0 < self.beta_start < self.beta_end and 0 < self.beta_scale * self.beta_end < 1):
      problems.append("betas require 0 < beta_start < beta_end and beta_scale * beta_end < 1")
    for name in ("noise_scale", "alpha_init", "alpha_lr", "weight_scale"):
      if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0):
        problems.append(f"{name} must be finite and > 0")
    if self.prior_scale <= 0 or self.target_entropy_scale <= 0:
      problems.append("prior_scale and target_entropy_scale must be > 0")
    return problems


def diffusion_coefficients(num_steps, beta_scale, beta_start, beta_end):
  """The reference's `BetaScheduleCoefficients.from_beta`, linear schedule, in
  float64 on the host: a `(7, num_steps)` table indexed by the SDAC row names."""
  betas = beta_scale * np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)
  alphas = 1.0 - betas
  alpha_bar = np.cumprod(alphas)
  alpha_bar_prev = np.append(1.0, alpha_bar[:-1])
  post_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
  return np.stack([
    np.sqrt(alpha_bar),
    np.sqrt(1.0 - alpha_bar),
    np.sqrt(1.0 / alpha_bar),
    np.sqrt(1.0 / alpha_bar - 1.0),
    betas * np.sqrt(alpha_bar_prev) / (1.0 - alpha_bar),
    (1.0 - alpha_bar_prev) * np.sqrt(alphas) / (1.0 - alpha_bar),
    np.log(np.maximum(post_var, 1.0e-20)),
  ]).astype(np.float32)


@register_component
class PaperDDPMActor(Actor):
  """SDAC as released with the paper (Ma et al. 2025).

  * Acting: `num_particles` draws from the discrete DDPM chain, the one with the
    highest Q kept, then Gaussian noise of std `exp(log_alpha) * noise_scale`.
    Rollout and Bellman bootstrap share this sampler.
  * `log_alpha` is a scalar owned by the actor and trained under its own
    optimizer key; its loss depends on alpha alone (the entropy of the added
    noise against `-target_entropy_scale * plan_dim`), so it is a fixed
    schedule: alpha falls from `alpha_init` towards the value matching the
    target.
  * Loss: a fresh policy action is diffused to a random step, `num_candidates`
    reconstructions are drawn from `eps ~ N(0, I)` (clipped to the box) and
    the eps prediction is regressed onto them with softmax(Q * weight_scale /
    alpha) weights. The temperature is therefore an absolute Q scale set by
    alpha, not a per-batch normalisation.

  Deviations from the release: the final action is clipped to [-1, 1] after the
  noise (the repo's action contract); the net's time input is the sinusoidal
  step index; `sample` carries exploration noise at evaluation too, as every
  actor here does."""

  kind = "actor"
  config_type = PaperDDPMActorConfig

  def __init__(self, env_stats, config: PaperDDPMActorConfig, *, rngs: nnx.Rngs, **kwargs):
    super().__init__(env_stats, config, rngs=rngs, **kwargs)
    problems = self.config_problems(config)
    if problems:
      raise ValueError(f"{type(self).__name__}: " + "; ".join(problems))
    self.field = TimeConditionedField(
      self.obs_dim, self.plan_dim, latent_dim=config.latent_dim, layers=config.layers,
      time=config.time, mlp=config.mlp, rngs=rngs)
    self.log_alpha = nnx.Param(jnp.asarray(math.log(config.alpha_init), jnp.float32))
    self.coef = Buffer(jnp.asarray(diffusion_coefficients(
      config.num_steps, config.beta_scale, config.beta_start, config.beta_end)))

  def create_optimizer(self) -> dict:
    return {"actor": self.config.build(), "alpha": adam(self.config.alpha_lr, 0.9, 1.0, "none")}

  def param_labels(self, subpath):
    return "alpha" if str(subpath[0]) == "log_alpha" else None

  def noise_alpha(self):
    return jnp.exp(self.log_alpha[...])

  # ---- the chain -----------------------------------------------------------
  def predict_eps(self, obs, x, n):
    t = jnp.reshape(jnp.asarray(n, jnp.float32) * TIME_PRE_SCALE, (-1, 1))
    return self.field(obs, x, t).astype(jnp.float32)

  def denoise(self, obs, key):
    """`GaussianDiffusion.p_sample`: x_T = prior_scale * N(0, I), then n = T-1 .. 0,
    the x0 estimate clipped to the box at every step."""
    steps, B = self.config.num_steps, obs.shape[0]
    x_key, noise_key = jax.random.split(key)
    x = self.config.prior_scale * jax.random.normal(x_key, (B, self.plan_dim), jnp.float32)
    noise = jax.random.normal(noise_key, (steps, B, self.plan_dim), jnp.float32)
    coef = self.coef[...]

    def body(x, step):
      n, eps = step
      x0 = jnp.clip(x * coef[SQRT_RECIP, n] - self.predict_eps(obs, x, n) * coef[SQRT_RECIPM1, n], -1.0, 1.0)
      mean = x0 * coef[POST_MEAN_1, n] + x * coef[POST_MEAN_2, n]
      return mean + (n > 0) * jnp.exp(0.5 * coef[POST_LOGVAR, n]) * eps, None

    x, _ = jax.lax.scan(body, x, (jnp.arange(steps)[::-1], noise))
    return x

  def generate_action(self, obs, *, critic=None, key=None):
    """No entropy estimate (alpha has its own loss), so the second output is None."""
    B, N = obs.shape[0], (self.config.num_particles if critic is not None else 1)
    key, noise_key = jax.random.split(self.next_key(key))
    if N == 1:
      action = jnp.clip(self.denoise(obs, key), -1.0, 1.0)
    else:
      candidates = jax.vmap(lambda k: jnp.clip(self.denoise(obs, k), -1.0, 1.0))(jax.random.split(key, N))
      q = frozen(critic).value(jnp.tile(obs, (N, 1)), candidates.reshape(N * B, -1)).reshape(N, B)
      action = candidates[jnp.argmax(q, axis=0), jnp.arange(B)]
    std = self.noise_alpha() * self.config.noise_scale
    action = action + std * jax.random.normal(noise_key, action.shape, action.dtype)
    return self.to_plan(jnp.clip(action, -1.0, 1.0)), None

  # ---- reverse-sampling score matching --------------------------------------
  def loss(self, obs, critic, alpha, *, key=None):
    """`alpha` is the policy's (unused) dual; this actor's own is `log_alpha`."""
    cfg, B, K, d = self.config, obs.shape[0], self.config.num_candidates, self.plan_dim
    keys = jax.random.split(self.next_key(key), 4)
    coef = self.coef[...]

    action = jax.lax.stop_gradient(self.generate_action(obs, critic=critic, key=keys[0])[0]).reshape(B, d)
    n = jax.random.randint(keys[1], (B,), 0, cfg.num_steps)
    x_t = (coef[SQRT_ALPHA_BAR, n][:, None] * action
           + coef[SQRT_ONE_MINUS, n][:, None] * jax.random.normal(keys[2], action.shape))

    eps = jax.random.normal(keys[3], (K, B, d))
    candidates = jnp.clip(x_t[None] * coef[SQRT_RECIP, n][None, :, None]
                          - eps * coef[SQRT_RECIPM1, n][None, :, None], -1.0, 1.0)

    alpha_sg = jax.lax.stop_gradient(self.noise_alpha())
    qv = frozen(critic).value(jnp.tile(obs, (K, 1)), candidates.reshape(K * B, d)).reshape(K, B)
    weight = jax.lax.stop_gradient(jax.nn.softmax(qv * cfg.weight_scale / alpha_sg, axis=0))

    eps_pred = self.predict_eps(obs, x_t, n)
    actor_loss = (weight[..., None] * (eps_pred[None] - eps) ** 2).mean()

    noise_entropy = 0.5 * d * jnp.log(2 * jnp.pi * jnp.e * (cfg.noise_scale * alpha_sg) ** 2)
    alpha_loss = self.log_alpha[...] * (noise_entropy - (-cfg.target_entropy_scale * d))

    self.metrics.scalar("alpha", alpha_sg)
    self.metrics.scalar("q_spread", jax.lax.stop_gradient(qv.std(axis=0).mean()))
    self.metrics.dist("weight_ess", 1.0 / (weight ** 2).sum(axis=0))
    self.metrics.dist("weight_max", weight.max(axis=0))
    self.metrics.vec("eps_pred", eps_pred)
    return {"actor": actor_loss, "alpha": alpha_loss}


@dataclass(frozen=True)
class SDACPaperPolicyConfig(PolicyConfig):
  utd_ratio: float = 0.2
  policy_delay: int = 2
  random_steps: int = 30000
  delay_alpha_update: int = 250


@register_policy
class SDACPaper(RLPolicy):
  """`PaperDDPMActor` on the shared actor-critic template. The release updates
  the policy every 2nd step and alpha every `delay_alpha_update`-th."""

  actor_cls = PaperDDPMActor

  @classmethod
  def config_problems(cls, params) -> list[str]:
    return super().config_problems(params) + (
      [f"delay_alpha_update={params.delay_alpha_update!r}: must be >= 1"]
      if params.delay_alpha_update < 1 else [])

  def optimizer_cadence(self) -> dict:
    return {"actor": self.policy_delay, "alpha": self.params.delay_alpha_update}


@register_model
@dataclass(frozen=True)
class SDACPaperModelConfig(ModelConfig):
  policy_class: str = "SDACPaper"
  params: SDACPaperPolicyConfig = field(default_factory=SDACPaperPolicyConfig)
  components: ComponentsConfig = field(default_factory=lambda: ComponentsConfig(
    actor=PaperDDPMActorConfig(), critic=C51CriticConfig(), entropy=None))
