import math

import gymnasium as gym
import torch
from torch import nn

from barl.networks.registry import register_component
from barl.networks.regulator import AutotuneRegulator
from barl.policy import Policy, register_policy
from barl.networks.actor import DiscreteDiffusionActorBase
from barl.networks.critic import QCriticBase
from barl.networks import bfloat16, float32
from barl.networks.squash import Squash


@register_component
class VPDiffusionActor(DiscreteDiffusionActorBase):
  """Variance-preserving cosine ladder + DDPM ancestral sampler.
  Network predicts eps; to_score converts. Fixed schedule (Ma et al. Table 4)."""

  def __init__(self, env_stats, latent_dim, layers, num_steps,
               cosine_s=0.008, ab_start=0.999, ab_end=0.001, **kw):
    super().__init__(env_stats, latent_dim, layers, num_steps, **kw)
    tt = torch.linspace(0, 1, num_steps + 1)
    f  = torch.cos((tt + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
    self.register_buffer("alpha_bar", ab_end + (ab_start - ab_end) * (f / f[0]))

  def marginal_coef(self, n):
    ab = self.alpha_bar[n].unsqueeze(-1)
    return ab.sqrt(), (1.0 - ab)

  def to_score(self, raw, x, n):
    return -raw / self.marginal_coef(n)[1].sqrt()      # raw == eps_theta

  def prior_sample(self, B, dev):
    return torch.randn(B, self.action_dim, device=dev)

  def reverse_step(self, obs, x, n):
    ab_t, ab_prev = self.alpha_bar[n], self.alpha_bar[n - 1]
    a_t    = ab_t / ab_prev
    beta_t = 1.0 - a_t
    mean   = (x + beta_t * self.predict_score(obs, x, n)) / a_t.sqrt()
    if n > 1:
      var = (1.0 - ab_prev) / (1.0 - ab_t) * beta_t     # sigma_t^2 -- Eq. 8 omits the sqrt
      mean = mean + var.sqrt() * torch.randn_like(x)
    return mean, 0.0

@register_policy
class SDAC(Policy):
  """Soft Diffusion Actor-Critic (Ma et al. 2025)  """

  def __init__(self, env: gym.Env, **kwargs):

    super().__init__(env, **kwargs)

    params = self.resolve_params()
    self.entropy_scale  = params["entropy_scale"]
    self.num_candidates = params["num_candidates"]   # K: phi_{0|t} candidates for the softmax weight
    self.num_bon        = params["num_bon"]          # N: Best-of-N action selection
    self.explore_init   = params["explore_init"]

    assert isinstance(env.action_space, gym.spaces.Box)

    self.actor  : VPDiffusionActor = self.build_component("actor", VPDiffusionActor)
    self.critic : QCriticBase        = self.build_component("critic", QCriticBase)
    self.squash : Squash             = self.build_component("squash", Squash)

    
    self.entropy : AutotuneRegulator = self.build_component(
        "entropy", AutotuneRegulator, target=self.entropy_scale * self.action_dim)

    # Per-dimension exploration noise, applied in SQUASHED space (post-tanh), so
    # no Jacobian correction is needed in the entropy -- unlike DIME's ELBO,
    # which lives in raw space and needs Eq. 58/59.
    self._log_std = nn.Parameter(torch.full((self.action_dim,), math.log(self.explore_init)))

  # ---------------------------------------------------------------- utilities

  @property
  def explore_std(self) -> torch.Tensor:
    return self._log_std.clamp(math.log(1e-3), math.log(1.0)).exp()

  def noise_entropy(self) -> torch.Tensor:
    """H of the additive exploration noise, BoN output treated as deterministic."""
    return (0.5 * math.log(2 * math.pi * math.e) + self.explore_std.log()).sum()

  @torch.no_grad()
  def best_of_n(self, obs: torch.Tensor, n: int) -> torch.Tensor:
    """N diffusion samples, squashed, argmax over Q. Returns (B, action_dim)."""
    B = obs.shape[0]
    obs_rep = obs.unsqueeze(0).expand(n, B, -1).reshape(n * B, -1)
    act     = self.squash(self.actor.sample(obs_rep))                     # (n*B, D)
    qv      = self.critic.value(obs_rep, act).reshape(n, B, -1).squeeze(-1)  # (n, B)
    idx     = qv.argmax(dim=0)                                            # (B,)
    return act.reshape(n, B, -1)[idx, torch.arange(B, device=obs.device)]

  def explore(self, act: torch.Tensor) -> torch.Tensor:
    return (act + self.explore_std * torch.randn_like(act)).clamp(-1.0, 1.0)

  def create_optimizer(self):
    return {
        **self.critic.create_optimizer(),
        **self.actor.create_optimizer(),
        **self.entropy.create_optimizer(),
        "explore": torch.optim.Adam([self._log_std], lr=1e-2, fused=True),
    }

  def train_step(self, batch) -> dict:

    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))

    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    alpha = self.entropy.value()

    B, K = obs.shape[0], self.num_candidates
    losses = {}

    with bfloat16():

      # ---- critic -------------------------------------------------------
      with torch.no_grad():
        next_action = self.explore(self.best_of_n(next_obs, self.num_bon))
        entropy_bonus_val = (alpha * self.noise_entropy()).detach().reshape(1, 1).expand(B, 1)

      with torch.no_grad():
        bootstrap = self.critic.bootstrap(next_obs, next_action) + entropy_bonus_val
        q_target  = self.critic.bellman_target(reward, term, bootstrap)

      self.critic.loss(obs, action, q_target, losses)

      with torch.no_grad():
        t   = torch.randint(1, self.actor.num_steps + 1, (B,), device=self.device)
        x_t = self.actor.diffuse(self.squash.inverse(action), t)

        a0_samples = self.actor.sample_phi0(x_t, t, K)                    # (K, B, D), RAW

        qv = self.critic.value(
            obs.unsqueeze(0).expand(K, B, -1).reshape(K * B, -1),
            self.squash(a0_samples).reshape(K * B, -1),
        ).reshape(K, B, -1).squeeze(-1)                                   # (K, B)

        weight = torch.softmax(qv / alpha, dim=0).unsqueeze(-1)      # (K, B, 1), sums to 1 over K
        target = self.actor.score_target(a0_samples, x_t.unsqueeze(0), t) # (K, B, D)

        dsm_w = self.actor.dsm_weight(t)                                  # (B, 1)

      score_pred = self.actor.predict_score(obs, x_t, t)                  # (B, D)

    with float32():
      sdac_loss = (weight * dsm_w * (score_pred.unsqueeze(0) - target).pow(2)).sum(-1).sum(0).mean()

      a_bon    = self.best_of_n(obs, self.num_bon)                        # already detached (no_grad)
      a_noisy  = a_bon + self.explore_std * torch.randn_like(a_bon)
      std_loss = -(self.critic.value(obs, a_noisy).mean() + alpha.detach() * self.noise_entropy())

      self.entropy.loss(self.noise_entropy(), losses)

    losses["actor"]   = sdac_loss
    losses["explore"] = std_loss

    self.metrics.scalar("explore_std", self.explore_std.detach().mean())
    self.metrics.scalar("noise_entropy", self.noise_entropy().detach())
    # effective sample size of the softmax(Q/alpha) candidate weights, in [1, K]
    self.metrics.dist("weight_ess", 1.0 / weight.squeeze(-1).pow(2).sum(0))
    self.actor.metrics.vec("score_pred", score_pred)
    self.actor.metrics.vec("score_target", target.reshape(-1, target.shape[-1]))
    self.actor.metrics.vec("action", next_action, limit=1.0)

    return losses

  @torch.no_grad()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    act = self.best_of_n(obs, self.num_bon)
    return self.explore(act), {}