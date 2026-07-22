import math

import torch
from torch import nn
from torch.nn import functional as F

from barl.policy import Policy, HParam, register_policy
from barl.networks import bfloat16, float32
from barl.networks.actor import ScoreContinuousActor
from barl.networks.critic import QCriticBase
from barl.networks.ema import Shadow
from barl.networks.flow import sinusoidal_emb
from barl.networks.squash import Squash

import gymnasium as gym


@register_policy
class AMDP(Policy):
  """AMDP (Thilges et al. 2026): direct-score regression against a critic's
  pessimistic action-gradient, over a ScoreContinuousActor's forward-SDE
  vector field, with an entropy dual and a trust-region dual regularizing
  against `actor_old` (a Shadow snapshot resynced once per epoch)."""

  tr_lambda_init = HParam(0.5,   "Trust region lambda (dual variable init)",  range=(0.01, 5.0))
  tr_epsilon     = HParam(0.1,   "Trust region KL/L2 bound",                  range=(0.01, 1.0))

  def __init__(self, env: gym.Env, **kwargs):

    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    self.critic : QCriticBase = self.build_component("critic", QCriticBase, self.observation_dim, self.action_dim, pessimistic_score=True)
    self.actor  : ScoreContinuousActor = self.build_component("actor", ScoreContinuousActor, self.observation_dim, self.action_dim)

    self.actor_old = Shadow(self.actor)

    self.entropy_target = -1.0 * self.action_dim

    self.log_alpha     = nn.Parameter(torch.tensor(math.log(1.0)))
    self.log_tr_lambda = nn.Parameter(torch.tensor(math.log(self.tr_lambda_init)))

    sig_sq_one = self.actor.sigma_sq_int_fn(torch.tensor(self.actor.tau_max))
    self.squash = self.build_component("squash", Squash, erf_k=(2 * sig_sq_one) ** -0.5)

  def clamp_tr_lambda(self, lo: float = 1e-4, hi: float = 10.0):
    with torch.no_grad():
      self.log_tr_lambda.clamp_(math.log(lo), math.log(hi))

  def clamp_log_alpha(self, lo: float = 1e-4, hi: float = 5.0):
    with torch.no_grad():
      self.log_alpha.clamp_(math.log(lo), math.log(hi))

  def on_epoch_start(self):
    self.critic.on_epoch_start()
    self.actor_old.on_epoch_start()
    return super().on_epoch_start()

  def create_optimizer(self):
    return {
      **self.critic.create_optimizer(),
      **self.actor.create_optimizer(),
      "alpha":     torch.optim.Adam([self.log_alpha],     lr=1e-3),
      "tr_lambda": torch.optim.Adam([self.log_tr_lambda], lr=1e-3, maximize=True),
    }

  def train_step(self, batch) -> tuple[dict, dict]:

    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))

    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    if "X" in batch:
      X     = batch["X"].to(self.device)
    else:
      X = self.squash.inverse(action)
      
    self.clamp_tr_lambda()
    self.clamp_log_alpha()

    alpha     = self.log_alpha.exp().detach()
    tr_lambda = self.log_tr_lambda.exp()

    losses, log = {}, {}

    with bfloat16():

      with torch.no_grad():
        
        next_action, entropy_penalty = self.actor.generate_action(next_obs)

        next_action = self.squash(next_action)
        penalty = (alpha * entropy_penalty).detach()

      self.critic.loss(obs, action, reward, term, next_obs, next_action, penalty, log, losses)

      with float32():
        entropy_target_loss = (self.log_alpha.float() * (entropy_penalty.detach().clamp(-100, 100).float() - self.entropy_target)).mean()

      tau = torch.rand((obs.shape[0],), device=obs.device)

      sigma_tau = self.actor.sigma_fn(tau).unsqueeze(-1)
      sig_sq_tau = self.actor.sigma_sq_int_fn(tau).unsqueeze(-1)
      sig_sq_one = self.actor.sigma_sq_int_fn(torch.full((obs.shape[0],), self.actor.tau_max, device=obs.device)).unsqueeze(-1)

      X_t = X * (sig_sq_tau / sig_sq_one) + (sig_sq_tau - (sig_sq_tau ** 2 / sig_sq_one)).sqrt() * torch.randn_like(X)

      time_embed = sinusoidal_emb(tau.view(obs.shape[0], 1), self.actor.time_dim)

      u     = self.actor.diffusion_act(obs, X_t, time_embed) * sigma_tau
      u_old = self.actor_old.target.diffusion_act(obs, X_t, time_embed)   * sigma_tau

      def q_values(x):
        a = self.squash(x)
        return self.critic.target_value(obs, a).sum()

      x_score = torch.func.grad(q_values)(X)

      with float32():
        amdp_target = sigma_tau * (1 / alpha) * x_score

        amdp_loss = 0.5 * ((u.float() - amdp_target.float()) ** 2 / sigma_tau).mean()
        tr_loss   = 0.5 * F.mse_loss(u.float(), u_old.float())

        tr_lambda_loss = tr_lambda * (tr_loss.detach() - self.tr_epsilon)
        actor_loss     = amdp_loss + tr_lambda.detach() * (tr_loss - self.tr_epsilon)

    losses["actor"]     = actor_loss
    losses["alpha"]     = entropy_target_loss
    losses["tr_lambda"] = tr_lambda_loss

    log.update({
      "train/loss/actor":          actor_loss.detach(),
      "train/loss/alpha":          entropy_target_loss.detach(),
      "train/loss/trust_region":   tr_loss.detach().mean(),

      "stats/amdp":                     amdp_loss.detach(),
      "stats/alpha":                    alpha.detach(),
      "stats/entropy":                  entropy_penalty.detach().mean(),
      "stats/tr_lambda":                tr_lambda.detach(),
      "stats/score_norm":               x_score.detach().norm(dim=-1).mean(),
    })
    return losses, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    X, _ = self.actor.generate_action(obs)
    action = self.squash(X)
    return action, { "X" : X }
