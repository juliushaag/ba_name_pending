import math

import torch
from torch import nn
from torch.nn import functional as F

from barl.networks.regulator import AutotuneRegulator
from barl.policy import Policy, register_policy
from barl.networks import bfloat16, float32
from barl.networks.actor import ScoreContinuousActor
from barl.networks.critic import QCriticBase
from barl.networks.ema import EMA, Shadow
from barl.networks.flow import sinusoidal_emb
from barl.networks.squash import Squash

import gymnasium as gym


@register_policy
class AMDP(Policy):
  """AMDP (Thilges et al. 2026): direct-score regression against a critic's
  pessimistic action-gradient, over a ScoreContinuousActor's forward-SDE
  vector field, with an entropy dual and a trust-region dual regularizing
  against `actor_old` (a Shadow snapshot resynced once per epoch)."""

  def __init__(self, env: gym.Env, **kwargs):

    super().__init__(env, **kwargs)

    params = self.resolve_params()
    self.entropy_scale        = params["entropy_scale"]

    assert isinstance(env.action_space, gym.spaces.Box)

    self.critic : QCriticBase = self.build_component("critic", QCriticBase, self.observation_dim, self.action_dim, pessimistic_score=True)
    self.actor  : ScoreContinuousActor = self.build_component("actor", ScoreContinuousActor, self.observation_dim, self.action_dim)

    self.actor_old = EMA(self.actor)


    self.entropy : AutotuneRegulator = self.build_component("entropy", AutotuneRegulator, target=self.action_dim * self.entropy_scale, name="entropy")
    self.tr_lambda : AutotuneRegulator = self.build_component("trust_region", AutotuneRegulator, name="trust_region")

    self.squash = self.build_component("squash", Squash, erf_k=(2 * self.actor.terminal_var) ** -0.5)

  def on_epoch_start(self):
    self.critic.on_epoch_start()
    # self.actor_old.on_epoch_start()
    return super().on_epoch_start()

  def create_optimizer(self):
    return {
      **self.critic.create_optimizer(),
      **self.actor.create_optimizer(),
      **self.entropy.create_optimizer(),
    }

  def train_step(self, batch) -> tuple[dict, dict]:

    obs      = batch["obs"].to(self.device)
    next_obs = batch["next_obs"].to(self.device)

    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)


    self.critic.step()
    self.actor_old.step()

    alpha     = self.entropy.value()
    tr_lambda = self.tr_lambda.value()

    losses, log = {}, {}

    with bfloat16():
      with torch.no_grad():
        next_X, entropy_penalty = self.actor.generate_action(next_obs)
        entropy_bonus = (alpha * entropy_penalty).detach()
        next_action = self.squash(next_X)


      with float32():
        self.critic.loss(obs, action, reward, term, next_obs, next_action, entropy_bonus, log, losses)
        
      tau = torch.rand(obs.shape[0], device=obs.device)

      sigma_tau = self.actor.sigma_fn(tau).unsqueeze(-1)
      sig_sq_tau = self.actor.sigma_sq_int_fn(tau).unsqueeze(-1)
      sig_sq_one = self.actor.sigma_sq_int_fn(torch.full((obs.shape[0],), self.actor.tau_max, device=obs.device)).unsqueeze(-1)

      # !!! Very important to sample X from the policy NOT from the batch !!!
      var = (sig_sq_tau - (sig_sq_tau ** 2 / sig_sq_one)).clamp_min(0.0)
      X_t = next_X * (sig_sq_tau / sig_sq_one) + var.sqrt() * torch.randn_like(next_X)

      time_embed = sinusoidal_emb(tau.view(obs.shape[0], 1), self.actor.time_dim)

      # Taking next_X because freeee 
      u_tilde = self.actor.diffusion_act(next_obs, X_t, time_embed)
      # u_old = self.actor_old.target.diffusion_act(obs, X_t, time_embed)

      def q_values(x):
        return self.critic.value(next_obs, self.squash(x)).sum()
      

      with float32():

        x_score : torch.Tensor = torch.func.grad(q_values)(next_X).detach()

        u      = sigma_tau * u_tilde
        target = sigma_tau * x_score.div(alpha)
        amdp_loss = (0.5 * (u - target).pow(2).sum(-1) / sigma_tau.squeeze(-1)).mean()

        # tr_loss   = 0.5 * F.mse_loss(u.float(), u_old.float())

        # self.tr_lambda.loss(tr_loss, log, losses)
        self.entropy.loss(entropy_penalty.detach(), log, losses)

        actor_loss     = amdp_loss # + tr_lambda * (tr_loss - self.tr_lambda.target())

    losses["actor"]     = actor_loss

    log.update({
      "stats/amdp":                     amdp_loss.detach(),
      "stats/amdp_target" :             target.detach().mean(),
      "stats/u_norm":                   u.detach().norm(dim=-1).mean(),  
      "stats/score_norm":               x_score.detach().norm(dim=-1).mean(),
      "stats/reward_mean" :             reward.mean(),
      "stats/act_abs" :                 next_action.abs().mean(),
    })
    return losses, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = observation.to(self.device)
    X, ell = self.actor.generate_action(obs)
    action = self.squash(X)
    return action, { }
