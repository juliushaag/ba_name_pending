import math

import gymnasium as gym
import torch
from torch import nn

from barl.networks.regulator import AutotuneRegulator
from barl.policy import Policy, register_policy
from barl.networks.actor import ScoreDiscreteActor
from barl.networks.critic import QCriticBase
from barl.networks import bfloat16, float32
from barl.networks.squash import Squash


@register_policy
class SDAC(Policy):
  """Soft Diffusion Actor-Critic (Ma et al. 2025, "Efficient Online RL for
  Diffusion Policy").

  Policy improvement is reweighted score matching (Eq. 14): candidates are drawn
  from the closed-form phi_{0|t}, weighted by softmax(Q/alpha), and regressed
  onto the denoising score target. No backprop through the reverse chain -- that
  is the entire point of RSM, and it is what makes SDAC cheap relative to DIME.

  Action selection is Best-of-N over diffusion samples plus additive Gaussian
  exploration noise. Entropy is the log-likelihood of that additive noise with
  the BoN result treated as deterministic; the reverse-chain ELBO (ScoreDiscrete
  Actor.generate_action's `ell`) is a DIME construct and is NOT used here.

  Divergences from the paper, called out because they are deliberate:

    * Ma et al. list no temperature learning rate (Table 4) -- alpha is a fixed
      hyperparameter there. Autotuning alpha against a constant entropy is
      ill-posed (the dual gradient never changes sign), so `explore_std` is made
      learnable and trained against Q + alpha*H. That closes the loop: alpha up
      -> sigma up -> H up -> alpha back down. To run the paper version instead,
      freeze `_log_std` and hold alpha constant.
  """

  def __init__(self, env: gym.Env, **kwargs):

    super().__init__(env, **kwargs)

    params = self.resolve_params()
    self.entropy_scale  = params["entropy_scale"]
    self.num_candidates = params["num_candidates"]   # K: phi_{0|t} candidates for the softmax weight
    self.num_bon        = params["num_candidates"]          # N: Best-of-N action selection
    self.explore_init   = params["explore_init"]

    assert isinstance(env.action_space, gym.spaces.Box)

    self.actor  : ScoreDiscreteActor = self.build_component("actor", ScoreDiscreteActor, self.observation_dim, self.action_dim)
    self.critic : QCriticBase        = self.build_component("critic", QCriticBase, self.observation_dim, self.action_dim, pessimistic_score=True)
    self.squash : Squash             = self.build_component("squash", Squash)

    
    self.entropy : AutotuneRegulator = self.build_component(
        "entropy", AutotuneRegulator, target=self.entropy_scale * self.action_dim, name="entropy")

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

  def on_epoch_start(self):
    self.critic.on_epoch_start()
    return super().on_epoch_start()

  def create_optimizer(self):
    return {
        **self.critic.create_optimizer(),
        **self.actor.create_optimizer(),
        **self.entropy.create_optimizer(),
        "explore": torch.optim.Adam([self._log_std], lr=1e-2),
    }

  def train_step(self, batch) -> tuple[dict, dict]:

    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))

    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    alpha = self.entropy.value()

    B, K = obs.shape[0], self.num_candidates
    losses, log = {}, {}

    with bfloat16():

      # ---- critic -------------------------------------------------------
      with torch.no_grad():
        next_action = self.explore(self.best_of_n(next_obs, self.num_bon))
        entropy_bonus_val = (alpha * self.noise_entropy()).detach().reshape(1, 1).expand(B, 1)

      self.critic.loss(obs, action, reward, term, next_obs, next_action, entropy_bonus_val, log, losses)

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

        dsm_w = (1.0 - self.actor.alpha_bars[t]).unsqueeze(-1)            # (B, 1)

      score_pred = self.actor.predict_score(obs, x_t, t)                  # (B, D)

    with float32():
      sdac_loss = (weight * dsm_w * (score_pred.unsqueeze(0) - target).pow(2)).sum(-1).sum(0).mean()

      a_bon    = self.best_of_n(obs, self.num_bon)                        # already detached (no_grad)
      a_noisy  = a_bon + self.explore_std * torch.randn_like(a_bon)
      std_loss = -(self.critic.value(obs, a_noisy).mean() + alpha * self.noise_entropy())

      self.entropy.loss(self.noise_entropy(), log, losses)

    losses["actor"]   = sdac_loss
    losses["explore"] = std_loss

    log["stats/score_pred_norm"] = score_pred.detach().norm(dim=-1).mean()
    log["stats/score_tgt_norm"]  = target.detach().norm(dim=-1).mean()
    log["stats/explore_std"]     = self.explore_std.detach().mean()
    log["stats/noise_entropy"]   = self.noise_entropy().detach()
    log["stats/weight_ess"]      = (1.0 / weight.squeeze(-1).pow(2).sum(0)).mean()  # effective sample size, in [1, K]

    return losses, log

  @torch.inference_mode()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    act = self.best_of_n(obs, self.num_bon)
    return self.explore(act), {}