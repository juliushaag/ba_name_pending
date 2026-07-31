import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from barl.networks.registry import register_component
from barl.networks.regulator import AutotuneRegulator
from barl.policy import Policy, register_policy
from barl.networks.actor import DiscreteDiffusionActorBase
from barl.networks.critic import QCriticBase
from barl.networks import bfloat16, float32
from barl.networks.squash import Squash

import gymnasium as gym

def _inv_softplus(y: float) -> float:
  return math.log(math.expm1(y))


def _inv_sigmoid(y: float) -> float:
  return math.log(y / (1.0 - y))

@register_component
class OUDiffusionActor(DiscreteDiffusionActorBase):
  """DIME (Celik et al. 2025). Prior N(0, eta^2 I) (Table 2). ."""

  EULER_MAX_STEP = 0.35
  MIN_TOTAL_NOISE = 2.0

  def __init__(self, env_stats, latent_dim, layers, num_steps,
               beta_min, beta_max, beta_scale_min=0.7, beta_scale_max=1.8,
               learn_beta_scale=True, prior_var=2.5, anchor_score=True, **kw):

    super().__init__(env_stats, latent_dim, layers, num_steps, **kw)
    self.anchor_score = anchor_score
    self.eta2, self.eta, self.inv_eta2 = prior_var, math.sqrt(prior_var), 1.0 / prior_var

    t    = torch.linspace(0.0, 1.0, num_steps + 1)
    beta = beta_min + 0.5 * (beta_max - beta_min) * (1.0 - torch.cos(math.pi * t))
    self.register_buffer("beta", beta)
    self.register_buffer("logvar_offset", 0.5 * self.action_dim * (beta[1:].log() - beta[:-1].log()))

    lo = self.beta_scale_min = beta_scale_min
    hi = self.beta_scale_max = min(beta_scale_max, self.EULER_MAX_STEP / float(beta[1:].max()))
    self._beta_scale_raw = nn.Parameter(torch.full((self.action_dim,), _inv_sigmoid((1.0 - lo) / (hi - lo))),
                                        requires_grad=learn_beta_scale)

    self.prior_logZ = 0.5 * self.action_dim * math.log(2 * math.pi * prior_var)   # constant now
    self._assert_mixed()

  def _assert_mixed(self):
    # The reverse update is an Euler step, only stable while beta*scale stays
    # small; and the forward process only reaches the N(0, eta^2) prior if it
    # accumulates enough total noise (signal decays as exp(-total)).
    with torch.no_grad():
      used  = self.beta[1:].unsqueeze(-1) * self.beta_scale
      total = used.sum(0).min()
      assert used.max() <= self.EULER_MAX_STEP, (
        f"beta*scale peaks at {float(used.max()):.3f} > EULER_MAX_STEP "
        f"{self.EULER_MAX_STEP} -- lower beta_max or beta_scale_max")
      assert total >= self.MIN_TOTAL_NOISE, (
        f"total forward noise {float(total):.3f} < MIN_TOTAL_NOISE "
        f"{self.MIN_TOTAL_NOISE} (signal at step N still "
        f"{math.exp(-float(total)):.3f}) -- raise num_steps or beta_min/beta_max")

  @property
  def beta_scale(self):
    lo, hi = self.beta_scale_min, self.beta_scale_max
    return lo + (hi - lo) * torch.sigmoid(self._beta_scale_raw)

  def process_parameters(self):
    return [self._beta_scale_raw] if self._beta_scale_raw.requires_grad else []

  # ---- hooks -------------------------------------------------------------
  def prior_var(self):            return self.eta2                     # plain float
  def prior_sample(self, B, dev): return self.eta * torch.randn(B, self.action_dim, device=dev)
  def to_score(self, raw, x, n):  return raw - x * self.inv_eta2 if self.anchor_score else raw
  def init_aux(self, x):          return self.prior_logZ + 0.5 * self.inv_eta2 * x.pow(2).sum(-1, keepdim=True)

  def _noise_table(self):
    b    = self.beta[1:].unsqueeze(-1) * self.beta_scale                # (N, D)
    keep = 1.0 - b
    gamma = torch.cat([torch.ones_like(b[:1]), keep.cumprod(0)], 0)
    w, cur = [torch.zeros_like(b[0])], torch.zeros_like(b[0])
    for k in range(self.num_steps):
      cur = keep[k].pow(2) * cur + b[k]                                 # eta-free
      w.append(cur)
    return gamma, torch.stack(w)

  def marginal_coef(self, n):
    g, w = self._noise_table()
    return g[n], (2.0 * self.eta2 * w[n]).clamp_min(1e-6)

  def reverse_step(self, obs, x, n):
    e2, scale = self.eta2, self.beta_scale
    b_rev, b_fwd = self.beta[n] * scale, self.beta[n - 1] * scale
    var_rev, var_fwd = 2.0 * e2 * b_rev, 2.0 * e2 * b_fwd               # (D,), broadcast
    eps   = torch.randn_like(x)
    x_new = x + b_rev * x + 2.0 * e2 * b_rev * self.predict_score(obs, x, n) + var_rev.sqrt() * eps
    resid = x - (x_new - b_fwd * x_new)
    inc   = (0.5 * eps.pow(2) - 0.5 * resid.pow(2) / var_fwd).sum(-1, keepdim=True) \
            + self.logvar_offset[n - 1]
    return x_new, inc
  
@register_policy
class DIMEPolicy(Policy):
  """Diffusion-policy MaxEnt actor-critic over a ScoreDiscreteActor's
  DDPM-style reverse process, with an entropy dual (`log_alpha`, auto-tuned
  against `target_entropy`). Unlike AMDP, needs no squash-Jacobian correction:
  `entropy` feeds in as a plain additive bonus/dual target, not a log-density."""

  def __init__(self, env: gym.Env, **kwargs):

    super().__init__(env, **kwargs)

    params = self.resolve_params()
    self.entropy_scale        = params["entropy_scale"]

    assert isinstance(env.action_space, gym.spaces.Box)

    self.critic : QCriticBase = self.build_component("critic", QCriticBase)
    self.actor : OUDiffusionActor = self.build_component("actor", OUDiffusionActor)

    self.squash : Squash = self.build_component("squash", Squash)

    self.entropy : AutotuneRegulator = self.build_component("entropy", AutotuneRegulator, target=self.action_dim * self.entropy_scale)

  def create_optimizer(self):
    return {**self.critic.create_optimizer(), **self.actor.create_optimizer(), **self.entropy.create_optimizer()}

  def train_step(self, batch) -> tuple[dict, dict]:

    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))

    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)

    alpha = self.entropy.value()

    losses = {}

    with bfloat16():

      next_X, entropy = self.actor.generate(next_obs)
      next_action = self.squash(next_X)
 
      entropy_bonus = (alpha * entropy).detach()

      with torch.no_grad():
        bootstrap = self.critic.bootstrap(next_obs, next_action) + entropy_bonus
        q_target  = self.critic.bellman_target(reward, term, bootstrap)

      self.critic.loss(obs, action, q_target, losses)

      q_new = self.critic.value(next_obs, next_action)

      with float32():
        dime_loss = -(q_new + (alpha * entropy)).mean()
        self.entropy.loss(entropy.detach(), losses)
    
    losses["actor"]  = dime_loss

    self.actor.metrics.dist("entropy", entropy)
    self.actor.metrics.vec("action", next_action, limit=1.0)
    self.critic.metrics.dist("q_new", q_new)

    return losses

  @torch.no_grad()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    with bfloat16():
      obs = self.normalize_obs(observation.to(self.device))
      return self.squash(self.actor.sample(obs)), {}
