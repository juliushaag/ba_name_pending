import torch
from torch import nn

from barl.policy import Policy, register_policy
from barl.networks.critic import QCriticBase
from barl.networks import bfloat16, float32
from barl.schedule import Schedule

import gymnasium as gym


@register_policy
class NCLQLPolicy(Policy):
  """Noise-Conditioned Langevin Q-Learning (Ki et al. 2026).

  Actor-free. There is no policy network, no entropy estimate and no entropy
  dual -- the whole DIME apparatus of `actor` + `AutotuneRegulator` disappears,
  because grad_a log pi_soft(a|s) = grad_a Q(s,a) means the target score is
  already available from policy evaluation. Learning reduces to fitting one
  noise-conditioned critic; the policy is annealed Langevin dynamics on its
  action-gradient field.

  The `q_net` slot is an ordinary QCriticBase: level conditioning is folded into
  its action input by `augment_action`, so it never learns that levels exist and
  any ensemble/quantile/categorical critic can fill the slot. Index convention
  follows the paper -- level 0 == sigma_1 (coarsest), level L-1 == sigma_L
  (clean); annealing runs 0 -> L-1.
  """

  def __init__(self, env: gym.Env, **kwargs):

    super().__init__(env, **kwargs)

    params = self.resolve_params()
    embed_dim           = params["level_embed_dim"]

    self.steps_per_level    = params["steps_per_level"]
    self.step_size          = float(params["step_size"])
    self.temperature        = float(params["temperature"])
    self.normalize_score    = params["normalize_score"]
    self.drift_scale        = float(params["drift_scale"])

    self.clip_actions       = params["clip_actions"]
    self.action_limit       = params["action_limit"]
    self.init_scale         = params["init_scale"]

    assert isinstance(env.action_space, gym.spaces.Box)

    # The ladder sigma_1 > ... > sigma_L. The paper reports only its two
    # endpoints (Table 6), so the spacing is the `sigmas` slot's `class:` --
    # ExponentialSchedule is the geometric Song & Ermon (2019) default.
    self.sigmas_sched: Schedule = self.build_component("sigmas", Schedule)

    self.register_buffer("sigmas", self.sigmas_sched.ladder())
    # alpha_i / eps = sigma_i^2 / sigma_L^2   (Definition 2)
    self.register_buffer("alpha_ratio", (self.sigmas / self.sigmas[-1]).pow(2))

    self.level_embed = nn.Embedding(self.sigmas.numel(), embed_dim)
    nn.init.normal_(self.level_embed.weight, std=0.02)


    self.critic: QCriticBase = self.build_component("critic", QCriticBase, env_stats=self.env_stats.derive(act_dim=self.action_dim + embed_dim))

  @property
  def num_levels(self):
    return self.sigmas.numel()

  @property
  def clean_level(self):
    return self.num_levels - 1
  
  def create_optimizer(self):
    opts = self.critic.create_optimizer()
    # The level embedding is a Policy-level parameter, so no component's
    # optimizer covers it -- without this it stays at its init forever.
    opts["q"].add_param_group({"params": list(self.level_embed.parameters())})
    return opts

  def augment_action(self, act, level):
    if not torch.is_tensor(level):
      level = torch.full((act.shape[0],), level, device=act.device, dtype=torch.long)
    return torch.cat([act, self.level_embed(level)], dim=-1)

  def critic_score(self, obs, act, level):
    """expectation(), not value(): the Langevin chain *is* the policy here, so a
    pessimistic bound would sample from a different distribution than exp(wQ)/Z
    rather than merely shrink a value estimate."""
    return torch.func.grad(lambda a: self.critic.expectation(obs, self.augment_action(a, level)).sum())(act)

  def sample(self, obs: torch.Tensor):
    """Annealed Langevin dynamics (Definition 2 / Algorithm 6) -- the policy:

        alpha_i = eps * sigma_i^2 / sigma_L^2
        a_{i,t} = a_{i,t-1} + drift_i(a_{i,t-1}) + sqrt(alpha_i) * z_t

    with hand-off a_{i+1,0} <- a_{i,T} across levels, so coarse levels take
    large, heavily-perturbed jumps across a smoothed landscape and the final
    level is a near-deterministic refinement whose stationary law is exp(wQ)/Z.

    The two drift modes are not the same algorithm. `normalize_score` off is the
    paper exactly -- (alpha_i/2) * w * grad_a~ Q -- and works because alpha_i's
    sigma_i^2 cancels the score's ~1/sigma_i^2 magnitude, leaving an O(1) step
    down the whole ladder. Normalizing to a unit vector bounds an untrustworthy
    critic gradient but destroys exactly that cancellation, so the normalized
    drift instead gets its own level-appropriate length, `drift_scale * sigma_i`:
    each step moves a fixed fraction of the level's own smoothing scale, and the
    drift-to-noise ratio `drift_scale / (sqrt(eps)/sigma_L)` stays constant
    across levels rather than collapsing at one end. `w` is a temperature on Q
    and means nothing once the score is normalized, so it applies only to the
    exact path."""
    # No inference_mode(False) guard here: torch.func.grad does silently return a
    # zero gradient under inference_mode, but no caller enters it any more, and
    # nesting the context manager inside torch.func.grad is what stopped this
    # from tracing once both train_step and inference_step were compiled.
    with torch.no_grad():
      dev = obs.device

      a   = self.init_scale * (2.0 * torch.rand(obs.shape[0], self.action_dim, device=dev) - 1.0)

      score_norm = a.new_zeros(())

      for i in range(self.num_levels):
        alpha = self.step_size * self.alpha_ratio[i]
        for _ in range(self.steps_per_level):
          g = self.critic_score(obs, a, i)
          score_norm = g.norm(dim=-1).mean()

          if self.normalize_score:
            drift = self.drift_scale * self.sigmas[i] * g / (g.norm(dim=-1, keepdim=True) + 1e-8)
          else:
            drift = 0.5 * alpha * self.temperature * g

          a = a + drift + alpha.sqrt() * torch.randn_like(a)

          if self.clip_actions:
            a = a.clamp(-self.action_limit, self.action_limit)

      return a, score_norm
  
  def train_step(self, batch) -> tuple[dict, dict]:

    obs      = self.normalize_obs(batch["obs"].to(self.device))
    next_obs = self.normalize_obs(batch["next_obs"].to(self.device))

    action   = batch["act"].to(self.device)
    reward   = batch["rew"].to(self.device)
    term     = batch["term"].to(self.device)


    losses = {}

    with float32():
      next_action, score_norm = self.sample(next_obs)
      next_action = next_action.detach()

    clean = self.clean_level

    # -- Boostrap loss
    with bfloat16():
      with torch.no_grad():
        bootstrap = self.critic.bootstrap(next_obs, self.augment_action(next_action, clean))
        q_target  = self.critic.bellman_target(reward, term, bootstrap)

      self.critic.loss(obs, self.augment_action(action, clean), q_target, losses)
      td_loss = losses.pop("q")

    # -- Smoothing loss
    with bfloat16():
      idx   = torch.randint(0, clean, (obs.shape[0],), device=self.device)
      noisy = action + self.sigmas[idx].unsqueeze(-1) * torch.randn_like(action)

      with torch.no_grad():
        smoothing_target = self.critic.terminal_target(
          self.critic.expectation(obs, self.augment_action(action, clean), target=True))

      self.critic.loss(obs, self.augment_action(noisy, idx), smoothing_target, losses)
      smoothing_loss = losses.pop("q")

    losses["q"] = td_loss + smoothing_loss

    self.metrics.scalar("td_loss", td_loss.detach())
    self.metrics.scalar("smoothing_loss", smoothing_loss.detach())
    self.metrics.vec("action", next_action, limit=1.0)
    self.metrics.scalar("score_norm", score_norm)

    return losses

  @torch.no_grad()
  def inference_step(self, observation: torch.Tensor, step_info=None):
    obs = self.normalize_obs(observation.to(self.device))
    with float32():
      action, _ = self.sample(obs)
    return action, {}
