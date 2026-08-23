import math

import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.actor import DiscreteDiffusionActorBase
from barl_jax.networks.critic import QCriticBase
from barl_jax.networks.registry import register_component
from barl_jax.networks.regulator import AutotuneRegulator
from barl_jax.networks.squash import Squash
from barl_jax.policy import Buffer, Policy, frozen, register_policy


def _inv_sigmoid(y: float) -> float:
  return math.log(y / (1.0 - y))


@register_component
class OUDiffusionActor(DiscreteDiffusionActorBase):
  """DIME (Celik et al. 2025). Ornstein-Uhlenbeck forward process with prior
  N(0, eta^2 I) (Table 2), and a per-dimension learnable scaling of the beta
  ladder."""

  def __init__(self, env_stats, *, latent_dim: int, layers: int, num_steps: int,
               beta_min: float, beta_max: float, beta_scale_min: float,
               beta_scale_max: float, learn_beta_scale: bool,
               prior_var: float, anchor_score: bool,
               euler_max_step: float, min_total_noise: float,
               rngs: nnx.Rngs, **kwargs):

    super().__init__(env_stats, latent_dim=latent_dim, layers=layers, num_steps=num_steps,
                     rngs=rngs, **kwargs)

    # Used to be module constants, which meant the bound that silently clamps
    # beta_scale_max below was invisible to every run that hit it.
    self.euler_max_step  = float(euler_max_step)
    self.min_total_noise = float(min_total_noise)
    self.anchor_score = bool(anchor_score)
    self.eta2 = float(prior_var)
    self.eta  = math.sqrt(prior_var)
    self.inv_eta2 = 1.0 / prior_var

    t    = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=jnp.float32)
    beta = beta_min + 0.5 * (beta_max - beta_min) * (1.0 - jnp.cos(math.pi * t))
    self.beta = Buffer(beta)
    self.logvar_offset = Buffer(0.5 * self.action_dim * (jnp.log(beta[1:]) - jnp.log(beta[:-1])))

    lo = self.beta_scale_min = float(beta_scale_min)
    hi = self.beta_scale_max = min(float(beta_scale_max),
                                   self.euler_max_step / float(beta[1:].max()))
    raw = jnp.full((self.action_dim,), _inv_sigmoid((1.0 - lo) / (hi - lo)), jnp.float32)
    # Trainable only when it is meant to be: a frozen Buffer keeps it out of
    # every optimizer, which is what `requires_grad=False` did in torch.
    self.beta_scale_raw = nnx.Param(raw) if learn_beta_scale else Buffer(raw)

    self.prior_logZ = 0.5 * self.action_dim * math.log(2 * math.pi * prior_var)
    self._assert_mixed()

  def _assert_mixed(self):
    # The reverse update is an Euler step, only stable while beta*scale stays
    # small; and the forward process only reaches the N(0, eta^2) prior if it
    # accumulates enough total noise (signal decays as exp(-total)).
    used  = self.beta[...][1:, None] * self.beta_scale
    total = float(used.sum(axis=0).min())
    peak  = float(used.max())
    assert peak <= self.euler_max_step, (
      f"beta*scale peaks at {peak:.3f} > euler_max_step {self.euler_max_step} "
      f"-- lower beta_max or beta_scale_max")
    assert total >= self.min_total_noise, (
      f"total forward noise {total:.3f} < min_total_noise {self.min_total_noise} "
      f"(signal at step N still {math.exp(-total):.3f}) -- raise num_steps or beta_min/beta_max")

  @property
  def beta_scale(self):
    lo, hi = self.beta_scale_min, self.beta_scale_max
    return lo + (hi - lo) * jax.nn.sigmoid(self.beta_scale_raw[...])

  # ---- hooks -------------------------------------------------------------
  def prior_sample(self, batch: int, key):
    return self.eta * jax.random.normal(key, (batch, self.action_dim), jnp.float32)

  def to_score(self, raw, x, n):
    return raw - x * self.inv_eta2 if self.anchor_score else raw

  def init_aux(self, x):
    return self.prior_logZ + 0.5 * self.inv_eta2 * (x ** 2).sum(axis=-1, keepdims=True)

  def _noise_table(self):
    b    = self.beta[...][1:, None] * self.beta_scale          # (N, D)
    keep = 1.0 - b
    gamma = jnp.concatenate([jnp.ones_like(b[:1]), jnp.cumprod(keep, axis=0)], axis=0)
    w, cur = [jnp.zeros_like(b[0])], jnp.zeros_like(b[0])
    for k in range(self.num_steps):
      cur = keep[k] ** 2 * cur + b[k]                          # eta-free
      w.append(cur)
    return gamma, jnp.stack(w)

  def marginal_coef(self, n):
    g, w = self._noise_table()
    return g[n], jnp.maximum(2.0 * self.eta2 * w[n], 1e-6)

  def reverse_step(self, obs, x, n, key):
    e2, scale = self.eta2, self.beta_scale
    b_rev, b_fwd = self.beta[...][n] * scale, self.beta[...][n - 1] * scale
    var_rev, var_fwd = 2.0 * e2 * b_rev, 2.0 * e2 * b_fwd      # (D,), broadcast

    eps   = jax.random.normal(key, x.shape, x.dtype)
    x_new = x + b_rev * x + 2.0 * e2 * b_rev * self.predict_score(obs, x, n) \
            + jnp.sqrt(var_rev) * eps
    resid = x - (x_new - b_fwd * x_new)
    inc   = (0.5 * eps ** 2 - 0.5 * resid ** 2 / var_fwd).sum(axis=-1, keepdims=True) \
            + self.logvar_offset[...][n - 1]
    return x_new, inc


@register_policy
class DIME(Policy):
  """Diffusion-policy MaxEnt actor-critic over an OUDiffusionActor's DDPM-style
  reverse process, with an entropy dual auto-tuned against a target entropy.

  The actor loss is differentiated through the *whole* reverse chain -- 16
  network evaluations deep with a critic on top -- which is what makes this the
  most expensive train_step in the framework."""

  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    params = self.resolve_params()
    self.entropy_scale = params["entropy_scale"]
    self.policy_delay           = params["policy_delay"]

    self.critic: QCriticBase = self.build_component("critic", QCriticBase)
    self.actor: OUDiffusionActor = self.build_component("actor", OUDiffusionActor)
    self.squash: Squash = self.build_component("squash", Squash)
    self.entropy: AutotuneRegulator = self.build_component(
      "entropy", AutotuneRegulator, target=self.action_dim * self.entropy_scale)

  def _squash_correction(self, X):
    """The change of variables taking the entropy of the raw terminal state to
    the entropy of the squashed action: H(y) = H(x) + E[log|dy/dx|].

    `log_abs_det_jacobian`, not `inverse`. The reference implementation
    (dime_am, od_sampling.single_sample) adds Tanh().forward_log_det_jacobian
    here; atanh of `X` is not a change of variables at all -- and since `X` is
    the *pre*-tanh latent, TanhSquash.inverse's clamp pinned every |x| > 1 to a
    constant +-7.25, so most of the batch contributed no gradient and a wrong
    value."""
    return self.squash.log_abs_det_jacobian(X).sum(axis=-1, keepdims=True)


  def optimizer_cadence(self) -> dict:
    """dime_am's `policy_delay`: the critic steps every gradient step, the actor
    and the entropy dual only every `policy_delay`-th -- both gated by the same
    branch there, so both are gated by the same key here."""
    return {"actor": self.policy_delay, "entropy_alpha": self.policy_delay}

  def train_step(self, batch):
    obs      = self.normalize_obs(batch["obs"])
    next_obs = self.normalize_obs(batch["next_obs"])
    action   = batch["act"]
    reward   = batch["rew"]
    term     = batch["term"]

    alpha  = self.entropy.value()
    losses = {}

    # ---- critic: the bootstrap sample is a target, not a path to train through
    next_X, entropy = jax.lax.stop_gradient(self.actor.generate(next_obs))
    entropy = entropy + self._squash_correction(next_X)
    next_action = self.squash(next_X)

    entropy_bonus = jax.lax.stop_gradient(alpha * entropy)
    self.critic.td_loss(obs, action, next_obs, next_action, reward, term,
                        losses, shift=entropy_bonus)

    # ---- actor: backprop through the reverse chain into the diffusion net
    new_X, new_entropy = self.actor.generate(obs)
    new_entropy = new_entropy + self._squash_correction(new_X)

    q_new = frozen(self.critic).value(obs, self.squash(new_X))
    dime_loss = -(q_new + jax.lax.stop_gradient(alpha) * new_entropy).mean()

    self.entropy.loss(entropy, losses)
    losses["actor"] = dime_loss

    self.actor.metrics.dist("entropy", entropy)
    self.actor.metrics.vec("action", next_action, limit=1.0)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_obs(observation)
    return self.squash(self.actor.sample(obs, key)), {}
