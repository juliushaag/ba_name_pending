import gymnasium as gym
import jax
import jax.numpy as jnp
from flax import nnx

from barl_jax.networks.critic import ConcatQ, QCriticBase
from barl_jax.networks.flow import TimeEmbedding

from barl_jax.networks.registry import register_component
from barl_jax.policy import Buffer, Policy, frozen, register_policy
from barl_jax.networks.schedule import Schedule


@register_component
class NCLQLCritic(QCriticBase):
  """The critic from ku-dmlab/NCLQL, as its own component rather than a
  configuration of the generic ones.

  Three things set it apart from every other critic here, all of them the
  reference's:

  * **A Polyak target network.** Every other critic in this codebase is
    target-free and leans on CrossQ's shared-forward BatchRenorm instead. That
    works when an actor keeps the bootstrap action on-distribution; NC-LQL has
    no actor -- its bootstrap action is whatever annealed Langevin lands on in
    the critic's *own* gradient field -- so the reference anchors the bootstrap
    with `min(target_q1, target_q2)` and `tau = 0.005`.
  * **Two heads, reduced differently by role.** `min` over the heads is the
    bootstrap (pessimism); their `mean` is what the Langevin chain climbs and
    what the smoothing loss regresses onto.
  * **Noise conditioning inside the critic**, so the Policy hands it a level
    index rather than pre-concatenating an embedding.

  Defaults follow the reference: no normalization, `mish`, `lr = 1e-4`."""

  def __init__(self, env_stats, latent_dim: int, layers: int, n_models: int = 2,
               time_dim: int = 16, polyak_tau: float = 0.005,
               activation: str = "mish", norm: str = "none",
               norm_input: bool = False, norm_stem: bool = False, norm_after_act: bool = True,
              norm_kwargs: dict | None = None, dtype: str = "float32",
               *, rngs: nnx.Rngs, **kwargs):
    
    self.time_dim = int(time_dim)
    
    self.time_embed = TimeEmbedding(
      time_dim=int(time_dim),
      kind="fourier",
      rngs=rngs,
    )

    q_function = ConcatQ(
      env_stats["obs_dim"] + env_stats["act_dim"] + time_dim, 1, latent_dim, layers,
      n_models=n_models, activation=activation, norm=norm,
      norm_input=norm_input, norm_stem=norm_stem,
      norm_after_act=norm_after_act, norm_kwargs=norm_kwargs, dtype=dtype, rngs=rngs
    )

    super().__init__(env_stats, q_function, rngs=rngs, **kwargs)

    self.n_models   = int(n_models)
    self.polyak_tau = float(polyak_tau)

    self.time_embed_target = nnx.clone(self.time_embed)
    self.q_target = nnx.clone(q_function)

  # ---- level conditioning -------------------------------------------------

  def augment(self, act, level, target=False):

    if not isinstance(level, jax.Array):
      level = jnp.full((act.shape[0],), level, jnp.int32)

    if not target:
      te = self.time_embed(level.astype(jnp.float32)[:, None])
    else:
      te = self.time_embed_target(level.astype(jnp.float32)[:, None])

    return jnp.concatenate([act, te], axis=-1)
    

  def expectation_at(self, obs, act, level):
    """E[Q] at a level, augmenting internally. The Langevin chain differentiates
    this, so the encoder it uses must be the one on the module it was handed --
    not one captured off `self` at another trace level."""
    return self.reduce_expectation(self.raw(obs, self.augment(act, level)))

  # ---- reductions ---------------------------------------------------------

  def reduce_expectation(self, raw):
    """E[Q] over the heads. NC-LQL's own, not part of the QCriticBase contract:
    the Langevin chain differentiates the *unbounded* mean, because there the
    gradient field is the policy and a pessimistic bound would change the
    distribution being sampled. Everything else prices through `reduce_value`."""
    return raw.astype(jnp.float32).mean(axis=0)

  def reduce_value(self, raw):
    """min over the heads -- the reference's `jnp.minimum(q1, q2)` bootstrap."""
    return raw.astype(jnp.float32).min(axis=0)

  def loss_from_raw(self, raw, target, losses: dict) -> None:
    pred = raw.astype(jnp.float32)
    tgt  = jax.lax.stop_gradient(target).astype(jnp.float32)
    losses["q"] = ((pred - jnp.broadcast_to(tgt, pred.shape)) ** 2).mean()

    self.metrics.dist("q", pred.mean(axis=0))
    self.metrics.dist("q_target", tgt)
    self.metrics.ensemble("ensemble", pred)

  # ---- target network -----------------------------------------------------

  def target_bootstrap(self, next_obs, next_act, level):
    """`min` over the *target* heads, the reference's `q_backup` input. The
    chain that produced `next_act` ran on the live critic; only the value it is
    scored with comes from the target."""
    raw = self.q_target(next_obs, self.augment(next_act, level, target=True))
    return jax.lax.stop_gradient(self.reduce_value(raw))

  def after_step(self):
    """Polyak the target toward the live weights, once per gradient step."""

    """Q-Function"""
    live = nnx.state(self.q_function, nnx.Param)
    tgt  = nnx.state(self.q_target, nnx.Param)

    nnx.update(self.q_target,
               jax.tree.map(lambda t, l: t + self.polyak_tau * (l - t), tgt, live))
    """Time-Embedding"""
    live = nnx.state(self.time_embed, nnx.Param)
    tgt  = nnx.state(self.time_embed_target, nnx.Param)
    
    nnx.update(self.time_embed_target,
               jax.tree.map(lambda t, l: t + self.polyak_tau * (l - t), tgt, live))


@register_policy
class NCLQL(Policy):
  """Noise-Conditioned Langevin Q-Learning (Ki et al. 2026).

  Actor-free. There is no policy network, no entropy estimate and no entropy
  dual -- the whole actor + AutotuneRegulator apparatus disappears, because
  grad_a log pi_soft(a|s) = grad_a Q(s,a) means the target score is already
  available from policy evaluation. Learning reduces to fitting one
  noise-conditioned critic; the policy is annealed Langevin dynamics on its
  action-gradient field.

  The `critic` slot is an ordinary QCriticBase: level conditioning is folded
  into its action input by `augment_action`, so it never learns that levels
  exist and any ensemble/quantile/categorical critic can fill the slot. Index
  convention follows the paper -- level 0 == sigma_1 (coarsest), level L-1 ==
  sigma_L (clean); annealing runs 0 -> L-1."""


  def __init__(self, env: gym.Env, **kwargs):
    super().__init__(env, **kwargs)

    assert isinstance(env.action_space, gym.spaces.Box)

    params = self.resolve_params()

    self.steps_per_level = params["steps_per_level"]
    self.noise_levels    = float(params["noise_levels"])
    self.step_size       = float(params["step_size"])
    self.temperature     = float(params["temperature"])
    self.normalize_score = params["normalize_score"]
    self.reward_scale    = float(params["reward_scale"])
    self.clip_actions    = params["clip_actions"]
    self.action_limit    = params["action_limit"]
    self.init_scale      = params["init_scale"]

   
    self.sigmas_sched: Schedule = self.build_component("sigmas", Schedule, steps=self.noise_levels)

    sigmas = self.sigmas_sched.ladder()
    self.sigmas = Buffer(sigmas)
    self.alpha_ratio = Buffer((sigmas / sigmas[-1]) ** 2)

    self.critic: NCLQLCritic = self.build_component("critic", NCLQLCritic)

  @property
  def num_levels(self) -> int:
    return int(self.sigmas[...].size)

  @property
  def clean_level(self) -> int:
    return self.num_levels - 1

  def augment_action(self, act, level):
    return self.critic.augment(act, level)

  def critic_score(self, obs, act, level):
    def q(critic, a):
      return critic.expectation_at(obs, a, level).sum()
    return nnx.grad(q, argnums=1)(frozen(self.critic), act)

  def sample(self, obs, key):
    key, init_key = jax.random.split(key)
    a = self.init_scale * (2.0 * jax.random.uniform(
      init_key, (obs.shape[0], self.action_dim), jnp.float32) - 1.0)

    score_norm = jnp.zeros(())
    keys = jax.random.split(key, self.num_levels * self.steps_per_level)

    alpha_ratio = self.alpha_ratio[...]
    for i in range(self.num_levels):
      alpha = self.step_size * alpha_ratio[i]
      for t in range(self.steps_per_level):
        g = self.critic_score(obs, a, i)
        score_norm = jnp.linalg.norm(g, axis=-1).mean()

        if self.normalize_score:
          g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
        drift = 0.5 * alpha * self.temperature * g

        noise = jax.random.normal(keys[i * self.steps_per_level + t], a.shape, a.dtype)
        a = a + drift + jnp.sqrt(alpha) * noise

        if self.clip_actions:
          a = jnp.clip(a, -self.action_limit, self.action_limit)

    return a, score_norm

  def train_step(self, batch):
    obs      = self.normalize_obs(batch["obs"])
    next_obs = self.normalize_obs(batch["next_obs"])
    action   = batch["act"]
    term     = batch["term"]
    reward   = batch["rew"] * self.reward_scale

    losses = {}
    keys = jax.random.split(self.rngs.noise(), 3)

    next_action, score_norm = self.sample(next_obs, keys[0])
    next_action = jax.lax.stop_gradient(next_action)

    idx   = jax.random.randint(keys[1], (obs.shape[0],), 0, self.clean_level)
    noisy = action + self.sigmas[...][idx][:, None] * jax.random.normal(
      keys[2], action.shape, action.dtype)


    pred_clean, pred_noisy = self.critic.raw_batches(
      (obs, self.augment_action(action, self.clean_level)),
      (obs, self.augment_action(noisy, idx))
    )

    bootstrap = self.critic.target_bootstrap(next_obs, next_action, self.clean_level)

    self.critic.loss_from_raw(pred_clean,
                              self.critic.bellman_target(reward, term, bootstrap), losses)

    td_loss = losses.pop("q")

    smoothing_target = jax.lax.stop_gradient(self.critic.reduce_expectation(pred_clean))

    self.critic.loss_from_raw(pred_noisy, smoothing_target, losses)
    smoothing_loss = losses.pop("q")

    losses["q"] = td_loss + smoothing_loss

    self.metrics.scalar("td_loss", td_loss)
    self.metrics.scalar("smoothing_loss", smoothing_loss)
    self.metrics.vec("action", next_action, limit=1.0)
    self.metrics.scalar("score_norm", score_norm)

    return losses

  def inference_step(self, observation, step_info=None, key=None):
    obs = self.normalize_obs(observation)
    key = self.rngs.noise() if key is None else key
    action, _ = self.sample(obs, key)
    return action, {}
