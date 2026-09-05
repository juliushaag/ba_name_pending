"""Learning objectives over critic prediction and regression primitives."""
import jax


def td_loss(critic, obs, action, next_obs, next_action, reward, term,
            losses: dict, shift=0.0, horizon=None) -> None:
  if critic.has_target:
    pred_raw = critic.raw(obs, action)
    next_raw = critic.target_raw(next_obs, next_action)
  else:
    pred_raw, next_raw = critic.raw_batches((obs, action), (next_obs, next_action))
  target = critic.bellman_target(reward, term, critic.bootstrap_from_raw(next_raw), shift,
                               horizon=horizon)
  critic.loss_from_raw(pred_raw, target, losses)

def noise_td_loss(critic, obs, action, next_obs, next_action, reward, term,
                           sigmas, key, losses: dict, shift=0.0, horizon=None) -> None:
  clean_level = sigmas.shape[0] - 1

  level_key, noise_key = jax.random.split(key)
  levels = jax.random.randint(level_key, (action.shape[0],), 0, clean_level)
  noisy  = action + sigmas[levels][:, None] * jax.random.normal(
    noise_key, action.shape, action.dtype)

  pred_clean, pred_noisy = critic.raw_batches(
    (obs, critic.augment(action, clean_level)),
    (obs, critic.augment(noisy, levels)),
  )

  next_raw  = critic.target_raw(next_obs, critic.augment(next_action, clean_level, target=True))
  td_target = critic.bellman_target(reward, term, critic.bootstrap_from_raw(next_raw), shift,
                                  horizon=horizon)

  td_loss = critic.regression_loss(pred_clean, td_target)
  smoothing_loss = critic.regression_loss(pred_noisy, critic.target_from_raw(pred_clean))

  losses["critic"] = td_loss + critic.smoothing_coef * smoothing_loss

  critic.record(pred_clean, td_target)
  critic.metrics.scalar("td_loss", td_loss)
  critic.metrics.scalar("smoothing_loss", smoothing_loss)
