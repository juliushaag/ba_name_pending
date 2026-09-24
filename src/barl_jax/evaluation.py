"""Episode evaluation independent of policy construction and training."""
import gymnasium as gym
import jax
import numpy as np

from barl_jax.action_chunking import ActionChunkExecutor, step_env
from barl_jax.components import action_bounds_numpy
from barl_jax.seeding import derive_seed


def episode_success(info, i: int, is_vec: bool, terminated: bool) -> bool:
  """Did the episode that just ended on slot `i` succeed?

  Where the env says so itself, that answer wins: ManiSkill reports
  `info["success"]` and sets `terminated` *from* it, so the survival convention
  below would read its success rate exactly backwards. Every other family here
  reports no such key, and for them an episode that never terminated is one the
  agent did not fail out of -- a MuJoCo humanoid terminates by falling over.

  Gymnasium's vector envs batch each sub-env info key into an array plus an
  `_<key>` presence mask (a key only some sub-envs reported); a plain env just
  carries the key."""
  value = info.get("success")
  if value is None:
    return not terminated
  if not is_vec:
    return bool(np.asarray(value).reshape(-1)[0])
  present = info.get("_success")
  if present is not None and not bool(np.asarray(present)[i]):
    return not terminated
  return bool(np.asarray(value)[i])


def run_episodes(actor, env: gym.Env, n: int, render: bool, *, seed: int = 0,
                 **inference_kwargs) -> list[dict]:
  """`n` complete episodes, in parallel waves of `num_envs` if `env` is a
  VectorEnv. A slot that finishes before the rest of its wave is reset and
  stops recording -- autoreset_mode=DISABLED asserts rather than letting a
  finished slot be stepped again -- so a wave costs its longest episode.

  `actor` is anything exposing `inference_fn()`, usually a Policy or a
  StackedPolicy -- the latter is how the trainer evaluates every seed in one
  vmapped pass instead of one loop per seed.

  `seed` owns both RNG streams this function uses -- the policy's own action
  key and each wave's `env.reset(seed=...)` -- independent of whatever
  training RNG state exists elsewhere (the caller's own; this function never
  reads it). The caller derives `seed` from something stable (run seed +
  evaluation identity, see `barl_jax.seeding.derive_seed`) so repeated calls
  with the same `seed` see the same episodes; each wave within one call still
  gets its own derived sub-seed, so wave 2 does not replay wave 1."""
  if n <= 0:
    return []

  is_vec = isinstance(env, gym.vector.VectorEnv)
  n_envs = env.num_envs if is_vec else 1

  def reset(wave: int):
    obs, _ = env.reset(seed=derive_seed(seed, "wave", wave))
    return obs if is_vec else obs[None, ...]

  act = actor.inference_fn()
  key = jax.random.PRNGKey(seed)
  chunk_size = getattr(actor, "chunk_size", 1)
  action_space = env.single_action_space if is_vec else env.action_space
  act_scale_np, act_bias_np = action_bounds_numpy(action_space)
  act_dim = getattr(actor, "action_dim",
                    int(action_space.n) if isinstance(action_space, gym.spaces.Discrete)
                    else int(np.prod(action_space.shape)))

  out = []
  wave_index = 0
  while len(out) < n:
    wave = min(n_envs, n - len(out))
    obs  = reset(wave_index)
    refill_index = 0
    chunks = ActionChunkExecutor(act, chunk_size=chunk_size, act_dim=act_dim, n_envs=n_envs)
    data = [dict(frames=[], rewards=[], terminated=False, success=False,
                 act_nonfinite=0, n_act=0, chunk_size=chunk_size,
                 chunk_refills=0, chunk_waste_frac=0.0)
            for _ in range(n_envs)]
    live = np.zeros(n_envs, dtype=bool)
    live[:wave] = True

    while live.any():
      key, subkey = jax.random.split(key)
      action_np, _ = chunks.current(obs, subkey, live=live, full_batch=hasattr(actor, "num_seeds"))
      nonfinite = ~np.isfinite(action_np)
      obs, reward, term, trunc, info = step_env(env, action_np, act_scale_np, act_bias_np,
                                                is_vec=is_vec)

      frames = (env.call("render") if is_vec else [env.render()]) if render else None

      done = term | trunc
      chunks.advance(done)
      slot_refills = chunks.slot_refills()
      slot_wasted = chunks.slot_wasted()
      for i in np.flatnonzero(live):
        data[i]["rewards"].append(reward[i])
        data[i]["act_nonfinite"] += int(nonfinite[i].sum())
        data[i]["n_act"] += int(nonfinite[i].size)
        if render:
          data[i]["frames"].append(frames[i])
        if done[i]:
          data[i]["terminated"] = bool(term[i])
          data[i]["success"] = episode_success(info, i, is_vec, bool(term[i]))
          data[i]["chunk_refills"] = int(slot_refills[i])
          executed = max(1, data[i]["chunk_refills"] * chunk_size)
          data[i]["chunk_waste_frac"] = float(slot_wasted[i]) / executed
          live[i] = False

      if live.any() and done.any():
        # Unreachable for a single (non-vector) env: `live[0]` already went
        # False above the one time `done[0]` can fire, so this only ever
        # actually executes for a VectorEnv's `reset_mask` partial reset.
        reset_seed = derive_seed(seed, "wave", wave_index, "refill", refill_index)
        if is_vec:
          obs, _ = env.reset(seed=reset_seed, options={"reset_mask": done})
        else:
          obs, _ = env.reset(seed=reset_seed)
          obs = obs[None, ...]
        refill_index += 1

    wave_index += 1
    out += data[:wave]

  return out


