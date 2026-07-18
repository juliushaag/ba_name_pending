import minari
import numpy as np
import torch

from barl.rollout import ReplayBuffer, Trajectory


@torch.no_grad()
def compute_gae(trajectories: list[Trajectory], critic, gamma: float = 0.99,
                 lam: float = 0.95, device: str = "cpu") -> None:
  """
  Fills each Trajectory's extras with GAE(lambda) advantages ("adv") and
  bootstrapped returns ("ret"), in place.

  Uses the per-step values PPOPolicy.inference_step already stashed in
  extras["value"] at rollout time, plus one extra critic.value() call to
  bootstrap the trailing observation each chunk stores but never ran
  inference on. Must run once per rollout, over the still-in-order
  trajectories, BEFORE SingleStepDataset flattens/shuffles them into
  single-step batches -- GAE's backward recursion needs the sequential
  rewards/values a shuffled dataset no longer has.
  """
  for traj in trajectories:
    values  = np.asarray(traj.extras["value"], dtype=np.float32)
    rewards = np.asarray(traj.rewards,         dtype=np.float32)
    T = len(rewards)

    last_obs  = torch.as_tensor(traj.observations[-1], dtype=torch.float32, device=device).unsqueeze(0)
    bootstrap = 0.0 if traj.terminated else critic.value(last_obs).item()

    next_values = np.append(values[1:], bootstrap)

    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
      nonterminal = 0.0 if (t == T - 1 and traj.terminated) else 1.0
      delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
      gae   = delta + gamma * lam * nonterminal * gae
      adv[t] = gae

    traj.extras["adv"] = adv
    traj.extras["ret"] = adv + values


def convert_dtype(x : torch.Tensor):
    if x.dtype == np.float64: return torch.float32
    if x.dtype == np.float32: return torch.float32
    if x.dtype == np.int64: return torch.long
    if x.dtype == np.bool_: return torch.bool

    return x.dtype


class SingleStepDataset(torch.utils.data.Dataset):

  def __init__(self, trajectories : list[Trajectory]):
    self.buffer = trajectories

    lengths = [len(i) for i in self.buffer]

    self.length_cumsum = np.cumsum(lengths)
  
    self.discrete_actions = np.issubdtype(
        np.asarray(self.buffer[0].actions).dtype, np.integer
    )

  def __len__(self):
    return self.length_cumsum[-1]
  
  def __getitem__(self, idx) -> dict: 

    assert idx < len(self)

    base_idx = np.searchsorted(self.length_cumsum, idx, side="right")

    prev = 0 if base_idx == 0 else self.length_cumsum[base_idx - 1]
    traj_idx = idx - prev
    
    traj_end = traj_idx == len(self.buffer[base_idx]) - 1

    act = torch.asarray(self.buffer[base_idx].actions[traj_idx])
    # discrete actions -> long (for embedding/cross-entropy), continuous -> float32
    act = act.long() if self.discrete_actions else act.float()

    sample = dict(
      obs=      torch.asarray(self.buffer[base_idx].observations[traj_idx]).float(),
      next_obs= torch.asarray(self.buffer[base_idx].observations[traj_idx + 1]).float(),
      act=      act,
      rew=      torch.asarray(self.buffer[base_idx].rewards[traj_idx]).float(),
      term=     traj_end and self.buffer[base_idx].terminated
    )

    for key, values in self.buffer[base_idx].extras.items():
      sample[key] = torch.asarray(values[traj_idx]).float()

    return sample
  

  @classmethod
  def from_replay(cls, replay : "ReplayBuffer") -> "SingleStepDataset":
    return cls(replay.buffer)
  
  def normalize_observations(self, eps: float = 1e-6) -> dict:
    all_obs = np.concatenate([np.asarray(t.observations) for t in self.buffer], axis=0)

    self.obs_mean = all_obs.mean(axis=0, dtype=np.float64).astype(np.float32)
    self.obs_std  = all_obs.std(axis=0, dtype=np.float64).astype(np.float32)
    self.obs_std  = np.maximum(self.obs_std, eps)   # guard constant dims

    return self.normalization_stats()

  def normalization_stats(self) -> dict:
    """All scaling params needed to reproduce preprocessing at inference."""
    return {
      "obs_mean":     getattr(self, "obs_mean", None),
      "obs_std":      getattr(self, "obs_std", None),
      "reward_scale": getattr(self, "reward_scale", 1.0),
    }

  def save_normalization(self, path: str):
    np.savez(path, **{k: v for k, v in self.normalization_stats().items() if v is not None})

  @staticmethod
  def load_normalization(path: str) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}
  
  def normalize_rewards(self, target_range: float = 1000.0) -> float:
    """Rescale all rewards in-place so that the spread of episode returns
    equals `target_range` (IQL-paper style: reward *= 1000 / (max_ret - min_ret)).
    Returns the scale factor (store it if you ever need raw-scale values back).
    """
    ep_returns = [float(np.sum(traj.rewards)) for traj in self.buffer]
    ret_max, ret_min = max(ep_returns), min(ep_returns)
    spread = ret_max - ret_min
    if spread < 1e-8:
      # all episodes have identical return; nothing meaningful to normalize
      self.reward_scale = 1.0
      return self.reward_scale

    self.reward_scale = target_range / spread
    for traj in self.buffer:
      traj.rewards = (np.asarray(traj.rewards, dtype=np.float32) * self.reward_scale)

    return self.reward_scale

  @classmethod
  def from_minari(cls, ds: minari.MinariDataset) -> "SingleStepDataset":
    
    trajectories = []
    for ep in ds.iterate_episodes():
        obs = ep.observations
        if isinstance(obs, dict):
            raise ValueError(
                f"Dict observation space not supported (keys: {list(obs.keys())}). "
                "Flatten/concatenate the keys you need first."
            )
        trajectories.append(
            Trajectory(
                id=ep.id,
                observations=np.asarray(obs, dtype=np.float32),       # (T+1, obs_dim)
                actions=np.asarray(ep.actions),                       # (T, ...)
                rewards=np.asarray(ep.rewards, dtype=np.float32),     # (T,)
                terminated=bool(ep.terminations[-1]),
                truncated=bool(ep.truncations[-1]),
            )
        )
    return cls(trajectories)
    
  @classmethod
  def from_ogbench(
      cls,
      dataset
  ) -> "SingleStepDataset":
      """
      Load an OGBench dataset and convert it into a list of Trajectory objects.

      Example:
          ds = SingleStepDataset.from_ogbench(
              "pointmaze-medium-v0"
          )
      """

      if "next_observations" in dataset:
        raise ValueError("Use compact_dataset=True for the ogbench dataset creation")

      obs = np.asarray(dataset["observations"], dtype=np.float32)
      actions = np.asarray(dataset["actions"])
      rewards = np.asarray(dataset["rewards"], dtype=np.float32)

      terminals = np.asarray(
          dataset.get("terminals", dataset.get("terminateds")),
          dtype=bool,
      )
      # compact OGBench layout: each episode ends with two consecutive terminal
      # entries — the last valid (obs, action) at index i (valids[i]=1), followed
      # by the terminal next_obs at index i+1 (valids[i+1]=0, action is garbage).
      # We split only on valid terminals and include the i+1 obs as next_obs.
      valids = np.asarray(dataset["valids"], dtype=bool)

      trajectories = []

      start = 0
      ep_id = 0

      for i in range(len(terminals)):
        if terminals[i] and valids[i]:
          trajectories.append(
            Trajectory(
              id=ep_id,
              observations=obs[start:i + 2],   # T+1 obs, includes terminal next_obs
              actions=actions[start:i + 1],    # T actions
              rewards=rewards[start:i + 1],    # T rewards
              terminated=True,
              truncated=False,
            )
          )
          start = i + 2   # skip the valid=0 terminal-next_obs entry
          ep_id += 1

      assert start == len(terminals), "compact OGBench dataset should end with terminals[-1] == 1"

      return cls(trajectories)