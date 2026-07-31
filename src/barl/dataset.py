import minari
import numpy as np
import torch

from barl.rollout import ReplayBuffer, Trajectory


@torch.no_grad()
def compute_gae(trajectories: list[Trajectory], critic, gamma: float = 0.99,
                 lam: float = 0.95, device: str = "cpu") -> None:
  """Fills each Trajectory's extras with GAE(lambda) advantages ("adv") and
  bootstrapped returns ("ret"), in place. Must run once per rollout, over the
  still-in-order trajectories, BEFORE SingleStepDataset shuffles them --
  GAE's backward recursion needs sequential rewards/values."""
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

    lengths = np.array([len(i) for i in self.buffer])

    self.length_cumsum = np.cumsum(lengths)

    self.discrete_actions = np.issubdtype(
        np.asarray(self.buffer[0].actions).dtype, np.integer
    )

    # Flattened once here rather than re-derived per item: __getitem__ is on the
    # DataLoader's hot path, and a searchsorted + per-field torch.asarray per
    # sample dominated it. Observations are stored once -- `obs_row` indexes a
    # transition's observation and that row + 1 is its next_obs.
    self.observations = np.concatenate(
        [t.observations for t in self.buffer]).astype(np.float32, copy=False)
    starts = np.concatenate([[0], np.cumsum(lengths + 1)[:-1]])
    self.obs_row = np.concatenate([s + np.arange(n) for s, n in zip(starts, lengths)])

    self.term = np.zeros(int(lengths.sum()), dtype=bool)
    self.term[self.length_cumsum - 1] = [t.terminated for t in self.buffer]

    actions = np.concatenate([t.actions for t in self.buffer])
    # discrete actions -> long (for embedding/cross-entropy), continuous -> float32
    self.act = actions if self.discrete_actions else actions.astype(np.float32, copy=False)
    self.rew = np.concatenate([t.rewards for t in self.buffer]).astype(np.float32, copy=False)
    self.extras = {
      k: np.concatenate([t.extras[k] for t in self.buffer]).astype(np.float32, copy=False)
      for k in set.intersection(*(set(t.extras) for t in self.buffer))
    }

  def __len__(self):
    return self.length_cumsum[-1]

  def __getitem__(self, idx) -> dict:

    assert idx < len(self)

    row = self.obs_row[idx]

    sample = dict(
      obs=      torch.from_numpy(self.observations[row]),
      next_obs= torch.from_numpy(self.observations[row + 1]),
      act=      torch.from_numpy(self.act[idx]),
      rew=      torch.from_numpy(self.rew[idx: idx + 1])[0],
      term=     bool(self.term[idx]),
    )

    for key, values in self.extras.items():
      sample[key] = torch.from_numpy(values[idx: idx + 1])[0]

    return sample

  @staticmethod
  def collate(batch: dict) -> dict:
    """Pass-through: __getitems__ already returns a stacked batch, so a
    DataLoader over this dataset must be given this as its `collate_fn`
    (torch still routes __getitems__' output through collate_fn, and the
    default one expects a list of samples)."""
    return batch

  def __getitems__(self, indices: list[int]) -> dict:
    """Batched fetch. torch's DataLoader prefers this over per-index
    __getitem__ when present, which -- together with collate() above -- makes
    the whole minibatch one fancy-index per field."""
    idx  = np.asarray(indices)
    rows = self.obs_row[idx]

    batch = dict(
      obs=      torch.from_numpy(self.observations[rows]),
      next_obs= torch.from_numpy(self.observations[rows + 1]),
      act=      torch.from_numpy(self.act[idx]),
      rew=      torch.from_numpy(self.rew[idx]),
      term=     torch.from_numpy(self.term[idx]),
    )
    for key, values in self.extras.items():
      batch[key] = torch.from_numpy(values[idx])

    return batch



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
    """Rescale rewards in-place so episode-return spread equals `target_range`
    (IQL-paper style). Returns the scale factor."""
    ep_returns = [float(np.sum(traj.rewards)) for traj in self.buffer]
    ret_max, ret_min = max(ep_returns), min(ep_returns)
    spread = ret_max - ret_min
    if spread < 1e-8:
      self.reward_scale = 1.0
      return self.reward_scale

    self.reward_scale = target_range / spread
    for traj in self.buffer:
      traj.rewards = (np.asarray(traj.rewards, dtype=np.float32) * self.reward_scale)

    return self.reward_scale