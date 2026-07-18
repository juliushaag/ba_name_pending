import math

import torch


def sinusoidal_emb(t: torch.Tensor, time_dim: int) -> torch.Tensor:
  shape = t.shape[:-1]  # all dims except the trailing 1
  t = t.reshape(-1)
  half = time_dim // 2
  freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
  args = t[:, None] * 1000.0 * freqs[None]
  time_emb = torch.cat([args.sin(), args.cos()], dim=-1)
  return time_emb.reshape(*shape, time_dim)


@torch.no_grad()
def midpoint_flow_sample(step_fn, obs: torch.Tensor, action_dim: int, low: torch.Tensor,
                          high: torch.Tensor, steps: int, z: torch.Tensor = None) -> torch.Tensor:
  """Deterministic-ish midpoint ODE integration of a learned vector field `step_fn(obs, x, t) -> v`,
  from x0 ~ N(0, I) to x1, clamped to [low, high]. Shared by every plain flow-matching actor."""
  B  = obs.shape[0]
  dt = 1.0 / steps

  with torch.amp.autocast("cuda", torch.bfloat16):
    z = z if z is not None else torch.randn((B, action_dim), device=obs.device)
    for i in range(steps):
      t     = torch.full((B, 1), i * dt, device=obs.device)
      v     = torch.nan_to_num(step_fn(obs, z, t), 0.0, 0.0, 0.0)
      z_mid = z + v * (dt / 2)
      v_mid = torch.nan_to_num(step_fn(obs, z_mid, t + dt / 2), 0.0, 0.0, 0.0)
      z     = z + v_mid * dt

  return z.clamp(low, high)
