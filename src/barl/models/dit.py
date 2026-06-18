from torch import nn
import torch 
import math

import gymnasium as gym

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding, same as DDPM/DiT"""
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.dim = dim

    def forward(self, t):  # t: [B] or [B, H]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        x = t[..., None] * freqs
        emb = torch.cat([x.sin(), x.cos()], dim=-1)  # [..., dim]
        return self.mlp(emb)


class AdaLNBlock(nn.Module):
    """DiT-style block: self-attn + adaLN conditioning on t + obs"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        # adaLN modulation: predicts scale+shift from conditioning signal
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)  # scale/shift for norm1 and norm2, + gate
        )
    def forward(self, x, cond, mask):
        # x:    [B, H, dim]
        # cond: [B, H, dim]  ← now per-token, not global
        shift1, scale1, gate1, shift2, scale2, gate2 = \
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        # no more unsqueeze(1) needed — cond already has H dim

        h = self.norm1(x)
        h = h * (1 + scale1) + shift1          # [B, H, dim]
        T = x.shape[1]
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + gate1 * h

        h = self.norm2(x)
        h = h * (1 + scale2) + shift2
        h = self.ff(h)
        x = x + gate2 * h
        return x
    
  
class DiTModel(nn.Module):
    
  def __init__(self, obs_dim : int, act_dim : int,  context_length : int, max_steps : int = 32, num_heads : int = 8, chunk_size : int = 16, depth : int = 3, latent_dim : int = 128):
    super().__init__()

    self.action_dim = act_dim
    self.obs_dim = obs_dim

    self.action_embed = nn.Linear(self.action_dim, latent_dim)
    self.obs_embed = nn.Linear(self.obs_dim, latent_dim)
    self.time_embed = TimestepEmbedding(latent_dim)
    
    self.context_len = context_length
    self.max_steps = max_steps

    self.blocks = nn.ModuleList([
        AdaLNBlock(latent_dim, num_heads) for _ in range(depth)
    ])
    self.norm_out = nn.LayerNorm(latent_dim)
    self.proj_out = nn.Linear(latent_dim, self.action_dim)
    self.register_buffer(
        "causal_mask",
        torch.triu(torch.ones(self.context_len, self.context_len) * float("-inf"), diagonal=1)
    )

    self.chunk_size = chunk_size
    # obs gets its own positional embedding
    self.obs_pos_embed = nn.Parameter(torch.randn(1, self.context_len, latent_dim) * 0.02)
    # action chunk positional embedding (fixed size)
    self.act_pos_embed = nn.Parameter(torch.randn(1, chunk_size, latent_dim) * 0.02)
    # obs encoder: causal self-attention
    self.obs_blocks = nn.ModuleList([AdaLNBlock(latent_dim, num_heads) for _ in range(depth // 2)])
    # action denoiser: full attention + cross-attention into obs
    self.act_blocks = nn.ModuleList([AdaLNBlock(latent_dim, num_heads) for _ in range(depth // 2)])