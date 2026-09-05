"""Diagonal-Gaussian log-densities, in the three shapes this codebase reads them
in. Hand-rolled rather than pulled from distrax/TFP: distrax's Transformed+Tanh
has a long-standing NaN log_prob issue that lands exactly on SAC, and neither
library is otherwise needed.

`log_prob` is elementwise because SAC's squash correction is applied per
dimension before summing; `diag_log_prob` sums and keeps the axis, which is the
(B, 1) shape every path weight here accumulates in."""

import math

import jax.numpy as jnp

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def log_prob(x, mean, std):
  """Elementwise log N(x; mean, std)."""
  return -0.5 * ((x - mean) / std) ** 2 - jnp.log(std) - LOG_SQRT_2PI


def diag_log_prob(x, mean, std):
  """log N(x; mean, diag(std^2)) as (B, 1) -- a transition kernel's log-density."""
  return log_prob(x, mean, std).sum(axis=-1, keepdims=True)


def surprisal(x, var: float, dim: int):
  """-log N(x; 0, var I) as (B, 1), for an isotropic `var` known at construction.

  Not `-diag_log_prob(x, 0.0, sqrt(var))`: the normalizer is one Python float
  folded at trace time rather than `dim` identical array constants summed, which
  is what lets a caller hold `var` as a plain float. That matters where the value
  is read inside a vmapped step, since a Buffer read there is a tracer."""
  return 0.5 * dim * math.log(2 * math.pi * var) \
       + 0.5 * (x ** 2).sum(axis=-1, keepdims=True) / var
