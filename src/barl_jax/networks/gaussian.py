"""Diagonal-Gaussian log-densities. Hand-rolled rather than pulled from
distrax/TFP: distrax's Transformed+Tanh has a long-standing NaN log_prob issue
that lands exactly on SAC, and neither library is otherwise needed.

Elementwise, not reduced: every caller sums it themselves, within whatever
axis their own shape convention wants -- SAC's squash correction sums the whole
plan_dim before negating into an entropy, DIME/AMDP's path weights sum within
each chunk position's action_dim block (`Component.chunk_sum`) so their
per-chunk-position alpha reads the right per-position quantity."""

import math

import jax.numpy as jnp

LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def log_prob(x, mean, std):
  """Elementwise log N(x; mean, std)."""
  return -0.5 * ((x - mean) / std) ** 2 - jnp.log(std) - LOG_SQRT_2PI
