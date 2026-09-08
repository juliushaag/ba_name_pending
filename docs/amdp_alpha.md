# AMDP with per-action temperatures

`per_action` gives each position in the action chunk one temperature, repeated
across that position's action dimensions. `per_chunk` uses one scalar. Let
`A = diag(chunk_broadcast(alpha))`, treating alpha as fixed during actor updates.

The implemented Euler process is

    X[k+1] = X[k] + v[k] c(X[k], k) + sqrt(v[k]) eps[k].

Here `c` is `u_tilde` and `v[k]` is the integrated noise variance for the step.
The expectation of the entropy estimator's stochastic integral is zero. Thus
maximizing the critic value plus the alpha-weighted entropy surrogate is
minimizing

    J(c) = E[sum_k v[k]/2 * c[k]^T A c[k] + g(X[N])],
    g(x) = -Q(squash(x)) - sum_i alpha[i] * (prior_i(x) + log_jacobian_i(x)).

The entropy quantities are path-based lower bounds/surrogates, not exact
marginal action entropies. This change optimizes those existing quantities.

## Why an on-policy training pair is required

For this zero-reference-drift process, the lean adjoint is the terminal
cost gradient `grad(g(X[N]))`. The conditional matching equation is

    c(x, k) = -A^-1 E[grad(g(X[N])) | X[k] = x].

The endpoint and intermediate state must come from the **same controlled
trajectory**. The old code drew an endpoint and then independently sampled a
reference Brownian bridge to it. That substitutes a different conditional law.
Even with scalar alpha, the reference bridge identity requires the appropriate
terminally tilted reference path law, which a learned controller need not have.
Unequal action costs generally do not admit that scalar terminal-tilt argument.

`sample_training_pair` saves one uniformly selected pre-step state per batch row
while executing the same Euler transitions as `generate_action`. The endpoint,
selected state, and time are stopped. Only one further control evaluation is
differentiated for the regression, as before; no control Jacobian or backward
pass through the trajectory is required. Both alpha granularities use this path.

## Discrete stationary-condition derivation

The full discrete costate for a fixed trajectory obeys

    p[N] = grad(g(X[N])),
    p[k] = p[k+1] + v[k] Dc(X[k], k)^T (A c[k] + p[k+1]).

At an unrestricted optimal Markov control, first-order optimality gives

    A c(x, k) + E[p[k+1] | X[k] = x] = 0.

Consequently, the Jacobian term has zero conditional expectation. By backward
induction and the Markov property,
`E[p[k+1] | X[k]] = E[grad(g(X[N])) | X[k]]`. This yields the matching equation
above. Conversely, at a control satisfying that matching equation everywhere,
the same induction recovers the full first-order condition. This establishes
stationary-condition consistency; it does not claim global convergence for a
finite neural network or equality to the full pathwise gradient away from a
stationary point.

This extends the lean-adjoint construction in
[Adjoint Matching, Section 5.2](https://arxiv.org/html/2409.08861v3#S5.SS2)
to this diagonal control cost. `A^-1 grad(g)` is a **control target**, not a
claimed log-density score. It need not be conservative. A curl test alone
therefore does not invalidate vector alpha; the missing controlled path law was
the substantive problem in the original extension.

The code retains its positive time weighting and Euclidean regression metric:

    u = sigma(t) c(X[t], t),
    target = -sigma(t) A^-1 grad(g(X[N])),
    loss = mean_batch[||u - stop_gradient(target)||^2 / (2 sigma(t))].

These weights leave the unrestricted conditional regression optimum unchanged.
For a finite shared network they choose how regression errors are weighted.
The existing critic-gradient clipping is retained; when clipping activates it
is a stabilization heuristic, so the exact stationary-condition argument applies
to the unclipped target. The alpha optimizer still receives only its dual loss.

## Terminal entropy term and validation

For the matched erf squash, the prior gradient `x / terminal_var` cancels its
log-Jacobian gradient separately in every dimension, including with unequal
alphas. The implementation includes the weighted terminal entropy derivative
explicitly so a different configured squash, such as tanh, also has the right
target.

The regression tests include an independently solved finite-horizon linear
quadratic control problem with coupled coordinates and unequal alphas. A
symmetric cubature rule integrates its required first and second noise moments
exactly, testing that the actor gradient vanishes at the known optimum and is
nonzero after perturbing it. Further tests cover actual rollout/state pairing,
scalar versus repeated-vector alpha, terminal squash corrections, gradient
isolation, and policy updates with nonuniform temperatures.
