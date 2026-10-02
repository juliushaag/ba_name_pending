# FloQ critic

Select `model.components.critic.class=FloqCritic` to use a flow-matching
critic with SAC or another compatible actor. This is an adaptation of the
[authors' implementation](https://github.com/CMU-AIRe/floq/blob/main/agents/floq.py)
to the shared actor, Bellman-target, ensemble, and optimizer interfaces.

The velocity field takes observation, action, an HL-Gauss encoding of the
interpolated return, and encoded time. Training samples a source return `z0`
and time `t`, then regresses the field at `(1-t)*z0 + t*y` onto `y-z0`, where
`y` is a stopped-gradient Bellman target. Euler integration of a target
field supplies the bootstrap. The default averages eight noise samples and
uses eight integration steps.

With `distill=true`, a separate scalar-Q ensemble learns the live flow's
mean endpoint and supplies the actor's values and action gradients. Its
distillation target is stopped-gradient. With `distill=false`, the actor
differentiates through the integrated live flow, using deterministic midpoint
quadrature for its source returns.

## Source-noise scale

The initial interval is `noise_coverage * [v_min, v_max]`, with coverage 0.1.
`noise_limit=1.0` caps its largest absolute endpoint in scaled return units,
rescaling both endpoints by the same factor. Narrow intervals stay unchanged.
The histogram's support and Bellman targets are unaffected.

For HalfCheetah's episode-return bounds `[-3200,3200]`, unrestricted source
noise spans `[-320,320]`. At initialization its eight-sample mean has standard
deviation about 65, which overwhelms reward-scale bootstrap and distillation
signals. The bounded default instead uses `[-1,1]`. Set `noise_limit=null` to
restore unrestricted noise, or increase the limit deliberately for a task
with larger rewards. `noise_coverage` still sets the interval before this cap.

## Learning diagnostic

Run a local SAC comparison with:

```bash
.venv/bin/python scripts/train.py --config experiments/sac_floq_smoke.yaml \
  --run-dir /tmp/floq-bounded
```

The config uses one seed, reduced network widths, 10,000 gradient updates,
and five evaluations of three episodes each. For an unrestricted comparison,
change its `noise_limit` to null and use a separate run directory. This is a
short learning diagnostic, not a multi-seed benchmark.

`tests/test_floq.py` checks Euler integration and flow/distillation equations,
target copying, gradients, sampling configuration, source-noise bounds, and
actual learning of a known action-dependent terminal reward in both modes.
