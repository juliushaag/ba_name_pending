# NCLQL sampler

NCLQL samples actions using a noise-conditioned critic and annealed Langevin
updates. The nominal step at level `i` is
`alpha_i = step_lr * (sigma_i / sigma_last)^2`. The update adds
`0.5 * alpha_i * temperature * score` and Gaussian noise with standard
deviation `sqrt(alpha_i)`, then clips the action to the configured bounds.

The default `max_step_size=0.001` caps `alpha_i` before computing both drift
and noise. With the default sigma ladder (`0.1` to `0.001`), `step_lr=0.0001`,
normalized scores and `temperature=500`, this limits drift norm to `0.25`
and noise standard deviation to about `0.032`. The finest step remains
`0.0001`. This avoids the unrestricted coarse step of `1`, whose drift norm
of `250` sends almost every action coordinate outside the `[-1,1]` box.

The cap is a numerical adaptation of the reference annealing schedule,
not a benchmark-validated improvement to returns. Lower caps also reduce
movement and exploration for a fixed number of sampling steps. Set
`model.components.actor.max_step_size=null` to restore unrestricted steps,
or tune the cap alongside `step_lr`, `temperature` and `steps_per_level`.

Debug metrics `actor/chain/step_size_l<i>`, `drift_norm_l<i>` and
`noise_scale_l<i>` report the effective scales. `clipped_frac_l<i>` reports
the fraction of coordinates outside the bounds before clipping in that
level's last substep. `actor/action/frac_at_limit` measures the final sampled
actions during training; these metrics do not measure optimizer clipping.
