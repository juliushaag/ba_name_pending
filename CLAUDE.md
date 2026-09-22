# Repository guidance

`barl_jax` is a JAX/Flax-NNX off-policy RL research framework. The current
architecture, commands, extension interfaces and checkpoint semantics are in
[README.md](README.md). 

Use `uv` for dependencies and `scripts/submit.py` to launch experiments. Run
`uv run pytest tests/ -q` and `uv run ruff check .`
for verification. Tests use the CPU; the training CLI requires an accelerator
unless `BARL_ALLOW_CPU=1` is explicitly set.

Keep package and config imports lightweight. Load simulator dependencies inside
environment factories. Pass a `ModelSpec`/`RunConfig` explicitly in new runtime
code and keep configuration instance-local. Components
import contracts from `components.py`, not from the policy implementation.

Algorithm defaults live in typed frozen dataclasses beside their
implementations (`*Config` with `config_type`, e.g. `StochasticActorConfig`)
-- the single source of truth for that component; `config/schemas.py` stays
generic scaffolding (run/environment metadata, the model/components
composition shape, the registries) and never redeclares them. MLP settings
are grouped under one nested `mlp` dataclass, read back as
`self.config.mlp.activation`. There is no capacity-tier selector; a
component's width/depth are set directly (`latent_dim`, `layers`). A
more-derived dataclass field overrides its ancestor.

A component keeps its whole materialized config as `self.config` (identity-hashed
via `_Static`, see `components.py`) rather than copying each field onto its own
flat attribute -- read `self.config.foo` from any method, and mutate it in a
test with `c.config = dataclasses.replace(c.config, foo=...)`. `lr`/`adam_b1`/
`grad_clip`/`grad_clip_mode` follow the same rule, including in the shared
`ActorOptim`/`QCriticBase`/`AutotuneRegulator` optimizer-role mixins.

Comments should explain non-obvious invariants or numerical/shape constraints.
Prefer clear identifiers and short comments over historical implementation essays.
But reduce the overall commenting to a minimum !

## Algorithmic clarity

Maximize abstraction reuse and algorithmic clarity together. Shared mathematical
operations should have one implementation and explicit, readable composition.

## Active implementation handoff

The typed-config restructure is complete (`RunConfig.from_dict`/`defaults`,
`ModelSpec`, `config.serialization`). Keep algorithm changes explicit,
especially normalization, gradient isolation, action chunking, evaluation RNG
ownership, and checkpoint continuation semantics.

The handoff also specifies policy-independent action executors (Default, Best-of-N,
Mean-of-N). Keep action selection separate from job executors and algorithm losses.
