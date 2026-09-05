# Repository guidance

`barl_jax` is a JAX/Flax-NNX off-policy RL research framework. The current
architecture, commands, extension interfaces and checkpoint semantics are in
[README.md](README.md). Follow-up proposals are in [docs/QOL.md](docs/QOL.md).

Use `uv` for dependencies and `scripts/submit.py` to launch experiments. Run
`uv run pytest tests/ -q` and `JAX_PLATFORMS=cpu uv run python scripts/lint_config.py`
for verification. Tests use the CPU; the training CLI requires an accelerator
unless `BARL_ALLOW_CPU=1` is explicitly set.

Keep package and config imports lightweight. Load simulator dependencies inside
environment factories. Pass a `ModelSpec`/`ConfigTree` explicitly in new runtime
code and keep configuration instance-local. Components
import contracts from `components.py`, not from the policy implementation.

Defaults have one source in `src/barl_jax/defaults`. Algorithm knobs are `HParam` declarations and structural defaults are
`Config` declarations. A more-derived declaration overrides its ancestor even
when its declaration kind changes.

Comments should explain non-obvious invariants or numerical/shape constraints.
Prefer clear identifiers and short comments over historical implementation essays.
