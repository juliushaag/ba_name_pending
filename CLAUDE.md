# Repository guidance

`barl_jax` is a JAX/Flax-NNX off-policy RL research framework. The current
architecture, commands, extension interfaces and checkpoint semantics are in
[README.md](README.md). Follow-up proposals are in [docs/QOL.md](docs/QOL.md).

Use `uv` for dependencies and `scripts/submit.py` to launch experiments. Run
`uv run pytest tests/ -q` and `uv run ruff check .`
for verification. Tests use the CPU; the training CLI requires an accelerator
unless `BARL_ALLOW_CPU=1` is explicitly set.

Keep package and config imports lightweight. Load simulator dependencies inside
environment factories. Pass a `ModelSpec`/`ConfigTree` explicitly in new runtime
code and keep configuration instance-local. Components
import contracts from `components.py`, not from the policy implementation.

Algorithm defaults live in each actor's `POLICY` dictionary.
MLP settings are grouped under one `ConfigDict` declaration (`mlp = ConfigDict(
activation=..., norm=..., dtype=..., ...)`, read back as `self.mlp.activation`);
capacity tiers live in `networks/mlp.py`. Every class-attribute default --
algorithm knob or structural setting alike -- is a `Config` declaration
(`ConfigDict` included); there is no separate searchable kind, and a
`ConfigDict` field still overrides at its own flat YAML key, same as a
standalone `Config`. A more-derived declaration overrides its ancestor.

Comments should explain non-obvious invariants or numerical/shape constraints.
Prefer clear identifiers and short comments over historical implementation essays.

## Algorithmic clarity

Maximize abstraction reuse and algorithmic clarity together. Shared mathematical
operations should have one implementation and explicit, readable composition.
Read [docs/ALGORITHMIC_CLARITY.md](docs/ALGORITHMIC_CLARITY.md) before changing
learning objectives or their data contracts. It defines the required overhaul:
explicit chunk n-step targets, meaningful replay fields, visible gradient and
update semantics, and equation-based acceptance tests. Evaluate abstraction and
clarity jointly with the QOL simplification goals; redesign unclear boundaries
before accepting either duplicated objectives or hidden mathematical choices.

## Active implementation handoff

Read [docs/CLAUDE_IMPLEMENTATION_PLAN.md](docs/CLAUDE_IMPLEMENTATION_PLAN.md)
before continuing the QOL backlog. It defines the repo's direction, ownership map,
numerical and lifecycle invariants, implementation phases, and acceptance checks.
The QOL draft was reverted; earlier uncommitted refactors must be preserved.

This is a development-stage repository: update callers together and remove obsolete
APIs instead of adding compatibility shims. Prefer fewer concepts and shared behavior
over more wrappers. The requested outcome is a net reduction in production code;
measure it separately from tests/docs and never trade correctness for line counts.
Keep algorithm changes explicit, especially normalization, gradient isolation,
action chunking, evaluation RNG ownership, and checkpoint continuation semantics.

The handoff also specifies policy-independent action executors (Default, Best-of-N,
Mean-of-N). Keep action selection separate from job executors and algorithm losses.
