# Quality-of-life backlog

These are follow-up proposals, not features implemented by the modularity refactor.
Priority reflects how often an issue can waste an experiment or developer session.

| Priority | Improvement | Benefit | Concrete completion criterion |
| --- | --- | --- | --- |
| P1 | CI with CPU tests, config lint and a wheel-install smoke test | Catches import, packaging and configuration regressions before experiments start | Every PR runs all three checks; optional simulator tests are reported separately |
| P1 | `doctor` command | Replaces trial-and-error debugging of CUDA, EGL, simulator and submodule setup | Reports versions, accelerator visibility, selected backend, optional dependencies and a minimal render result |
| P1 | One documented fast-check command | Makes local verification discoverable | A task runner runs focused tests and config lint with reproducible CPU settings |
| P1 | Run-config validation before allocating resources | Prevents expensive jobs failing on invalid numeric combinations | Rejects zero/negative cadences, incompatible chunk/batch/buffer sizes and rollout/env-count mismatches with named-key errors |
| P1 | Offline/no-op logger backend | Enables quick local experiments without a W&B account or service | Training accepts a logger factory; terminal/JSON logging works without W&B installed |
| P1 | Sweep status and retry CLI | Makes failed experiments recoverable without editing manifests | Shows pending/running/failed/completed runs across executors and resubmits selected failures from frozen configs |
| P1 | Checkpoint retention and configurable resume cadence | Bounds the cost of full replay persistence | Users can set checkpoint frequency, retained generations and a disk-space budget; deletion occurs only after a new checkpoint succeeds |
| P2 | Config diff and checkpoint inspection | Makes experiment comparisons and resume diagnosis faster | Prints effective value changes with provenance, model implementation, format version and saved counters without constructing environments |
| P2 | Scaffold an algorithm or environment | Reduces copy/paste errors when adding experiments | Generates a minimal implementation, model YAML and contract test using current public APIs |
| P2 | Small CPU end-to-end example | Gives contributors a known-good first run | A documented example completes collection, training, save and resume in under a minute without network logging |
| P2 | Deterministic evaluation seeds | Makes score comparisons less sensitive to unrelated evaluations | Evaluation owns explicit per-seed RNGs and records the episode seeds used |
| P2 | Transition schema validation | Makes custom replay/collector integration safer | Validates batch lengths, shapes, dtypes and reserved extras before mutating storage |
| P2 | Ruff and gradual type checking | Finds stale imports and interface mistakes early | Establishes a checked baseline for config/contracts first, without requiring unrelated formatting changes |
| P2 | Strict machine-readable result files | Improves interoperability with dashboards and HPO tools | Atomic JSON writes use `null` plus explicit status for absent/non-finite objectives; readers handle partially completed runs |
| P2 | Lightweight performance benchmarks | Detects regressions hidden by small unit tests | Records import cost, worker RSS, rollout throughput and gradient-step latency with hardware metadata |
| P3 | Environment-state adapters for exact resume | Enables trajectory-identical continuation where simulators permit it | An optional adapter saves/restores simulator state, episode RNG and pending chunks, with a continuation equivalence test |
| P3 | Plugin entry-point discovery | Removes manual imports for installed external extensions | Installed packages advertise implementations with collision checks and actionable dependency errors |
| P3 | Sequence and prioritized replay implementations | Supports recurrent agents and alternative sampling research | New buffers implement the existing collector sink/sampling boundary and include sampling-contract tests |

Foundations already implemented: lightweight imports, explicit model specifications,
component/objective validation, separate collection and storage, independent replay
RNGs, versioned checkpoints, training-state resume, resource ownership, and packaged
configuration defaults. See the [README](../README.md) for their APIs and limitations.

Implementation order, architecture constraints, failure cases, and detailed
acceptance checks are in the [Claude Code implementation handoff](CLAUDE_IMPLEMENTATION_PLAN.md).
The backlog remains proposed work; the interrupted QOL draft was reverted.

Additional requested feature: policy-independent action executors (Default, Best-of-N,
Mean-of-N), with shared collection/evaluation integration. See the detailed
[design and acceptance tests](CLAUDE_IMPLEMENTATION_PLAN.md#additional-requirement--policy-independent-action-executors).
