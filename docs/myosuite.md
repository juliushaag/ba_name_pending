# MyoSuite

The repository integrates MyoSuite 2.12.2's CPU MuJoCo environments through the
same `Environment` interface used by Gymnasium MuJoCo and DeepMind Control.
Run `uv sync --locked` from the repository root; no additional source checkout
or asset download is needed for the registered MyoBase tasks.

## Tasks

Use these names in `env:` or `scripts/submit.py --envs`:

| Task names | MyoSuite IDs | Episode limit | Default Q support |
| --- | --- | --- | --- |
| `MyoElbowPose1D6MFixed`, `MyoElbowPose1D6MRandom` | `myoElbowPose1D6M{Fixed,Random}-v0` | 100 | ±1200 |
| `MyoFingerPoseFixed`, `MyoFingerPoseRandom` | `myoFingerPose{Fixed,Random}-v0` | 100 | ±1200 |
| `MyoFingerReachFixed`, `MyoFingerReachRandom` | `myoFingerReach{Fixed,Random}-v0` | 100 | ±1200 |
| `MyoHandPoseFixed`, `MyoHandPoseRandom` | `myoHandPose{Fixed,Random}-v0` | 100 | ±1200 |
| `MyoHandReachFixed`, `MyoHandReachRandom` | `myoHandReach{Fixed,Random}-v0` | 100 | ±1200 |
| `MyoLegStandRandom` | `myoLegStandRandom-v0` | 150 | ±3000 |
| `MyoLegWalk` | `myoLegWalk-v0` | 1000 | ±5000 |

Small tasks use synchronous vectorization; leg tasks default to subprocesses.
`get_vector_env(..., asynchronous=...)` can override that choice. Vectorization
uses the collector's existing explicit reset masks and distinguishes time
limits from true termination for critic bootstrapping. Policy actions use the
environment's normalized `[-1, 1]` space; MyoSuite applies its own nonlinear
muscle excitation transform. No extra activation remapping is applied.

## Running

```bash
# Validate a submission without starting training.
uv run python scripts/submit.py --algos SAC --envs MyoHandReachRandom --seeds 3 --dry-run

# Short CPU training smoke run with local logging and two evaluations.
BARL_ALLOW_CPU=1 JAX_PLATFORMS=cpu uv run python scripts/train.py --config experiments/sac_myosuite_smoke.yaml

# Reload and evaluate the saved smoke checkpoint.
JAX_PLATFORMS=cpu uv run python scripts/evaluate.py /tmp/barl-sac-myosuite-smoke --env MyoElbowPose1D6MRandom --mode DirectExecutor --episodes 2

# GPU training with the usual defaults.
uv run python scripts/submit.py --algos SAC --envs MyoLegWalk --seeds 3 --executor local

# Simulator and adapter regression tests (rendering lifecycle is mocked).
uv run pytest tests/test_myosuite.py tests/test_env_supports.py -q
```

The smoke config is a plumbing check, not a tuned learning baseline. Change
`env:` and the training budget for experiments. For videos, set
`eval_video_every: 1`; training's existing EGL setup is used. Rendering creates
a context only on the first frame and closes it when the environment closes.
On machines with libOSMesa installed, `MUJOCO_GL=osmesa` selects software
rendering for CPU video checks.

## Evaluation and critic supports

The adapter reseeds MyoSuite before it generates targets or initial positions,
including the robot's sensor-noise RNG. Unseeded episode resets keep advancing
the stream. This preserves the training/evaluation seed separation used by the
rest of the framework.

`eval/success_rate` uses MyoSuite's default `evaluate_success` convention: an
episode succeeds after **more than five solved steps**, which need not be
consecutive. The adapter preserves `info['solved']` and adds the accumulated
`info['success']`; reaching the time limit alone does not imply success.

Categorical/HL-Gauss critics receive task-specific symmetric **discounted Q**
supports, using the default gamma 0.99 and roughly twice the positive discounted
reward bound. Pose/reach rewards peak at 8 per step, standing at 18, and walking
at 25. The symmetric margins also accommodate negative shaping and terminal
penalties. These are practical supports for the standard rewards, not proofs
of bounds on all unstable simulator states. Override critic `v_min`/`v_max`
when changing rewards or discount, or when support clipping appears.

## Dependencies and scope

MyoSuite 2.12.2 declares Gymnasium `<1.3` and MuJoCo `>=3.6,<3.7`. This repo's
existing `tool.uv.override-dependencies` keeps Gymnasium 1.3 and MuJoCo 3.10
shared with the other simulator families. The integration tests exercise the
registered tasks against those versions. A plain `pip install` of the root
project does not apply uv's overrides; use the repository's uv workflow.

MyoSuite supports Python 3.10–3.13; this repo starts at 3.11, so its MyoSuite
dependency is installed on 3.11–3.13. Registry discovery still works without
MyoSuite; selecting a task reports the missing dependency. Simulator tests
skip if MyoSuite is absent in a minimal installation.

This integration covers the twelve listed MyoBase tasks. MyoChallenge, fatigue
and sarcopenia variants, additional manipulation tasks, and MyoSuite's MJX/GPU
physics backend are not registered. Additional tasks can subclass
`MyoSuiteEnvironment`, implement `myosuite_id()`, declare appropriate `stats`,
and be selected via the existing `module:Class` mechanism.
