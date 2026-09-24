"""HumanoidBench (Sferrazza et al. 2024) -- the Unitree H1 whole-body
locomotion/manipulation benchmark, wrapped through the `third_party/humanoid-bench`
submodule (see its `[tool.uv.sources]` entry in pyproject.toml for why that's a
submodule + editable install rather than a plain git dependency).

`import humanoid_bench` is what registers every `<robot>-<task>-v0` gym id;
this file only ever uses the `h1hand` robot (the 5-finger hand, torque-limited
variant), which is the one every one of the suite's 32 tasks supports --
`h1strong`/`h1simplehand`/`h1touch`/`g1`/`digit` are other robot embodiments
for the same tasks, not registered here.

humanoid_bench pins MuJoCo 3.1.6 in its own setup.py (overridden in
pyproject.toml -- see the override-dependencies comment there), and its
dm_control-style named struct indexing (`dmc_deps/dmc_sizes.py`) is a table
generated against that exact version's mjModel/mjData layout. A MuJoCo this
much newer has since renamed/removed a handful of struct fields (e.g.
`flex_xvert0` -> `flex_vert0`), which without `_patch_stale_mujoco_fields()`
below raises an AttributeError building *any* task's `env.named` the moment
its `__init__` runs. None of humanoid_bench's own tasks read flex-* fields
(those are for cloth/rope simulation), so pruning whatever this MuJoCo build
doesn't have from the sizing table -- once, up front -- costs nothing."""

import gymnasium as gym

from barl_jax.env import Environment, register_tasks


# `humanoid_bench.wrappers` and `.envs.{reach,push,package}` each do
# `from humanoid_bench.mjx.flax_to_torch import TorchModel, TorchPolicy` at
# module level -- an eager `import torch` -- but only wrappers.py's
# SingleReachWrapper/DoubleReachBaseWrapper ever construct one, and neither
# is used by any task registered below (this file wraps Reach/Push/Package's
# own classes directly, not those wrappers). Pre-seeding sys.modules with a
# stub for that one submodule, before `import humanoid_bench` below ever
# triggers the chain, keeps a second CUDA context (torch's) out of every
# HumanoidBench run's process for two classes nothing here uses -- and makes
# this importable at all in a venv without torch installed. Python's import
# machinery resolves `humanoid_bench.mjx.flax_to_torch` as a full dotted
# name straight out of sys.modules, so this never touches the real
# `humanoid_bench.mjx` package (untouched, still real if something else
# needs it) or `torch` itself.
def _stub_unused_torch_dependency():
  import sys
  import types

  name = "humanoid_bench.mjx.flax_to_torch"
  if name in sys.modules:
    return

  class _UnusedTorchStub:
    def __init__(self, *args, **kwargs):
      raise RuntimeError(
        f"{type(self).__name__} is a barl_jax stub (see "
        "_stub_unused_torch_dependency in environments/humanoid_bench.py) -- "
        "no task registered here uses humanoid_bench.wrappers."
        "SingleReachWrapper/DoubleReachBaseWrapper, the only real callers. "
        "Install torch and delete this stub if you need one of those.")

  stub = types.ModuleType(name)
  stub.TorchModel = _UnusedTorchStub
  stub.TorchPolicy = _UnusedTorchStub
  sys.modules[name] = stub


def _patch_stale_mujoco_fields():
  import mujoco
  from humanoid_bench.dmc_deps import dmc_sizes

  scratch_model = mujoco.MjModel.from_xml_string("<mujoco/>")
  scratch_data = mujoco.MjData(scratch_model)
  probes = {"mjmodel": scratch_model, "mjdata": scratch_data}
  for struct_name, fields in dmc_sizes.array_sizes.items():
    probe = probes[struct_name]
    for field_name in [name for name in fields if not hasattr(probe, name)]:
      del fields[field_name]


# humanoid_bench pins gymnasium<1.0 in its own setup.py (also overridden in
# pyproject.toml, alongside the MuJoCo override above). Every task's
# render() -- Task.render plus the Push/Reach overrides that additionally
# reposition a goal marker before rendering -- calls the pre-1.0
# `MujocoRenderer.render(render_mode, camera_id, camera_name)` signature.
# gymnasium>=1.0 moved camera selection into `MujocoRenderer.__init__`
# (already applied via the `camera_name=` env.py passes when building the
# renderer) and narrowed render() to just `render(render_mode)`, so those
# extra two positional args now raise a TypeError on every frame. Patching
# MujocoRenderer.render itself -- rather than each task's render() override
# -- fixes every call site in one place without duplicating their marker
# logic; the ignored args are a no-op since the camera is already fixed at
# construction.
def _patch_stale_gymnasium_render():
  from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

  if getattr(MujocoRenderer.render, "_barl_jax_patched", False):
    return

  original_render = MujocoRenderer.render

  def render(self, render_mode, camera_id=None, camera_name=None):
    return original_render(self, render_mode)

  render._barl_jax_patched = True
  MujocoRenderer.render = render


# The goal markers in Reach/Push's render() overrides (and the unused
# double-reach/blocked-hands wrappers) call `viewer.add_marker(size=0.05, ...)`
# -- a bare radius, valid under MuJoCo's pre-3.2 marker API where a scalar
# broadcast into the geom's fixed-size 3-vector. MuJoCo>=3.2 (pinned above)
# switched gymnasium's non-legacy `_add_marker_to_scene` to call
# `mujoco.mjv_initGeom` directly, whose binding requires an actual 3-element
# `size` array and rejects a bare float. Normalizing a scalar `size` to
# `[size]*3` at `add_marker` time (its one call site) keeps every caller's
# radius semantics unchanged.
def _patch_stale_gymnasium_marker_size():
  import numpy as np
  from gymnasium.envs.mujoco.mujoco_rendering import BaseRender

  if getattr(BaseRender.add_marker, "_barl_jax_patched", False):
    return

  original_add_marker = BaseRender.add_marker

  def add_marker(self, **marker_params):
    size = marker_params.get("size")
    if isinstance(size, (int, float)):
      marker_params["size"] = np.array([size, size, size])
    return original_add_marker(self, **marker_params)

  add_marker._barl_jax_patched = True
  BaseRender.add_marker = add_marker


class HumanoidBenchEnvironment(Environment):
  """One HumanoidBench task id. Online-only, like MuJoCoGymEnvironment."""

  use_async_vector_env = True

  # Most tasks' get_reward is a weighted sum of dm_control-style tolerance()
  # terms whose weights sum to <=1, each term itself in [0, 1] -- the same
  # per-step shape as DMC's suite (verified against basic_locomotion_envs.py,
  # pole, highbar, balance, room, powerlift, spoon, window, door, cube and
  # insert's source). So the DMC derivation applies unchanged: twice the
  # discounted bound (1-gamma^T)/(1-gamma) at gamma 0.99, horizon 1000,
  # symmetric. Tasks whose get_reward instead sustains a larger per-step
  # magnitude (reach) or adds a sparse completion bonus well past 1 (push,
  # package, basketball, cabinet, bookshelf, truck, maze, kitchen) override
  # this below with their own derivation.
  stats = {"return_min": -200.0, "return_max": 200.0}

  def task_id(self) -> str:
    raise NotImplementedError

  def get_env(self, *, render: bool = False) -> gym.Env:
    _stub_unused_torch_dependency()
    import humanoid_bench  # noqa: F401
    _patch_stale_mujoco_fields()
    _patch_stale_gymnasium_render()
    _patch_stale_gymnasium_marker_size()
    # humanoid_bench's own Task.__init__ (third_party/humanoid-bench/humanoid_bench/tasks.py)
    # builds a full offscreen GL/EGL viewer immediately whenever render_mode is
    # not None -- unlike gymnasium's own MujocoRenderer, this is eager, not
    # lazy on first render(). A training rollout, the reference env and eval's
    # return-only env never call render(), so render_mode=None skips that
    # entirely for them -- one less GL context to fail on a flaky headless node.
    return gym.make(self.task_id(), render_mode="rgb_array" if render else None)


_TASKS = {
  "H1HandWalk": "h1hand-walk-v0",
  "H1HandStand": "h1hand-stand-v0",
  "H1HandRun": "h1hand-run-v0",
  "H1HandCrawl": "h1hand-crawl-v0",
  "H1HandHurdle": "h1hand-hurdle-v0",
  "H1HandMaze": "h1hand-maze-v0",
  "H1HandSitSimple": "h1hand-sit_simple-v0",
  "H1HandSitHard": "h1hand-sit_hard-v0",
  "H1HandBalanceSimple": "h1hand-balance_simple-v0",
  "H1HandBalanceHard": "h1hand-balance_hard-v0",
  "H1HandStair": "h1hand-stair-v0",
  "H1HandSlide": "h1hand-slide-v0",
  "H1HandPole": "h1hand-pole-v0",
  "H1HandHighbarSimple": "h1hand-highbar_simple-v0",
  "H1HandHighbarHard": "h1hand-highbar_hard-v0",
  "H1HandReach": "h1hand-reach-v0",
  "H1HandPush": "h1hand-push-v0",
  "H1HandCabinet": "h1hand-cabinet-v0",
  "H1HandDoor": "h1hand-door-v0",
  "H1HandTruck": "h1hand-truck-v0",
  "H1HandCube": "h1hand-cube-v0",
  "H1HandBookshelfSimple": "h1hand-bookshelf_simple-v0",
  "H1HandBookshelfHard": "h1hand-bookshelf_hard-v0",
  "H1HandBasketball": "h1hand-basketball-v0",
  "H1HandWindow": "h1hand-window-v0",
  "H1HandSpoon": "h1hand-spoon-v0",
  "H1HandKitchen": "h1hand-kitchen-v0",
  "H1HandPackage": "h1hand-package-v0",
  "H1HandPowerlift": "h1hand-powerlift-v0",
  "H1HandRoom": "h1hand-room-v0",
  "H1HandInsertNormal": "h1hand-insert_normal-v0",
  "H1HandInsertSmall": "h1hand-insert_small-v0",
}

# Tasks whose reward is not the family's <=1/step shaping term. Each bound is twice the
# discounted value near the largest sustained or one-off reward, at gamma 0.99:
#   Maze, Cabinet     4 checkpoint/subtask bonuses of 100*index (100..400) on top of a
#                     <=1/step term: ~1080 near the last one.
#   Bookshelf         5 shelving bonuses of 100*index (100..500): ~1560.
#   Truck             10 pickup/place bonuses of 100, plus a <=4/step background (~400): ~1360.
#   Reach             healthy(<=5) + close(<=5) + success(<=10) sustained: ~20/step, 20x the family.
#   Push, Package,    a one-off +1000 that also terminates the episode, so nothing is
#   Basketball        left to discount: ~1000.
#   Kitchen           one point per subtask, four subtasks.
_RETURN_BOUNDS = {
  "H1HandMaze": 2200.0,
  "H1HandReach": 4000.0,
  "H1HandPush": 2000.0,
  "H1HandCabinet": 2200.0,
  "H1HandTruck": 2800.0,
  "H1HandBookshelfSimple": 3000.0,
  "H1HandBookshelfHard": 3000.0,
  "H1HandBasketball": 2000.0,
  "H1HandKitchen": 8.0,
  "H1HandPackage": 2000.0,
}

register_tasks(HumanoidBenchEnvironment, "task_id", _TASKS, into=globals(), bounds=_RETURN_BOUNDS)
