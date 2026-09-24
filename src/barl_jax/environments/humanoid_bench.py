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

from barl_jax.env import Environment, register_environment


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
  """One HumanoidBench task id. Online-only, like MuJoCoGymEnvironment --
  get_dataset() always raises."""

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



# ---- locomotion ----

@register_environment
class H1HandWalk(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-walk-v0"

@register_environment
class H1HandStand(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-stand-v0"

@register_environment
class H1HandRun(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-run-v0"

@register_environment
class H1HandCrawl(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-crawl-v0"

@register_environment
class H1HandHurdle(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-hurdle-v0"

@register_environment
class H1HandMaze(HumanoidBenchEnvironment):
  # get_reward's <=1/step shaping term (discounted bound ~100, as in the
  # family default) is joined by a sparse `stage_convert_reward` of
  # 100*stage_index at each of 4 checkpoint crossings (100,200,300,400).
  # Discounting 4 back-to-back bonuses at gamma=0.99 barely dents their
  # 1000 undiscounted sum (still ~980) -- the value near the last crossing
  # is ~1080, doubled for headroom.
  stats = {"return_min": -2200.0, "return_max": 2200.0}
  def task_id(self): return "h1hand-maze-v0"

@register_environment
class H1HandSitSimple(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-sit_simple-v0"

@register_environment
class H1HandSitHard(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-sit_hard-v0"

@register_environment
class H1HandBalanceSimple(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-balance_simple-v0"

@register_environment
class H1HandBalanceHard(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-balance_hard-v0"

@register_environment
class H1HandStair(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-stair-v0"

@register_environment
class H1HandSlide(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-slide-v0"

@register_environment
class H1HandPole(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-pole-v0"

@register_environment
class H1HandHighbarSimple(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-highbar_simple-v0"

@register_environment
class H1HandHighbarHard(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-highbar_hard-v0"


# ---- manipulation ----

@register_environment
class H1HandReach(HumanoidBenchEnvironment):
  # get_reward = healthy_reward(<=5) - 1e-4*motion_penalty + reward_close(<=5)
  # + reward_success(<=10): once the hand holds position all three positive
  # terms are sustained every step, ~20/step rather than the family's ~1 --
  # 20x the shared bound, doubled for headroom.
  stats = {"return_min": -4000.0, "return_max": 4000.0}
  def task_id(self): return "h1hand-reach-v0"

@register_environment
class H1HandPush(HumanoidBenchEnvironment):
  # get_reward's dense term is a small distance penalty, but reaching the
  # goal (box within 5cm) adds a one-off +1000 and terminates the episode --
  # the value approaching that transition is ~1000, doubled for headroom.
  stats = {"return_min": -2000.0, "return_max": 2000.0}
  def task_id(self): return "h1hand-push-v0"

@register_environment
class H1HandCabinet(HumanoidBenchEnvironment):
  # 4 subtasks each pay a one-off 100*subtask_index completion bonus on top
  # of a <=1/step stabilization/subtask term (discounted bound ~100). Same
  # near-back-to-back discounting as Maze's 4 checkpoints: 100+200+300+400
  # discounts to ~980 at gamma=0.99, so the value near the last completion
  # is ~1080, doubled for headroom.
  stats = {"return_min": -2200.0, "return_max": 2200.0}
  def task_id(self): return "h1hand-cabinet-v0"

@register_environment
class H1HandDoor(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-door-v0"

@register_environment
class H1HandTruck(HumanoidBenchEnvironment):
  # 5 packages each pay a pickup (+100) and a place (+100) bonus -- 10
  # events, discounting to ~960 of their 1000 undiscounted sum at gamma=0.99
  # if reached back-to-back -- plus a <=4/step (not <=1, unlike most tasks:
  # `upright * (1 + 3 proximity terms)`) background whose discounted bound
  # is ~400. Together ~1360, doubled for headroom.
  stats = {"return_min": -2800.0, "return_max": 2800.0}
  def task_id(self): return "h1hand-truck-v0"

@register_environment
class H1HandCube(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-cube-v0"

@register_environment
class H1HandBookshelfSimple(HumanoidBenchEnvironment):
  # get_reward's <=1/step shaping term (discounted bound ~100) is joined by
  # a one-off 100*task_index bonus per shelved object, for 5 objects
  # (100+200+300+400+500=1500 undiscounted). Discounting 5 back-to-back
  # placements at gamma=0.99 only brings that to ~1460 -- the value near the
  # last placement is ~1560, doubled for headroom.
  stats = {"return_min": -3000.0, "return_max": 3000.0}
  def task_id(self): return "h1hand-bookshelf_simple-v0"

@register_environment
class H1HandBookshelfHard(HumanoidBenchEnvironment):
  stats = {"return_min": -3000.0, "return_max": 3000.0}
  def task_id(self): return "h1hand-bookshelf_hard-v0"

@register_environment
class H1HandBasketball(HumanoidBenchEnvironment):
  # get_terminated ends the episode the instant the ball is within 5cm of
  # the hoop, the same step get_reward's +1000 fires -- like Push/Package,
  # the value approaching that transition is ~1000 undiscounted (gamma
  # applies only to rewards *after* this step, and there are none), doubled
  # for headroom.
  stats = {"return_min": -2000.0, "return_max": 2000.0}
  def task_id(self): return "h1hand-basketball-v0"

@register_environment
class H1HandWindow(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-window-v0"

@register_environment
class H1HandSpoon(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-spoon-v0"

@register_environment
class H1HandKitchen(HumanoidBenchEnvironment):
  # get_reward is `len(completions)` -- at most one point per subtask, ever,
  # since completed subtasks are removed from tasks_to_complete. success_bar
  # (4) is the whole task's subtask count; doubled for headroom.
  stats = {"return_min": -8.0, "return_max": 8.0}
  def task_id(self): return "h1hand-kitchen-v0"

@register_environment
class H1HandPackage(HumanoidBenchEnvironment):
  # get_reward's dense term is a distance penalty, but reaching the
  # destination (within 10cm) adds a one-off +1000 and get_terminated ends
  # the episode the same step -- like Push, the value approaching that
  # transition is ~1000 undiscounted (no future rewards left to discount),
  # doubled for headroom.
  stats = {"return_min": -2000.0, "return_max": 2000.0}
  def task_id(self): return "h1hand-package-v0"

@register_environment
class H1HandPowerlift(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-powerlift-v0"

@register_environment
class H1HandRoom(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-room-v0"

@register_environment
class H1HandInsertNormal(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-insert_normal-v0"

@register_environment
class H1HandInsertSmall(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-insert_small-v0"
