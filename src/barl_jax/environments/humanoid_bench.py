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
import humanoid_bench  # noqa: F401 -- side effect: registers every <robot>-<task>-v0 gym id

from barl_jax.env import Environment, register_environment


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


_patch_stale_mujoco_fields()


class HumanoidBenchEnvironment(Environment):
  """One HumanoidBench task id. Online-only, like MuJoCoGymEnvironment --
  get_dataset() always raises."""

  use_async_vector_env = True

  # HumanoidBench's per-task reward scale isn't documented per task; a random
  # h1hand policy measures ~0.2/step -- the same order as DMC -- so this reuses
  # DMC's +-200 bound. Horizon is 1000, so the discounting derivation applies
  # unchanged. A task whose actual returns run past this should get its own.
  stats = {"return_min": -200.0, "return_max": 200.0}

  def task_id(self) -> str:
    raise NotImplementedError

  def get_env(self) -> gym.Env:
    return gym.make(self.task_id(), render_mode="rgb_array")

  def get_dataset(self):
    raise NotImplementedError(f"{type(self).__name__} has no offline dataset -- HumanoidBench is online-only.")


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
  def task_id(self): return "h1hand-reach-v0"

@register_environment
class H1HandPush(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-push-v0"

@register_environment
class H1HandCabinet(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-cabinet-v0"

@register_environment
class H1HandDoor(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-door-v0"

@register_environment
class H1HandTruck(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-truck-v0"

@register_environment
class H1HandCube(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-cube-v0"

@register_environment
class H1HandBookshelfSimple(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-bookshelf_simple-v0"

@register_environment
class H1HandBookshelfHard(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-bookshelf_hard-v0"

@register_environment
class H1HandBasketball(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-basketball-v0"

@register_environment
class H1HandWindow(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-window-v0"

@register_environment
class H1HandSpoon(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-spoon-v0"

@register_environment
class H1HandKitchen(HumanoidBenchEnvironment):
  def task_id(self): return "h1hand-kitchen-v0"

@register_environment
class H1HandPackage(HumanoidBenchEnvironment):
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
