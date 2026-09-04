"""JAX/Flax-NNX reimplementation of the barl framework.

Standalone: nothing here imports `barl`. Importing this package is what
populates the policy/environment/component registries, so anything not
imported below is unreachable from a YAML config (see networks/registry.py).
"""

from barl_jax.policy import Policy, register_policy
from barl_jax.env import Environment, register_environment

# Imported purely for @register_component side effects -- config/models.yaml
# resolves these by name, so an unused-import cleanup here would silently
# break that. `actors`/`critics` are the load-bearing ones: no module imports
# each concrete file directly, so these package imports are what register them.
import barl_jax.critics
import barl_jax.actors
import barl_jax.networks.squash
import barl_jax.networks.regulator
import barl_jax.networks.schedule

from barl_jax.policies import *
from barl_jax.environments.mujoco_gym import *
from barl_jax.environments.dm_control import *
from barl_jax.environments.mani_skill import *
from barl_jax.environments.humanoid_bench import *
