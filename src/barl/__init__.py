from barl.policy import Policy, register_policy
from barl.env import Environment, register_environment

# Imported purely for @register_component side effects -- config/models.yaml
# resolves these by name, so an unused-import cleanup here would silently
# break that (see barl/networks/registry.py).
import barl.networks.actor
import barl.networks.critic
import barl.networks.squash

from barl.policies import *
from barl.environments.mujoco_gym import *
from barl.environments.dm_control import *
from barl.environments.mujoco_playground import *

