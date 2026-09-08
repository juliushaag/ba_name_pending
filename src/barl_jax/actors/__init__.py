"""Built-in algorithms and their actor-owned policy definitions."""
from .amdp import ScoreContinuousActor
from .dime import DISActor
from .nclql import LangevinActor
from .policy_dict import bind_actor_classes
from .sac import StochasticActor
from .sdac import DDPMActor

ALGORITHMS = {
  "AMDP": ScoreContinuousActor,
  "DIME": DISActor,
  "NCLQL": LangevinActor,
  "SAC": StochasticActor,
  "SDAC": DDPMActor,
}

bind_actor_classes(ALGORITHMS)
