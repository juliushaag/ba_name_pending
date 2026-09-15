"""Built-in algorithms and their actor classes."""
from .amdp import ScoreContinuousActor
from .dime import DISActor
from .nclql import LangevinActor
from .sac import StochasticActor
from .sdac import DDPMActor

ALGORITHMS = {
  "AMDP": ScoreContinuousActor,
  "DIME": DISActor,
  "NCLQL": LangevinActor,
  "SAC": StochasticActor,
  "SDAC": DDPMActor,
}
