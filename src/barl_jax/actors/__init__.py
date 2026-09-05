# Imported purely for @register_component side effects -- config/models.yaml
# resolves these actor classes by name, and nothing imports them directly. An
# unused-import cleanup here would silently break every model slot that names
# one, which is why barl_jax/__init__ explicitly imports this package.
#
# The concrete actors are grouped here by owning algorithm: sac.py =
# StochasticActor, amdp.py = ScoreContinuousActor + clip_score, dime.py =
# DISActor + _inv_softplus, sdac.py = DDPMActor + alpha_bar_from_betas.
from barl_jax.actors import amdp, dime, sac, sdac  # noqa: F401
