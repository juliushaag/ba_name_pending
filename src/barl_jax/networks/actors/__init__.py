# Imported purely for @register_component side effects -- config/models.yaml
# resolves these actor classes by name, and nothing imports them directly. An
# unused-import cleanup here would silently break every model slot that names
# one, which is why barl_jax/__init__ explicitly imports this package.
#
# The actors used to live in barl_jax/policies/ as the SAC/AMDP/DIME/SDAC
# policy modules; the policies are now the RLPolicy template and the classes
# moved here to sit next to the critics they share a registry with. Each file
# keeps the old name's shape (stochastic.py = StochasticActor, score.py =
# ScoreContinuousActor + clip_score, dis.py = DISActor + _inv_softplus,
# ddpm.py = DDPMActor + alpha_bar_from_betas).
from barl_jax.networks.actors import stochastic, score, dis, ddpm  # noqa: F401