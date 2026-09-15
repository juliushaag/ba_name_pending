# The two shared training templates. Each per-algorithm Policy class (SAC,
# AMDP, DIME, SDAC, NCLQL) is a thin subclass of one of these, registered
# beside its actor in barl_jax.actors -- see registration.load_builtins's
# "policies" group, which imports both this package and barl_jax.actors.
from barl_jax.policies.rl import MaxEntRLPolicy, RLPolicy  # noqa: F401
