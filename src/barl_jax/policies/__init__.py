# SAC, AMDP, DIME and SDAC are all `rl.MaxEntRLPolicy` configs now -- there are
# no per-algorithm policy classes left. The actor classes each still names are
# registered from barl_jax.actors (which barl_jax/__init__ imports);
# importing this package is what registers the two policies below.
from barl_jax.policies.lql import NCLQL  # noqa: F401
from barl_jax.policies.rl import MaxEntRLPolicy, RLPolicy  # noqa: F401
