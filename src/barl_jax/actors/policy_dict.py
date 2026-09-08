"""The shared shape behind every actor's `POLICY` dict."""


def maxent_actor_policy(*, tier: str = "medium", critic: str = "C51Critic",
                        critic_tier: str = "xlarge", entropy: str = "AutotuneRegulator",
                        **params) -> dict:
  """`**params` should be only the real deltas from `MaxEntRLPolicy`'s own
  Config defaults (`policies/rl.py`) -- restating a default here does nothing
  but rot, since `ConfigTree`'s "decl params" layer already supplies it.
  `components.actor.class` is deliberately not set here -- `bind_actor_classes`
  injects each algorithm's own class name once, after every actor class
  exists (`POLICY` is assigned inside the class body, before the class object
  itself exists). `"params"` is always present, even empty, so
  `ConfigTree.models[name]["params"]` is always a dict."""
  return {
    "policy": "MaxEntRLPolicy",
    "params": dict(params),
    "components": {
      "actor": {"tier": tier},
      "critic": {"class": critic, "tier": critic_tier},
      "entropy": {"class": entropy},
    },
  }


def bind_actor_classes(algorithms: dict) -> None:
  """Inject each algorithm's own class name into its POLICY's actor slot --
  `components.actor.class` is always `type(self).__name__`; `ALGORITHMS`
  already maps the model name to this exact class, so this is the one place
  that needs to spell it out."""
  for cls in algorithms.values():
    cls.POLICY["components"]["actor"]["class"] = cls.__name__
