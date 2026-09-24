"""Small public configuration API.

A training YAML file contains one mapping and is materialized by
:class:`barl_jax.config.run.RunConfig`.  Built-in defaults are owned by the
typed classes in :mod:`barl_jax.config.schemas`; this module does no include
resolution or root overlay work.
"""

from pathlib import Path
from typing import Any

import yaml

from barl_jax.config.run import RunConfig


def from_yaml(path: str | Path) -> RunConfig:
  """Load exactly one run mapping from YAML."""
  path = Path(path)
  with path.open() as stream:
    data: Any = yaml.safe_load(stream)
  if data is None:
    raise ValueError(f"{path}: empty YAML does not describe a run")
  if not isinstance(data, dict):
    raise TypeError(f"{path}: expected one YAML mapping, got {type(data).__name__}")
  return RunConfig.from_dict(data)


__all__ = ["RunConfig", "from_yaml"]
