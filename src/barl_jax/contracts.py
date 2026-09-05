"""Interfaces between host collection, inference and replay storage."""
from typing import Callable, Protocol


class ActionProvider(Protocol):
  action_dim: int
  chunk_size: int

  def inference_fn(self) -> Callable:
    ...


class TransitionSink(Protocol):
  def add(self, **transition) -> None:
    ...
