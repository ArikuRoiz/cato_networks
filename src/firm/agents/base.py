"""BaseAgent — typed contract every agent must satisfy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseAgent(ABC, Generic[InputT, OutputT]):  # noqa: UP046
    @abstractmethod
    def run(self, inp: InputT) -> OutputT: ...
