from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

class Agent(ABC):
    name: str

    @abstractmethod
    def run(self, ctx: Dict[str, Any]) -> Any:
        raise NotImplementedError