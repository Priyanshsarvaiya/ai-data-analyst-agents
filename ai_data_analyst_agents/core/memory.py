from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class SharedMemory:
    facts: Dict[str, Any] = field(default_factory=dict)
    messages: List[Any] = field(default_factory=list)

    def set(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    def log(self, msg: Any) -> None:
        self.messages.append(msg)
