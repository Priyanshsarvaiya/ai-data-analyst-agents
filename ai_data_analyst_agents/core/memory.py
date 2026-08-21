from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock, local
from typing import Any


@dataclass
class SharedMemory:
    facts: dict[str, Any] = field(default_factory=dict)
    messages: list[Any] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)
    _actor_local: local = field(default_factory=local, init=False, repr=False)
    _access_counts: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict, init=False, repr=False)

    def _actor(self) -> str:
        return str(getattr(self._actor_local, "name", "external"))

    def _record(self, operation: str, key: str) -> None:
        actor = self._actor()
        actor_counts = self._access_counts.setdefault(actor, {"read": {}, "write": {}, "log": {}})
        bucket = actor_counts.setdefault(operation, {})
        bucket[key] = int(bucket.get(key, 0)) + 1

    @contextmanager
    def as_actor(self, name: str) -> Iterator[None]:
        previous = self._actor()
        self._actor_local.name = str(name)
        try:
            yield
        finally:
            self._actor_local.name = previous

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.facts[key] = value
            self._record("write", str(key))

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            self._record("read", str(key))
            return self.facts.get(key, default)

    def log(self, msg: Any) -> None:
        with self._lock:
            self.messages.append(msg)
            self._record("log", type(msg).__name__)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"facts": dict(self.facts), "messages": list(self.messages)}

    def audit(self) -> dict[str, Any]:
        with self._lock:
            return {
                "actors": {
                    actor: {
                        operation: dict(keys)
                        for operation, keys in operations.items()
                    }
                    for actor, operations in self._access_counts.items()
                },
                "fact_keys": sorted(self.facts),
                "message_count": len(self.messages),
            }
