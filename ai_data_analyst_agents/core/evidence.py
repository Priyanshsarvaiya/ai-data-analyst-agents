from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import uuid

@dataclass(frozen=True)
class EvidenceRef:
    id: str
    kind: str
    artifact_path: str
    pointer: Optional[str]
    summary: str

class EvidenceStore:
    def __init__(self) -> None:
        self._items: Dict[str, EvidenceRef] = {}

    def add(self, kind: str, artifact_path: str, summary: str, pointer: str | None = None) -> EvidenceRef:
        ev = EvidenceRef(
            id=f"EV-{uuid.uuid4().hex[:10]}",
            kind=kind,
            artifact_path=artifact_path,
            pointer=pointer,
            summary=summary,
        )
        self._items[ev.id] = ev
        return ev

    def all(self) -> Dict[str, EvidenceRef]:
        return dict(self._items)