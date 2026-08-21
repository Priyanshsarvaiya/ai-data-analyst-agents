from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
from threading import RLock
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
        self._lock = RLock()

    def add(self, kind: str, artifact_path: str, summary: str, pointer: str | None = None) -> EvidenceRef:
        raw_artifact = str(artifact_path).strip()
        artifact = Path(raw_artifact)
        if not raw_artifact or artifact == Path(".") or artifact.is_absolute() or ".." in artifact.parts:
            raise ValueError("Evidence artifact_path must be a safe run-relative path.")
        if not str(kind).strip() or not str(summary).strip():
            raise ValueError("Evidence kind and summary must be non-empty.")
        ev = EvidenceRef(
            id=f"EV-{uuid.uuid4().hex[:10]}",
            kind=str(kind).strip(),
            artifact_path=str(artifact),
            pointer=pointer,
            summary=str(summary).strip(),
        )
        with self._lock:
            while ev.id in self._items:
                ev = EvidenceRef(
                    id=f"EV-{uuid.uuid4().hex[:10]}",
                    kind=ev.kind,
                    artifact_path=ev.artifact_path,
                    pointer=ev.pointer,
                    summary=ev.summary,
                )
            self._items[ev.id] = ev
        return ev

    def all(self) -> Dict[str, EvidenceRef]:
        with self._lock:
            return dict(self._items)
