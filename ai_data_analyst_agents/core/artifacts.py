from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ArtifactStore:
    run_dir: Path
    on_artifact_written: Callable[[Path, Path], None] | None = None

    @staticmethod
    def create(
        base_dir: str | Path = "artifacts",
        on_artifact_written: Callable[[Path, Path], None] | None = None,
    ) -> "ArtifactStore":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(base_dir) / f"run_{ts}"
        (run_dir / "charts").mkdir(parents=True, exist_ok=True)
        return ArtifactStore(run_dir=run_dir, on_artifact_written=on_artifact_written)

    def _notify(self, p: Path) -> None:
        if self.on_artifact_written is None:
            return
        self.on_artifact_written(self.run_dir, p)

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.run_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        self._notify(p)
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self.run_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        self._notify(p)
        return p

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def register_file(self, name: str | Path) -> None:
        p = self.run_dir / name if not isinstance(name, Path) or not name.is_absolute() else name
        if p.exists() and p.is_file():
            self._notify(p)
