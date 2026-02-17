from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactStore:
    run_dir: Path

    @staticmethod
    def create(base_dir: str | Path = "artifacts") -> "ArtifactStore":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(base_dir) / f"run_{ts}"
        (run_dir / "charts").mkdir(parents=True, exist_ok=True)
        return ArtifactStore(run_dir=run_dir)

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.run_dir / name
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self.run_dir / name
        p.write_text(text, encoding="utf-8")
        return p

    def path(self, name: str) -> Path:
        return self.run_dir / name