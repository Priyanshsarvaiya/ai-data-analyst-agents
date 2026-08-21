from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


@dataclass(frozen=True)
class ArtifactStore:
    run_dir: Path
    on_artifact_written: Callable[[Path, Path], None] | None = None

    @staticmethod
    def create(
        base_dir: str | Path = "artifacts",
        on_artifact_written: Callable[[Path, Path], None] | None = None,
    ) -> "ArtifactStore":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = Path(base_dir) / f"run_{ts}_{uuid4().hex[:6]}"
        (run_dir / "charts").mkdir(parents=True, exist_ok=True)
        return ArtifactStore(run_dir=run_dir, on_artifact_written=on_artifact_written)

    def _notify(self, p: Path) -> None:
        if self.on_artifact_written is None:
            return
        self.on_artifact_written(self.run_dir, p)

    def _resolve(self, name: str | Path) -> Path:
        relative = Path(name)
        if relative.is_absolute():
            raise ValueError(f"Artifact path must be relative to the run directory: {name}")
        root = self.run_dir.resolve()
        candidate = (self.run_dir / relative).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"Artifact path escapes the run directory: {name}")
        return candidate

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, set):
            return sorted(value, key=str)
        if hasattr(value, "item"):
            scalar = value.item()
            if isinstance(scalar, float) and not math.isfinite(scalar):
                return None
            return scalar
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    @classmethod
    def _sanitize_json(cls, value: Any) -> Any:
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {str(k): cls._sanitize_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._sanitize_json(v) for v in value]
        if isinstance(value, set):
            return [cls._sanitize_json(v) for v in sorted(value, key=str)]
        if hasattr(value, "item"):
            return cls._sanitize_json(value.item())
        return value

    def _atomic_write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            Path(temp_name).replace(path)
        except Exception:
            Path(temp_name).unlink(missing_ok=True)
            raise

    def write_json(self, name: str, obj: Any) -> Path:
        p = self._resolve(name)
        encoded = json.dumps(
            self._sanitize_json(obj),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=self._json_default,
        )
        self._atomic_write(p, encoded)
        self._notify(p)
        return p

    def write_text(self, name: str, text: str) -> Path:
        p = self._resolve(name)
        self._atomic_write(p, str(text))
        self._notify(p)
        return p

    def path(self, name: str) -> Path:
        return self._resolve(name)

    def register_file(self, name: str | Path) -> None:
        p = self._resolve(name) if not isinstance(name, Path) or not name.is_absolute() else name.resolve()
        root = self.run_dir.resolve()
        if p != root and root not in p.parents:
            raise ValueError(f"Registered file must be inside the run directory: {name}")
        if p.exists() and p.is_file():
            self._notify(p)
