from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_run_dir(artifacts_root: Path) -> Path:
    run_dirs = sorted([p for p in artifacts_root.glob("run_*") if p.is_dir()])
    if not run_dirs:
        raise AssertionError(f"No run directory found under {artifacts_root}")
    return run_dirs[-1]


def assert_artifacts_exist(run_dir: Path, names: list[str]) -> None:
    missing = [n for n in names if not (run_dir / n).exists()]
    if missing:
        raise AssertionError(f"Missing artifacts: {missing}")
