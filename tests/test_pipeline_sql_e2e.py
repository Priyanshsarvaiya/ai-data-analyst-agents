from __future__ import annotations

from pathlib import Path

import pytest

from ai_data_analyst_agents.pipelines.run_sql_pipeline import _redact_db_url, run_pipeline
from tests.helpers import assert_artifacts_exist, read_json


def test_sql_pipeline_end_to_end(sqlite_star_db, patch_llm, patch_pipeline_cfg: Path) -> None:
    run_dir = Path(
        run_pipeline(
            db_url=f"sqlite:///{sqlite_star_db}",
            business_question="Why did India have less revenue than other countries?",
            base_table="orders",
        )
    )

    assert_artifacts_exist(
        run_dir,
        ["db_schema.json", "analysis_tasks.json", "metrics_outputs.json", "final_report.md"],
    )

    plan = read_json(run_dir / "analysis_tasks.json")
    task_types = [t["type"] for t in plan["tasks"]]
    assert "sql_query" in task_types
    assert "sql_join_profile" in task_types

    report = (run_dir / "final_report.md").read_text(encoding="utf-8")
    assert "India" in report
    assert "## 9) Evidence References" in report
    assert "[1]" in report


def test_sql_pipeline_invalid_source_raises(patch_llm, patch_pipeline_cfg: Path, tmp_path: Path) -> None:
    # SQLite creates file automatically, but no tables exist -> pipeline should fail with clear error.
    empty_db = tmp_path / "empty.db"
    with pytest.raises(ValueError, match="No tables found"):
        run_pipeline(
            db_url=f"sqlite:///{empty_db}",
            business_question="Any question",
            base_table="orders",
        )


def test_db_url_redaction() -> None:
    url = "postgresql+psycopg://user:secret@localhost:5432/analytics"
    redacted = _redact_db_url(url)
    assert "secret" not in redacted
    assert "user:***@" in redacted
