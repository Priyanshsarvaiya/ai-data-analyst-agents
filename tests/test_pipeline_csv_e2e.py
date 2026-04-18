from __future__ import annotations
from pathlib import Path
import re

import pandas as pd
import pytest

from ai_data_analyst_agents.pipelines.run_csv_pipeline import run_pipeline
from tests.helpers import assert_artifacts_exist, latest_run_dir, read_json


def test_csv_pipeline_end_to_end_outputs(
    sample_csv_path: Path,
    patch_llm,
    patch_pipeline_cfg: Path,
) -> None:
    run_dir = run_pipeline(str(sample_csv_path), "Why did India have less revenue than other countries?")
    run_dir = Path(run_dir)

    expected_files = [
        "analysis_plan.json",
        "data_profile.json",
        "quality_report.json",
        "cleaned.csv",
        "feature_log.json",
        "analysis_tasks.json",
        "metrics_outputs.json",
        "next_steps_plan.json",
        "next_steps_metrics_outputs.json",
        "eda_summary.json",
        "final_report.md",
        "report_metadata.json",
        "review_log.json",
        "run_scorecard.json",
        "agent_messages.json",
        "run_manifest.json",
    ]
    assert_artifacts_exist(run_dir, expected_files)

    report = (run_dir / "final_report.md").read_text(encoding="utf-8")
    for section in [
        "## 1) Executive Summary",
        "## 2) Question Answer (Evidence)",
        "## 5) Analysis Outputs",
    ]:
        assert section in report
    assert "## 8) Evidence References" in report
    assert re.search(r"\[\d+\]", report)

    review = read_json(run_dir / "review_log.json")
    assert review["status"] in {"pass", "fail"}

    metrics = read_json(run_dir / "metrics_outputs.json")
    assert metrics.get("schema_version")
    for item in metrics.get("computed", []):
        assert (run_dir / item["artifact"]).exists()

    scorecard = read_json(run_dir / "run_scorecard.json")
    assert scorecard["final_quality_status"] in {"pass", "fail"}
    manifest = read_json(run_dir / "run_manifest.json")
    assert manifest.get("quality_status") in {"pass", "fail"}


def test_csv_pipeline_handles_missing_expected_columns(
    tmp_path: Path, patch_llm, patch_pipeline_cfg: Path
) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p = tmp_path / "missing_cols.csv"
    df.to_csv(p, index=False)

    run_dir = Path(run_pipeline(str(p), "Total revenue by country?"))
    assert (run_dir / "final_report.md").exists()
    assert (run_dir / "run_manifest.json").exists()


def test_csv_pipeline_handles_wrong_dtypes_without_crashing(
    tmp_path: Path, patch_llm, patch_pipeline_cfg: Path
) -> None:
    df = pd.DataFrame(
        {
            "order_id": ["1", "2", "3"],
            "country": ["India", "USA", "Germany"],
            "revenue": ["abc", "100", "200"],  # mixed dtype on purpose
            "order_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )
    p = tmp_path / "wrong_dtype.csv"
    df.to_csv(p, index=False)

    run_dir = Path(run_pipeline(str(p), "Compare revenue by country"))
    assert (run_dir / "metrics_outputs.json").exists()
    assert (run_dir / "final_report.md").exists()


def test_csv_pipeline_fails_on_empty_file(tmp_path: Path, patch_llm, patch_pipeline_cfg: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(Exception):
        run_pipeline(str(p), "Any question")


def test_csv_pipeline_fails_on_malformed_csv(tmp_path: Path, patch_llm, patch_pipeline_cfg: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text('a,b\n1,"unterminated\n2,ok\n', encoding="utf-8")
    with pytest.raises(Exception):
        run_pipeline(str(p), "Any question")


def test_csv_pipeline_handles_unexpected_delimiter(tmp_path: Path, patch_llm, patch_pipeline_cfg: Path) -> None:
    # Delimiter mismatch is common in real uploads; pipeline should still produce outputs.
    p = tmp_path / "semicolon.csv"
    p.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")

    run_dir = Path(run_pipeline(str(p), "Summarize this dataset"))
    assert (run_dir / "final_report.md").exists()
    assert (run_dir / "data_profile.json").exists()


def test_csv_pipeline_fails_on_unsupported_encoding(tmp_path: Path, patch_llm, patch_pipeline_cfg: Path) -> None:
    p = tmp_path / "utf16.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-16")
    with pytest.raises(Exception):
        run_pipeline(str(p), "Any question")


def test_csv_pipeline_writes_failed_manifest_when_stage_breaks(
    monkeypatch: pytest.MonkeyPatch,
    sample_csv_path: Path,
    patch_llm,
    patch_pipeline_cfg: Path,
) -> None:
    import ai_data_analyst_agents.pipelines.run_csv_pipeline as csv_pipe

    class _BrokenMetricsAgent:
        def run(self, ctx):  # noqa: ANN001
            raise RuntimeError("forced-metrics-failure")

    monkeypatch.setattr(csv_pipe, "MetricsAgent", _BrokenMetricsAgent)

    with pytest.raises(RuntimeError, match="forced-metrics-failure"):
        run_pipeline(str(sample_csv_path), "Why did India have less revenue?")

    run_dir = latest_run_dir(patch_pipeline_cfg)
    manifest = read_json(run_dir / "run_manifest.json")
    assert manifest["status"] == "failed"
    assert (run_dir / "agent_messages.json").exists()


def test_csv_pipeline_emits_artifact_callbacks(
    sample_csv_path: Path,
    patch_llm,
    patch_pipeline_cfg: Path,
) -> None:
    seen: list[str] = []

    def _artifact_cb(run_dir: Path, artifact_path: Path) -> None:
        try:
            rel = artifact_path.relative_to(run_dir).as_posix()
        except Exception:
            rel = artifact_path.as_posix()
        seen.append(rel)

    run_dir = Path(
        run_pipeline(
            str(sample_csv_path),
            "Why did India have less revenue than other countries?",
            artifact_callback=_artifact_cb,
        )
    )
    assert run_dir.exists()
    assert "final_report.md" in seen
    assert "run_manifest.json" in seen


def test_csv_pipeline_generates_statistical_artifacts_for_ab_question(
    ab_test_csv_path: Path,
    patch_llm,
    patch_pipeline_cfg: Path,
) -> None:
    run_dir = Path(run_pipeline(str(ab_test_csv_path), "Did treatment improve conversion versus control?"))
    plan = read_json(run_dir / "analysis_tasks.json")
    assert any(t["type"] == "ab_test" for t in plan["tasks"])

    metrics = read_json(run_dir / "metrics_outputs.json")
    stat_artifacts = [
        item["artifact"]
        for item in metrics.get("computed", [])
        if str(item.get("artifact", "")).startswith("statistics/") and "two_proportion_z_test" in str(item.get("artifact", ""))
    ]
    report = (run_dir / "final_report.md").read_text(encoding="utf-8")

    assert stat_artifacts
    assert (run_dir / stat_artifacts[0]).exists()
    assert "### Statistical Questions Evaluated" in report
    assert "### Confidence Intervals" in report
