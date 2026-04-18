from __future__ import annotations

import json
from pathlib import Path

from ai_data_analyst_agents.evaluation.harness import (
    compute_quality_kpis,
    load_benchmark_suite,
    run_benchmark_suite,
)


def test_compute_quality_kpis_has_expected_fields() -> None:
    rows = [
        {
            "unsupported_claim": False,
            "planned_stat_task_count": 2,
            "computed_stat_task_count": 1,
            "computed_artifact_count": 5,
            "duplicate_artifact_count": 1,
            "answer_relevance_score": 0.8,
            "route_correct": True,
        },
        {
            "unsupported_claim": True,
            "planned_stat_task_count": 1,
            "computed_stat_task_count": 1,
            "computed_artifact_count": 4,
            "duplicate_artifact_count": 0,
            "answer_relevance_score": 0.6,
            "route_correct": False,
        },
    ]
    kpis = compute_quality_kpis(rows)
    assert set(kpis.keys()) == {
        "unsupported_claim_rate",
        "stat_task_success_rate",
        "duplicate_artifact_rate",
        "answer_relevance_score",
        "route_accuracy",
        "case_count",
    }
    assert kpis["case_count"] == 2.0
    assert 0.0 <= kpis["route_accuracy"] <= 1.0


def test_load_benchmark_suite_reads_bundled_suite() -> None:
    cases = load_benchmark_suite("benchmarks/core_quality_suite.yaml")
    assert cases
    assert any(c.source_type == "csv" for c in cases)


def test_run_benchmark_suite_writes_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_run_dir = tmp_path / "fake_run"
    fake_run_dir.mkdir(parents=True, exist_ok=True)
    (fake_run_dir / "analysis_plan.json").write_text(
        json.dumps({"analysis_type": "descriptive"}),
        encoding="utf-8",
    )
    (fake_run_dir / "analysis_tasks.json").write_text(
        json.dumps({"tasks": [{"id": "T1", "type": "groupby_agg", "params": {}}]}),
        encoding="utf-8",
    )
    (fake_run_dir / "metrics_outputs.json").write_text(
        json.dumps({"computed": [{"task_id": "T1", "artifact": "T1.json"}], "failed": [], "skipped": []}),
        encoding="utf-8",
    )
    (fake_run_dir / "review_log.json").write_text(json.dumps({"status": "pass"}), encoding="utf-8")
    (fake_run_dir / "report_metadata.json").write_text(
        json.dumps({"section_completeness_ok": True, "unsupported_numeric_claim_lines": 0}),
        encoding="utf-8",
    )
    (fake_run_dir / "run_scorecard.json").write_text(
        json.dumps({"final_quality_status": "pass"}),
        encoding="utf-8",
    )

    def _fake_run_csv_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003
        return fake_run_dir

    monkeypatch.setattr("ai_data_analyst_agents.evaluation.harness.run_csv_pipeline", _fake_run_csv_pipeline)

    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: smoke_csv",
                "    source_type: csv",
                "    file_path: ./dummy.csv",
                "    question: Summarize dataset",
                "    expected_route: descriptive",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "dummy.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    summary_path = run_benchmark_suite(suite, output_dir=tmp_path / "eval_out")
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "kpis" in summary
    assert summary["cases"][0]["id"] == "smoke_csv"
