from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest

from ai_data_analyst_agents.agents.quality import QualityAgent
from ai_data_analyst_agents.agents.reporting import ReportingAgent


def _prepare_reporting_ctx(ctx: Dict[str, Any], question: str, with_metrics: bool = True) -> Dict[str, Any]:
    store = ctx["store"]
    memory = ctx["memory"]
    evidence = ctx["evidence"]
    df = ctx["df"]

    profile = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "datetime_candidates": ["order_date"] if "order_date" in df.columns else [],
        "column_profiles": [],
    }
    qa = {
        "missingness": {c: float(df[c].isna().mean()) for c in df.columns},
        "duplicate_rate": float(df.duplicated().mean()) if len(df) else 0.0,
        "warnings": [],
    }
    store.write_json("data_profile.json", profile)
    store.write_json("quality_report.json", qa)
    ev_rows = evidence.add("metric", "data_profile.json", "rows", pointer="n_rows")
    ev_cols = evidence.add("metric", "data_profile.json", "cols", pointer="n_cols")
    ev_miss = evidence.add("json", "quality_report.json", "missingness", pointer="missingness")
    ev_dup = evidence.add("metric", "quality_report.json", "duplicate", pointer="duplicate_rate")

    memory.set("result.intake", {"business_question": question})
    memory.set("result.profiling", profile)
    memory.set("result.quality", qa)
    memory.set("result.eda", {"charts": [], "question_aware_charts": []})
    memory.set("result.planner", {"tasks": []})

    metrics = {"computed": [], "failed": [], "skipped": []}
    if with_metrics:
        artifact = "T1_groupby_country_revenue_sum.json"
        store.write_json(artifact, {"Germany": 200.0, "India": 150.0, "USA": 180.0})
        ev = evidence.add("json", artifact, "sum(revenue) by country", pointer=None)
        metrics["computed"].append({"task_id": "T1", "artifact": artifact, "evidence_id": ev.id})
    memory.set("result.metrics", metrics)

    # reference evidence vars to prevent accidental optimization/lint stripping in future edits
    _ = (ev_rows, ev_cols, ev_miss, ev_dup)
    return ctx


def test_quality_agent_writes_outputs_and_evidence(agent_ctx_factory, tmp_path, sample_df_with_duplicate: pd.DataFrame) -> None:
    df = sample_df_with_duplicate.copy()
    df.loc[:10, "country"] = None
    df.loc[len(df) - 1, "quantity"] = 9999
    ctx = agent_ctx_factory(tmp_path, df=df, include_cfg=True)

    out = QualityAgent().run(ctx)

    assert "missingness" in out
    assert "duplicate_rate" in out
    assert "warnings" in out
    assert ctx["store"].path("quality_report.json").exists()
    assert ctx["store"].path("quality_warnings.md").exists()
    evs = list(ctx["evidence"].all().values())
    assert any(ev.artifact_path == "quality_report.json" and ev.pointer == "duplicate_rate" for ev in evs)
    assert any(ev.artifact_path == "quality_report.json" and ev.pointer == "missingness" for ev in evs)


def test_quality_agent_handles_empty_dataframe(agent_ctx_factory, tmp_path) -> None:
    df = pd.DataFrame(columns=["a", "b"])
    ctx = agent_ctx_factory(tmp_path, df=df, include_cfg=True)
    out = QualityAgent().run(ctx)
    assert out["duplicate_rate"] == 0.0
    assert out["missingness"] == {"a": 0.0, "b": 0.0}


def test_reporting_agent_deterministic_fallback_with_metrics(
    agent_ctx_factory,
    tmp_path,
    sample_df: pd.DataFrame,
    patch_llm,
) -> None:
    question = "Why did India have less revenue than other countries?"
    ctx = agent_ctx_factory(tmp_path, df=sample_df, question=question, include_cfg=True)
    _prepare_reporting_ctx(ctx, question, with_metrics=True)

    report = ReportingAgent().run(ctx)
    assert "## 1) Executive Summary" in report
    assert "## 8) Artifacts Index" in report
    assert "India" in report
    assert "[[EV:" in report
    assert ctx["store"].path("final_report.md").exists()


def test_reporting_agent_fallback_when_llm_raises(
    agent_ctx_factory,
    tmp_path,
    sample_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_data_analyst_agents.agents.reporting as reporting_mod

    class _ExplodingClient:
        def __init__(self, timeout_s: int = 60) -> None:  # noqa: ARG002
            pass

        def chat(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
            raise RuntimeError("llm exploded")

    monkeypatch.setattr(reporting_mod, "OpenRouterClient", _ExplodingClient)
    question = "Why did India have less revenue than other countries?"
    ctx = agent_ctx_factory(tmp_path, df=sample_df, question=question, include_cfg=True)
    _prepare_reporting_ctx(ctx, question, with_metrics=True)

    report = ReportingAgent().run(ctx)
    assert "Data Analysis Report" in report
    assert "India" in report
    assert "[[EV:" in report


def test_reporting_agent_works_without_computed_metrics(
    agent_ctx_factory,
    tmp_path,
    sample_df: pd.DataFrame,
    patch_llm,
) -> None:
    question = "Summarize dataset quality"
    ctx = agent_ctx_factory(tmp_path, df=sample_df, question=question, include_cfg=True)
    _prepare_reporting_ctx(ctx, question, with_metrics=False)

    report = ReportingAgent().run(ctx)
    assert "## 5) Analysis Outputs" in report
    assert "Not computed in artifacts." in report or "Failed tasks:" in report


def test_reporting_agent_redacts_raw_sql_rows_in_llm_context(
    agent_ctx_factory,
    tmp_path,
    sample_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_data_analyst_agents.agents.reporting as reporting_mod

    captured: dict[str, Any] = {}

    class _CaptureClient:
        def __init__(self, timeout_s: int = 60) -> None:  # noqa: ARG002
            pass

        def chat(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
            captured["messages"] = kwargs.get("messages", [])
            return ""

    monkeypatch.setattr(reporting_mod, "OpenRouterClient", _CaptureClient)
    question = "Summarize SQL results"
    ctx = agent_ctx_factory(tmp_path, df=sample_df, question=question, include_cfg=True)
    _prepare_reporting_ctx(ctx, question, with_metrics=False)

    artifact = "T_sql_rows.json"
    ctx["store"].write_json(
        artifact,
        {
            "columns": ["customer_email"],
            "rows": [{"customer_email": "very-secret-value"}],
            "n_rows": 1,
        },
    )
    ev = ctx["evidence"].add("json", artifact, "sql rows", pointer=None)
    ctx["memory"].set(
        "result.metrics",
        {"computed": [{"task_id": "T1", "artifact": artifact, "evidence_id": ev.id}], "failed": [], "skipped": []},
    )

    ReportingAgent().run(ctx)
    msg = str(captured["messages"])
    assert "very-secret-value" not in msg
    assert "REDACTED(1 rows)" in msg
