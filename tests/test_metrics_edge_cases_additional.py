from __future__ import annotations

import pandas as pd

from ai_data_analyst_agents.agents.metrics import MetricsAgent
from tests.helpers import read_json


def test_metrics_unknown_task_type_is_recorded_failed(tmp_path, sample_df: pd.DataFrame, agent_ctx_factory) -> None:
    ctx = agent_ctx_factory(tmp_path, df=sample_df)
    ctx["memory"].set("result.planner", {"tasks": [{"id": "T1", "type": "made_up", "params": {}}]})
    out = MetricsAgent().run(ctx)
    assert len(out["failed"]) == 1
    assert "Unknown task type" in out["failed"][0]["reason"]


def test_metrics_sql_tasks_skipped_without_sql_source(tmp_path, sample_df: pd.DataFrame, agent_ctx_factory) -> None:
    ctx = agent_ctx_factory(tmp_path, df=sample_df)
    ctx["memory"].set(
        "result.planner",
        {
            "tasks": [
                {"id": "T1", "type": "sql_query", "params": {"query": "select 1"}},
                {"id": "T2", "type": "sql_join_profile", "params": {"fact_table": "orders"}},
            ]
        },
    )
    out = MetricsAgent().run(ctx)
    assert len(out["skipped"]) == 2
    assert all("SQL source unavailable" in s["reason"] for s in out["skipped"])


def test_metrics_filter_agg_supports_nested_where_params(tmp_path, sample_df: pd.DataFrame, agent_ctx_factory) -> None:
    ctx = agent_ctx_factory(tmp_path, df=sample_df)
    ctx["memory"].set(
        "result.planner",
        {
            "tasks": [
                {
                    "id": "T1",
                    "type": "filter_agg",
                    "params": {"where": {"country": "India"}, "metric": "revenue", "agg": "sum"},
                }
            ]
        },
    )
    out = MetricsAgent().run(ctx)
    assert out["failed"] == []
    assert len(out["computed"]) == 1
    payload = read_json(ctx["store"].path(out["computed"][0]["artifact"]))
    assert payload["filter"]["country"] == "India"
    assert isinstance(payload["value"], float)


def test_metrics_correlation_with_too_few_rows_is_skipped(tmp_path, agent_ctx_factory) -> None:
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 3.0]})
    ctx = agent_ctx_factory(tmp_path, df=df)
    ctx["memory"].set(
        "result.planner",
        {"tasks": [{"id": "T1", "type": "correlation", "params": {"x": "x", "y": "y"}}]},
    )
    out = MetricsAgent().run(ctx)
    assert len(out["skipped"]) == 1
    assert "Not enough numeric rows" in out["skipped"][0]["reason"]


def test_metrics_groupby_limit_zero_includes_all_groups(tmp_path, sample_df: pd.DataFrame, agent_ctx_factory) -> None:
    ctx = agent_ctx_factory(tmp_path, df=sample_df)
    ctx["memory"].set(
        "result.planner",
        {"tasks": [{"id": "T1", "type": "groupby_agg", "params": {"group_by": "country", "metric": "revenue", "agg": "sum", "limit": 0}}]},
    )
    out = MetricsAgent().run(ctx)
    assert out["failed"] == []
    payload = read_json(ctx["store"].path(out["computed"][0]["artifact"]))
    assert len(payload) == sample_df["country"].nunique()


def test_metrics_skips_invalid_metric_semantics(tmp_path, agent_ctx_factory) -> None:
    df = pd.DataFrame(
        {
            "segment": ["A", "A", "B", "B"],
            "conversion_rate": [0.12, 0.18, 0.25, 0.2],
        }
    )
    ctx = agent_ctx_factory(tmp_path, df=df)
    ctx["memory"].set(
        "result.planner",
        {
            "analysis_type": "segment_comparison",
            "tasks": [
                {
                    "id": "T1",
                    "type": "groupby_agg",
                    "params": {"group_by": "segment", "metric": "conversion_rate", "agg": "sum"},
                }
            ],
        },
    )
    out = MetricsAgent().run(ctx)
    assert out["failed"] == []
    assert out["computed"] == []
    assert len(out["skipped"]) == 1
    assert "does not allow agg" in out["skipped"][0]["reason"]
    assert any(item["status"] == "invalid" for item in out["semantic_validation"])
