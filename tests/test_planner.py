from __future__ import annotations

import json
import pandas as pd

from ai_data_analyst_agents.agents.planner import (
    COUNT_KEYWORDS,
    PlannerAgent,
    _contains_any,
)
from ai_data_analyst_agents.core.artifacts import ArtifactStore
from ai_data_analyst_agents.core.evidence import EvidenceStore
from ai_data_analyst_agents.core.logging import setup_logging
from ai_data_analyst_agents.core.memory import SharedMemory
from ai_data_analyst_agents.core.settings import load_app_cfg
from ai_data_analyst_agents.core.sql_source import SQLDataSource
from ai_data_analyst_agents.tools.pandas_tools import detect_probable_datetime_columns, infer_column_profiles


def _make_ctx(tmp_path, df: pd.DataFrame, question: str):
    cfg = load_app_cfg()
    store = ArtifactStore.create(tmp_path / "artifacts")
    logger = setup_logging("INFO", store.path("logs.txt"))
    memory = SharedMemory()
    evidence = EvidenceStore()
    return {
        "cfg": cfg,
        "store": store,
        "logger": logger,
        "df": df,
        "business_question": question,
        "memory": memory,
        "evidence": evidence,
    }


def test_contains_any_respects_word_boundaries() -> None:
    assert _contains_any("count orders by country", COUNT_KEYWORDS)
    assert not _contains_any("compare countries by revenue", COUNT_KEYWORDS)


def test_planner_fallback_builds_core_tasks(tmp_path, sample_df: pd.DataFrame, patch_llm) -> None:
    question = "Why did India have less revenue than other countries?"
    ctx = _make_ctx(tmp_path, sample_df, question)
    profile = {
        "columns": sample_df.columns.tolist(),
        "dtypes": {c: str(sample_df[c].dtype) for c in sample_df.columns},
        "datetime_candidates": detect_probable_datetime_columns(sample_df),
        "column_profiles": infer_column_profiles(sample_df),
    }
    ctx["memory"].set("result.profiling", profile)

    plan = PlannerAgent().run(ctx)
    tasks = plan["tasks"]

    assert any(t["type"] == "filter_agg" and t["params"].get("filter_val") == "India" for t in tasks)
    assert any(t["type"] == "groupby_agg" and t["params"].get("group_by") == "country" for t in tasks)
    assert any(t["type"] == "kpi_template_apply" for t in tasks)
    assert any(t["type"] == "segment_analysis" for t in tasks)
    store = ctx["store"]
    assert store.path("analysis_tasks.json").exists()


def test_planner_sql_generates_join_query(tmp_path, sqlite_star_db, patch_llm) -> None:
    source = SQLDataSource(db_url=f"sqlite:///{sqlite_star_db}")
    orders_df = source.load_table("orders")
    schema = source.inspect_schema(include_row_counts=True)

    question = "Why is India revenue lower than other countries?"
    ctx = _make_ctx(tmp_path, orders_df, question)
    profile = {
        "columns": orders_df.columns.tolist(),
        "dtypes": {c: str(orders_df[c].dtype) for c in orders_df.columns},
        "datetime_candidates": detect_probable_datetime_columns(orders_df),
        "column_profiles": infer_column_profiles(orders_df),
    }
    ctx["memory"].set("result.profiling", profile)
    ctx["sql_source"] = source
    ctx["sql_schema"] = schema
    ctx["source"] = {"type": "sql", "analysis_table": "orders"}

    plan = PlannerAgent().run(ctx)
    sql_tasks = [t for t in plan["tasks"] if t["type"] == "sql_query"]

    assert sql_tasks, "Expected at least one sql_query task."
    assert any("JOIN" in t["params"]["query"] and "customers" in t["params"]["query"] for t in sql_tasks)
    assert any(t["type"] == "sql_join_profile" for t in plan["tasks"])


def test_planner_adds_cohort_task_for_retention_question(tmp_path, sample_df: pd.DataFrame, patch_llm) -> None:
    question = "Show customer retention trend over time for ecommerce orders"
    ctx = _make_ctx(tmp_path, sample_df, question)
    profile = {
        "columns": sample_df.columns.tolist(),
        "dtypes": {c: str(sample_df[c].dtype) for c in sample_df.columns},
        "datetime_candidates": detect_probable_datetime_columns(sample_df),
        "column_profiles": infer_column_profiles(sample_df),
    }
    ctx["memory"].set("result.profiling", profile)

    plan = PlannerAgent().run(ctx)
    assert any(t["type"] == "cohort_analysis" for t in plan["tasks"])


def test_planner_drops_unsafe_sql_query_from_llm(
    tmp_path,
    sample_df: pd.DataFrame,
    monkeypatch,
) -> None:
    import ai_data_analyst_agents.agents.planner as planner_mod

    class _UnsafeSQLClient:
        def __init__(self, timeout_s: int = 60) -> None:  # noqa: ARG002
            pass

        def chat(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
            return json.dumps(
                {
                    "tasks": [
                        {
                            "id": "T1",
                            "type": "sql_query",
                            "params": {"query": "DROP TABLE orders", "limit": 1000, "output": "rows"},
                        }
                    ],
                    "notes": "unsafe",
                }
            )

    monkeypatch.setattr(planner_mod, "OpenRouterClient", _UnsafeSQLClient)

    question = "Analyze country performance"
    ctx = _make_ctx(tmp_path, sample_df, question)
    profile = {
        "columns": sample_df.columns.tolist(),
        "dtypes": {c: str(sample_df[c].dtype) for c in sample_df.columns},
        "datetime_candidates": detect_probable_datetime_columns(sample_df),
        "column_profiles": infer_column_profiles(sample_df),
    }
    ctx["memory"].set("result.profiling", profile)

    plan = PlannerAgent().run(ctx)
    assert all(t["type"] != "sql_query" for t in plan["tasks"])


def test_planner_adds_ab_test_for_experiment_questions(tmp_path, ab_test_df: pd.DataFrame, patch_llm) -> None:
    question = "Did treatment improve conversion versus control?"
    ctx = _make_ctx(tmp_path, ab_test_df, question)
    profile = {
        "columns": ab_test_df.columns.tolist(),
        "dtypes": {c: str(ab_test_df[c].dtype) for c in ab_test_df.columns},
        "datetime_candidates": detect_probable_datetime_columns(ab_test_df),
        "column_profiles": infer_column_profiles(ab_test_df),
    }
    ctx["memory"].set("result.profiling", profile)

    plan = PlannerAgent().run(ctx)
    assert any(t["type"] == "ab_test" for t in plan["tasks"])


def test_planner_adds_regression_for_association_questions(tmp_path, regression_df: pd.DataFrame, patch_llm) -> None:
    question = "Which variables are most associated with revenue? Use regression."
    ctx = _make_ctx(tmp_path, regression_df, question)
    profile = {
        "columns": regression_df.columns.tolist(),
        "dtypes": {c: str(regression_df[c].dtype) for c in regression_df.columns},
        "datetime_candidates": detect_probable_datetime_columns(regression_df),
        "column_profiles": infer_column_profiles(regression_df),
    }
    ctx["memory"].set("result.profiling", profile)

    plan = PlannerAgent().run(ctx)
    regression_tasks = [t for t in plan["tasks"] if t["type"] == "ols_regression"]
    assert regression_tasks
    assert regression_tasks[0]["params"]["target"] == "revenue"


def test_planner_does_not_add_generic_stat_test_for_why_customer_question(tmp_path, sample_df: pd.DataFrame, patch_llm) -> None:
    question = "Identify high-value customers and why."
    ctx = _make_ctx(tmp_path, sample_df, question)
    profile = {
        "columns": sample_df.columns.tolist(),
        "dtypes": {c: str(sample_df[c].dtype) for c in sample_df.columns},
        "datetime_candidates": detect_probable_datetime_columns(sample_df),
        "column_profiles": infer_column_profiles(sample_df),
    }
    ctx["memory"].set("result.profiling", profile)

    plan = PlannerAgent().run(ctx)
    assert all(t["type"] != "statistical_test" for t in plan["tasks"])
