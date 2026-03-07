from __future__ import annotations

from pathlib import Path

import pytest

from ai_data_analyst_agents.core.sql_source import SQLDataSource, build_groupby_query, compute_join_profile
from ai_data_analyst_agents.pipelines.run_sql_pipeline import run_pipeline
from tests.helpers import latest_run_dir, read_json


def test_build_groupby_query_raises_when_no_join_path(sqlite_orders_db) -> None:
    src = SQLDataSource(db_url=f"sqlite:///{sqlite_orders_db}")
    disconnected_schema = {
        "tables": [
            {"name": "orders", "columns": [{"name": "revenue"}, {"name": "customer_id"}]},
            {"name": "customers", "columns": [{"name": "country"}, {"name": "customer_id"}]},
        ],
        "relationships": [],
    }
    with pytest.raises(ValueError, match="No join path"):
        build_groupby_query(
            engine=src.engine,
            schema=disconnected_schema,
            metric_col="revenue",
            group_cols=["country"],
            preferred_fact_table="orders",
        )


def test_compute_join_profile_returns_no_path_status(sqlite_orders_db) -> None:
    src = SQLDataSource(db_url=f"sqlite:///{sqlite_orders_db}")
    disconnected_schema = {
        "tables": [{"name": "orders", "columns": [{"name": "revenue"}]}, {"name": "customers", "columns": [{"name": "country"}]}],
        "relationships": [],
    }
    out = compute_join_profile(
        engine=src.engine,
        schema=disconnected_schema,
        fact_table="orders",
        dimension_table="customers",
    )
    assert out["status"] == "no_path"
    assert out["join_path"] == []


def test_sql_datasource_execute_query_empty_raises(sqlite_orders_db) -> None:
    src = SQLDataSource(db_url=f"sqlite:///{sqlite_orders_db}")
    with pytest.raises(ValueError, match="Empty SQL query"):
        src.execute_query("")


def test_sql_datasource_load_missing_table_raises(sqlite_orders_db) -> None:
    src = SQLDataSource(db_url=f"sqlite:///{sqlite_orders_db}")
    with pytest.raises(Exception):
        src.load_table("does_not_exist")


def test_sql_pipeline_writes_failed_manifest_when_stage_breaks(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_star_db,
    patch_llm,
    patch_pipeline_cfg: Path,
) -> None:
    import ai_data_analyst_agents.pipelines.run_sql_pipeline as sql_pipe

    class _BrokenMetricsAgent:
        def run(self, ctx):  # noqa: ANN001
            raise RuntimeError("forced-sql-metrics-failure")

    monkeypatch.setattr(sql_pipe, "MetricsAgent", _BrokenMetricsAgent)

    with pytest.raises(RuntimeError, match="forced-sql-metrics-failure"):
        run_pipeline(
            db_url=f"sqlite:///{sqlite_star_db}",
            business_question="Why is India lower?",
            base_table="orders",
        )

    run_dir = latest_run_dir(patch_pipeline_cfg)
    manifest = read_json(run_dir / "run_manifest.json")
    assert manifest["status"] == "failed"
    assert (run_dir / "agent_messages.json").exists()
