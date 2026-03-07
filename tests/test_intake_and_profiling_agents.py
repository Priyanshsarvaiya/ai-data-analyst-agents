from __future__ import annotations

import pandas as pd

from ai_data_analyst_agents.agents.intake import IntakeAgent
from ai_data_analyst_agents.agents.profiling import ProfilingAgent
from ai_data_analyst_agents.core.artifacts import ArtifactStore
from ai_data_analyst_agents.core.evidence import EvidenceStore
from ai_data_analyst_agents.core.logging import setup_logging
from ai_data_analyst_agents.core.memory import SharedMemory


def _ctx(tmp_path, df: pd.DataFrame, question: str):
    store = ArtifactStore.create(tmp_path / "artifacts")
    logger = setup_logging("INFO", store.path("logs.txt"))
    return {
        "store": store,
        "logger": logger,
        "df": df,
        "business_question": question,
        "memory": SharedMemory(),
        "evidence": EvidenceStore(),
    }


def test_intake_agent_outputs_structured_plan(tmp_path, sample_df: pd.DataFrame) -> None:
    ctx = _ctx(tmp_path, sample_df, "Why is revenue lower in India?")
    ctx["source"] = {"type": "csv"}

    out = IntakeAgent().run(ctx)

    assert out["business_question"] == "Why is revenue lower in India?"
    assert out["source_type"] == "csv"
    assert out["suggested_domain"] == "ecommerce"
    assert set(["assumptions", "suggested_slices", "requested_metrics"]).issubset(out.keys())
    assert ctx["store"].path("analysis_plan.json").exists()


def test_profiling_agent_includes_sql_schema_summary_and_evidence(tmp_path, sample_df: pd.DataFrame) -> None:
    ctx = _ctx(tmp_path, sample_df, "Any")
    ctx["sql_schema"] = {
        "dialect": "sqlite",
        "tables": [
            {"name": "orders", "n_rows": 30, "columns": [{"name": "order_id"}, {"name": "revenue"}], "primary_key": ["order_id"]},
            {"name": "customers", "n_rows": 10, "columns": [{"name": "customer_id"}, {"name": "country"}], "primary_key": ["customer_id"]},
        ],
        "relationships": [{"from_table": "orders", "to_table": "customers"}],
    }

    out = ProfilingAgent().run(ctx)
    assert out["n_rows"] == len(sample_df)
    assert "datetime_candidates" in out
    assert "column_profiles" in out
    assert "sql_schema" in out
    assert out["sql_schema"]["table_count"] == 2
    assert out["sql_schema"]["relationship_count"] == 1
    assert ctx["store"].path("data_profile.json").exists()

    ev_all = ctx["evidence"].all()
    pointers = {(ev.artifact_path, ev.pointer) for ev in ev_all.values()}
    assert ("data_profile.json", "n_rows") in pointers
    assert ("data_profile.json", "n_cols") in pointers
    assert ("db_schema.json", None) in pointers
