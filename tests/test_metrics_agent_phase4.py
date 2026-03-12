from __future__ import annotations

import pandas as pd

from ai_data_analyst_agents.agents.metrics import MetricsAgent
from ai_data_analyst_agents.core.artifacts import ArtifactStore
from ai_data_analyst_agents.core.evidence import EvidenceStore
from ai_data_analyst_agents.core.logging import setup_logging
from ai_data_analyst_agents.core.memory import SharedMemory
from ai_data_analyst_agents.core.settings import load_app_cfg


def test_metrics_agent_executes_phase4_statistics_tasks(ab_test_df: pd.DataFrame, regression_df: pd.DataFrame, tmp_path) -> None:
    df = ab_test_df.merge(
        regression_df.drop(columns=["revenue"]).assign(user_id=[f"U{i:04d}" for i in range(1, len(regression_df) + 1)]),
        on="user_id",
        how="left",
    )
    df["marketing_spend"] = df["marketing_spend"].fillna(df["revenue"] * 8)
    df["sessions"] = df["sessions"].fillna(df["session_duration"] * 4)
    df["discount_pct"] = df["discount_pct"].fillna(0.03)

    cfg = load_app_cfg()
    store = ArtifactStore.create(tmp_path / "artifacts")
    logger = setup_logging("INFO", store.path("logs.txt"))
    memory = SharedMemory()
    evidence = EvidenceStore()

    memory.set(
        "result.planner",
        {
            "tasks": [
                {
                    "id": "T1",
                    "type": "statistical_test",
                    "params": {"group_col": "variant", "metric": "revenue", "group_a": "treatment", "group_b": "control"},
                },
                {
                    "id": "T2",
                    "type": "ab_test",
                    "params": {
                        "group_col": "variant",
                        "control": "control",
                        "treatment": "treatment",
                        "metric": "conversion",
                        "metric_type": "binary",
                    },
                },
                {
                    "id": "T3",
                    "type": "ols_regression",
                    "params": {"target": "revenue", "predictors": ["marketing_spend", "sessions", "discount_pct"]},
                },
            ]
        },
    )

    ctx = {
        "cfg": cfg,
        "store": store,
        "logger": logger,
        "df": df,
        "memory": memory,
        "evidence": evidence,
        "business_question": "Did treatment improve conversion and what predicts revenue?",
    }

    out = MetricsAgent().run(ctx)
    assert out["failed"] == []
    assert len(out["computed"]) == 3
    artifacts = [item["artifact"] for item in out["computed"]]
    assert all(artifact.startswith("statistics/") for artifact in artifacts)
    assert store.path("statistics/T1_welch_t_test/summary.json").exists()
    assert store.path("statistics/T2_two_proportion_z_test/summary.json").exists()
    assert store.path("statistics/T3_ols/coefficients.csv").exists()
    assert len(evidence.all()) == 3
