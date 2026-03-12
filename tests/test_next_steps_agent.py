from __future__ import annotations

import pandas as pd

from ai_data_analyst_agents.agents.metrics import MetricsAgent
from ai_data_analyst_agents.agents.next_steps import NextStepsAgent
from ai_data_analyst_agents.agents.planner import PlannerAgent
from ai_data_analyst_agents.core.artifacts import ArtifactStore
from ai_data_analyst_agents.core.evidence import EvidenceStore
from ai_data_analyst_agents.core.logging import setup_logging
from ai_data_analyst_agents.core.memory import SharedMemory
from ai_data_analyst_agents.core.settings import load_app_cfg
from ai_data_analyst_agents.tools.pandas_tools import detect_probable_datetime_columns, infer_column_profiles


def test_next_steps_agent_expands_and_executes_followup_tasks(tmp_path, sample_df: pd.DataFrame, patch_llm) -> None:
    cfg = load_app_cfg()
    store = ArtifactStore.create(tmp_path / "artifacts")
    logger = setup_logging("INFO", store.path("logs.txt"))
    memory = SharedMemory()
    evidence = EvidenceStore()

    profile = {
        "columns": sample_df.columns.tolist(),
        "dtypes": {c: str(sample_df[c].dtype) for c in sample_df.columns},
        "datetime_candidates": detect_probable_datetime_columns(sample_df),
        "column_profiles": infer_column_profiles(sample_df),
    }
    memory.set("result.profiling", profile)

    ctx = {
        "cfg": cfg,
        "store": store,
        "logger": logger,
        "df": sample_df,
        "business_question": "Why did India have less revenue than other countries?",
        "memory": memory,
        "evidence": evidence,
        "source": {"type": "csv"},
    }

    initial_plan = PlannerAgent().run(ctx)
    initial_task_count = len(initial_plan["tasks"])
    memory.set("result.planner", initial_plan)
    initial_metrics = MetricsAgent().run(ctx)
    initial_computed = len(initial_metrics["computed"])
    memory.set("result.metrics", initial_metrics)

    out = NextStepsAgent().run(ctx)
    merged_metrics = ctx["memory"].get("result.metrics")
    merged_plan = ctx["memory"].get("result.planner")

    assert out["tasks"]
    assert store.path("next_steps_plan.json").exists()
    assert store.path("next_steps_metrics_outputs.json").exists()
    assert len(merged_plan["tasks"]) > initial_task_count
    assert len(merged_metrics["computed"]) >= initial_computed
    assert merged_metrics.get("followup", {}).get("tasks_added", 0) == len(out["tasks"])
