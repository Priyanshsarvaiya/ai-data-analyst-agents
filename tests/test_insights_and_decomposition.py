from __future__ import annotations

import json

import pandas as pd

from ai_data_analyst_agents.agents.insights import InsightsAgent
from ai_data_analyst_agents.agents.metrics import MetricsAgent


def test_gap_decomposition_reconciles_observed_gap(tmp_path, agent_ctx_factory) -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "A", "B", "B", "B"],
            "revenue": [50.0, 50.0, 60.0, 60.0, 60.0],
        }
    )
    ctx = agent_ctx_factory(tmp_path, df=df, question="Why is revenue lower in A than B?")
    task = {
        "id": "T1",
        "type": "gap_decomposition",
        "params": {"segment_by": "country", "metric": "revenue", "focus_segment": "A"},
    }
    ctx["memory"].set(
        "result.planner",
        {
            "analysis_type": "diagnostic",
            "tasks": [task],
        },
    )

    out = MetricsAgent().run(ctx)
    assert out["failed"] == []
    payload = json.loads(ctx["store"].path(out["computed"][0]["artifact"]).read_text())
    assert payload["absolute_gap"] == 80.0
    assert payload["effects"]["row_volume_effect"] + payload["effects"]["average_value_effect"] == 80.0
    assert "not evidence of causality" in payload["interpretation_guardrail"]


def test_insights_marks_missing_required_computation_partial(tmp_path, agent_ctx_factory) -> None:
    ctx = agent_ctx_factory(tmp_path, df=pd.DataFrame({"country": ["A"], "revenue": [1.0]}))
    tasks = [
        {"id": "T1", "type": "kpi_template_apply"},
        {"id": "T2", "type": "groupby_agg"},
        {"id": "T3", "type": "segment_analysis"},
        {"id": "T4", "type": "gap_decomposition"},
    ]
    ctx["memory"].set("result.planner", {"analysis_type": "diagnostic", "tasks": tasks})
    ctx["memory"].set(
        "result.metrics",
        {"computed": [{"task_id": "T1"}, {"task_id": "T2"}, {"task_id": "T3"}], "failed": [], "skipped": []},
    )

    out = InsightsAgent().run(ctx)
    assert out["answer_status"] == "partial"
    assert out["missing_capabilities"] == ["gap_decomposition"]
