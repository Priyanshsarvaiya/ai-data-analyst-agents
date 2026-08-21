from __future__ import annotations

from pathlib import Path

from ai_data_analyst_agents.core.contracts import (
    validate_analysis_plan_contract,
    validate_analysis_tasks_contract,
    validate_metrics_outputs_contract,
    validate_report_metadata_contract,
    validate_run_scorecard_contract,
)
from ai_data_analyst_agents.pipelines.run_csv_pipeline import run_pipeline
from tests.helpers import read_json


def test_pipeline_outputs_validate_against_contracts(
    sample_csv_path: Path,
    patch_llm,
    patch_pipeline_cfg: Path,
) -> None:
    run_dir = Path(run_pipeline(str(sample_csv_path), "Why did India have less revenue than other countries?"))

    analysis_plan = read_json(run_dir / "analysis_plan.json")
    analysis_tasks = read_json(run_dir / "analysis_tasks.json")
    metrics_outputs = read_json(run_dir / "metrics_outputs.json")
    report_metadata = read_json(run_dir / "report_metadata.json")
    scorecard = read_json(run_dir / "run_scorecard.json")

    validate_analysis_plan_contract(analysis_plan)
    validate_analysis_tasks_contract(analysis_tasks)
    validate_metrics_outputs_contract(metrics_outputs)
    validate_report_metadata_contract(report_metadata)
    validate_run_scorecard_contract(scorecard)

    assert scorecard["final_quality_status"] in {"pass", "fail"}
