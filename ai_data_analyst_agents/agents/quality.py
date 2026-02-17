from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.tools.validation_tools import build_quality_report

class QualityAgent(Agent):
    name = "quality"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        cfg = ctx["cfg"]
        store = ctx["store"]
        logger = ctx["logger"]
        df: pd.DataFrame = ctx["df"]
        evidence = ctx["evidence"]

        report = build_quality_report(
            df,
            missingness_warn_threshold=cfg.qa.missingness_warn_threshold,
            duplicate_warn_threshold=cfg.qa.duplicate_warn_threshold,
            outlier_z_threshold=cfg.qa.outlier_z_threshold,
        )

        store.write_json("quality_report.json", report)
        store.write_text(
            "quality_warnings.md",
            "\n".join([f"- {w}" for w in report.get("warnings", [])]) or "- None",
        )

        # Evidence refs
        evidence.add(kind="metric", artifact_path="quality_report.json", pointer="duplicate_rate",
                     summary="Duplicate row rate")
        evidence.add(kind="json", artifact_path="quality_report.json", pointer="missingness",
                     summary="Missingness by column map")

        logger.info("Wrote quality_report.json + evidence refs")
        return report