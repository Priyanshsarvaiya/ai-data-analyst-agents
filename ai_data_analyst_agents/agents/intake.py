from __future__ import annotations

from typing import Any
import pandas as pd


def run_intake(cfg, df: pd.DataFrame, question: str, store, logger) -> dict[str, Any]:
    plan = {
        "business_question": question,
        "assumptions": [
            "Dataset contains the columns needed to measure the question.",
            "Time column (if any) is parseable or already clean.",
        ],
        "suggested_slices": [],
        "requested_metrics": [],
    }
    store.write_json("analysis_plan.json", plan)
    logger.info("Wrote analysis_plan.json")
    return plan