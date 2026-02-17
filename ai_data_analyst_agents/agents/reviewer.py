from __future__ import annotations

from typing import Any


def run_reviewer(cfg, report_md: str, store, logger) -> dict[str, Any]:
    # Phase 1: minimal reviewer that ensures artifact index exists.
    required = ["analysis_plan.json", "data_profile.json", "quality_report.json", "cleaned.csv", "eda_summary.json"]
    missing = [r for r in required if not store.path(r).exists()]

    review = {
        "status": "pass" if not missing else "fail",
        "missing_artifacts": missing,
        "notes": [
            "Phase 1 reviewer checks artifact presence only.",
            "Phase 2 will validate every claim -> evidence mapping."
        ],
    }
    store.write_json("review_log.json", review)
    logger.info("Wrote review_log.json")
    return review