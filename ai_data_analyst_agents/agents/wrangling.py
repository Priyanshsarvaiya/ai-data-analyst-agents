from __future__ import annotations

from typing import Any, Tuple
import pandas as pd


def run_wrangling(cfg, df: pd.DataFrame, store, logger) -> Tuple[pd.DataFrame, dict[str, Any]]:
    clean = df.copy()

    # Minimal Phase 1: drop exact duplicates, keep as-is otherwise
    before = len(clean)
    clean = clean.drop_duplicates()
    after = len(clean)

    feature_log = {
        "steps": [
            {"action": "drop_duplicates", "before_rows": before, "after_rows": after}
        ]
    }

    clean.to_csv(store.path("cleaned.csv"), index=False)
    store.write_json("feature_log.json", feature_log)
    logger.info("Wrote cleaned.csv")
    return clean, feature_log