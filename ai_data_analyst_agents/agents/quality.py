from __future__ import annotations

from typing import Any
import pandas as pd


def run_quality(cfg, df: pd.DataFrame, store, logger) -> dict[str, Any]:
    n_rows = max(int(df.shape[0]), 1)
    missing_by_col = {
        c: float(df[c].isna().mean())
        for c in df.columns
    }
    duplicate_rate = float(df.duplicated().mean()) if len(df) else 0.0

    warnings = []
    for c, frac in missing_by_col.items():
        if frac >= cfg.qa.missingness_warn_threshold:
            warnings.append(f"Column '{c}' has high missingness: {frac:.1%}")

    if duplicate_rate >= cfg.qa.duplicate_warn_threshold:
        warnings.append(f"High duplicate rate: {duplicate_rate:.1%}")

    report: dict[str, Any] = {
        "missingness": missing_by_col,
        "duplicate_rate": duplicate_rate,
        "warnings": warnings,
    }

    store.write_json("quality_report.json", report)
    store.write_text("quality_warnings.md", "\n".join([f"- {w}" for w in warnings]) or "- None")
    logger.info("Wrote quality_report.json")
    return report