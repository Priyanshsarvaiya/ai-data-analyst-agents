from __future__ import annotations

from typing import Any
import pandas as pd
import matplotlib.pyplot as plt


def run_eda(cfg, df: pd.DataFrame, store, logger) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    summary = {
        "numeric_columns": numeric_cols,
        "describe": {},
        "charts": [],
    }

    if numeric_cols:
        desc = df[numeric_cols].describe().to_dict()
        summary["describe"] = desc

        # Basic hist for first numeric col
        col = numeric_cols[0]
        plt.figure()
        df[col].dropna().hist()
        out = store.run_dir / "charts" / f"hist_{col}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        summary["charts"].append(str(out.name))

    store.write_json("eda_summary.json", summary)
    logger.info("Wrote eda_summary.json")
    return summary