from __future__ import annotations

from typing import Any
import pandas as pd


def run_profiling(cfg, df: pd.DataFrame, store, logger) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": [],
    }
    for col in df.columns:
        s = df[col]
        profile["columns"].append({
            "name": col,
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)),
            "example_values": [x for x in s.dropna().astype(str).head(5).tolist()],
        })

    store.write_json("data_profile.json", profile)
    logger.info("Wrote data_profile.json")
    return profile