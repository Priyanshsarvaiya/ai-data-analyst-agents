from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import warnings



def basic_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }


def infer_column_profiles(df: pd.DataFrame, max_examples: int = 5) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        missing_frac = float(s.isna().mean()) if len(df) else 0.0
        n_unique = int(s.nunique(dropna=True))
        examples = s.dropna().astype(str).head(max_examples).tolist()

        numeric_summary = None
        if pd.api.types.is_numeric_dtype(s):
            desc = s.dropna().describe()
            numeric_summary = {
                "count": float(desc.get("count", 0.0)),
                "mean": float(desc.get("mean", 0.0)),
                "std": float(desc.get("std", 0.0)),
                "min": float(desc.get("min", 0.0)),
                "p25": float(desc.get("25%", 0.0)),
                "median": float(desc.get("50%", 0.0)),
                "p75": float(desc.get("75%", 0.0)),
                "max": float(desc.get("max", 0.0)),
            }

        profiles.append(
            {
                "name": col,
                "dtype": str(s.dtype),
                "n_missing": n_missing,
                "missing_frac": missing_frac,
                "n_unique": n_unique,
                "example_values": examples,
                "numeric_summary": numeric_summary,
            }
        )
    return profiles




def detect_probable_datetime_columns(df: pd.DataFrame, max_try: int = 2000) -> List[str]:
    """
    Heuristic: parse a sample of values and see if most can be parsed as datetime.
    Pylance-safe: avoids len() on parsed (which it sometimes infers as Timestamp/NaT).
    """
    datetime_cols: List[str] = []
    sample_df = df.head(max_try)

    for col in df.columns:
        s = sample_df[col].dropna()
        if s.empty:
            continue

        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(s.astype(str), errors="coerce", utc=False)

        total = int(s.shape[0])
        if total == 0:
            continue

        valid_count = int(pd.notna(parsed).sum())
        ok_rate = valid_count / total

        if ok_rate >= 0.8 and total >= 10:
            datetime_cols.append(col)

    return datetime_cols