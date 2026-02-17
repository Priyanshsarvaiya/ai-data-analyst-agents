from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def missingness_report(df: pd.DataFrame) -> Dict[str, float]:
    if len(df) == 0:
        return {c: 0.0 for c in df.columns}
    return {c: float(df[c].isna().mean()) for c in df.columns}


def duplicate_rate(df: pd.DataFrame) -> float:
    return float(df.duplicated().mean()) if len(df) else 0.0


def outlier_report_zscore(df: pd.DataFrame, z_thresh: float = 4.0) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    out: Dict[str, Any] = {"z_thresh": z_thresh, "columns": {}}

    if not numeric_cols or len(df) == 0:
        return out

    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 20:
            continue

        mean = float(s.mean())
        std = float(s.std(ddof=0)) if float(s.std(ddof=0)) != 0.0 else 0.0
        if std == 0.0:
            continue

        z = (s - mean) / std
        n_out = int((np.abs(z) >= z_thresh).sum())
        out["columns"][col] = {
            "n_outliers": n_out,
            "outlier_frac_of_nonnull": float(n_out / max(len(s), 1)),
        }

    return out


def range_sanity_checks(df: pd.DataFrame) -> List[str]:
    warnings: List[str] = []

    # generic safe heuristic example
    for col in df.columns:
        lc = col.lower()
        if "age" in lc and pd.api.types.is_numeric_dtype(df[col]):
            if (df[col].dropna() < 0).any():
                warnings.append(f"Column '{col}' looks like age but has negative values.")

    return warnings


def build_quality_report(
    df: pd.DataFrame,
    *,
    missingness_warn_threshold: float = 0.2,
    duplicate_warn_threshold: float = 0.01,
    outlier_z_threshold: float = 4.0,
) -> Dict[str, Any]:
    miss = missingness_report(df)
    dup = duplicate_rate(df)
    outliers = outlier_report_zscore(df, z_thresh=outlier_z_threshold)
    sanity = range_sanity_checks(df)

    warnings: List[str] = []

    for c, frac in miss.items():
        if frac >= missingness_warn_threshold:
            warnings.append(f"High missingness: '{c}' = {frac:.1%}")

    if dup >= duplicate_warn_threshold:
        warnings.append(f"High duplicate rate: {dup:.1%}")

    for c, info in outliers.get("columns", {}).items():
        if info.get("outlier_frac_of_nonnull", 0.0) >= 0.01:
            warnings.append(
                f"Potential outliers: '{c}' has {info['n_outliers']} points with |z| >= {outlier_z_threshold}"
            )

    warnings.extend(sanity)

    return {
        "missingness": miss,
        "duplicate_rate": dup,
        "outliers_zscore": outliers,
        "warnings": warnings,
    }