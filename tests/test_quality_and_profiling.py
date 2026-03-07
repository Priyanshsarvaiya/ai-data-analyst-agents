from __future__ import annotations

import pandas as pd

from ai_data_analyst_agents.tools.pandas_tools import (
    basic_dataset_summary,
    detect_probable_datetime_columns,
    infer_column_profiles,
)
from ai_data_analyst_agents.tools.validation_tools import build_quality_report


def test_profile_summary_and_column_profiles(sample_df: pd.DataFrame) -> None:
    df = sample_df.copy()
    df.loc[:8, "product_category"] = None

    summary = basic_dataset_summary(df)
    assert summary["n_rows"] == len(df)
    assert summary["n_cols"] == df.shape[1]
    assert "revenue" in summary["dtypes"]

    profiles = infer_column_profiles(df)
    by_name = {p["name"]: p for p in profiles}
    assert by_name["revenue"]["numeric_summary"] is not None
    assert by_name["product_category"]["missing_frac"] > 0.2
    assert by_name["customer_id"]["n_unique"] > 1


def test_detect_probable_datetime_columns(sample_df: pd.DataFrame) -> None:
    dt_cols = detect_probable_datetime_columns(sample_df)
    assert "order_date" in dt_cols
    assert "country" not in dt_cols


def test_quality_report_flags_core_issues() -> None:
    df = pd.DataFrame(
        {
            "metric": [1.0] * 29 + [1000.0],
            "country": ["India"] * 15 + ["Germany"] * 15,
            "age": [30] * 29 + [-5],
            "mostly_missing": [None] * 10 + list(range(20)),
        }
    )
    # duplicate one row to trigger duplicate warnings.
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    report = build_quality_report(
        df,
        missingness_warn_threshold=0.2,
        duplicate_warn_threshold=0.01,
        outlier_z_threshold=4.0,
    )

    assert set(report.keys()) == {"missingness", "duplicate_rate", "outliers_zscore", "warnings"}
    assert report["missingness"]["mostly_missing"] > 0.2
    assert report["duplicate_rate"] > 0.01
    assert "metric" in report["outliers_zscore"]["columns"]

    warnings = "\n".join(report["warnings"])
    assert "High missingness" in warnings
    assert "High duplicate rate" in warnings
    assert "Potential outliers" in warnings
    assert "negative values" in warnings
