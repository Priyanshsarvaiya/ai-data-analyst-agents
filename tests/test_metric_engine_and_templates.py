from __future__ import annotations

import math

import pandas as pd
import pytest

from ai_data_analyst_agents.core.kpi_templates import (
    default_agg_for_metric,
    detect_business_domain,
    is_agg_allowed_for_metric,
    pick_cohort_columns,
    pick_template_dimension,
)
from ai_data_analyst_agents.core.metric_engine import (
    compute_cohort_retention,
    compute_metric_definition,
    compute_segment_profile,
    compute_template_kpis,
)


def test_business_domain_and_template_helpers() -> None:
    schema_cols = ["account_id", "plan", "mrr", "signup_date"]
    assert detect_business_domain("Show mrr and churn by plan", schema_cols) == "saas"
    assert pick_template_dimension("saas", schema_cols) == "plan"
    assert pick_cohort_columns("saas", schema_cols) == ("account_id", "signup_date")


def test_metric_semantics_helpers() -> None:
    assert default_agg_for_metric("revenue") == "sum"
    assert default_agg_for_metric("order_id") == "count"
    assert is_agg_allowed_for_metric("conversion_rate", "mean")
    assert not is_agg_allowed_for_metric("conversion_rate", "sum")


def test_compute_template_kpis_ecommerce(sample_df: pd.DataFrame) -> None:
    out = compute_template_kpis(sample_df, "ecommerce")
    assert out["domain"] == "ecommerce"
    assert out["kpis"]["total_revenue"] == pytest.approx(float(sample_df["revenue"].sum()))
    assert out["kpis"]["order_count"] == pytest.approx(float(sample_df["order_id"].count()))
    assert out["derived_kpis"]["avg_order_value"] == pytest.approx(
        float(sample_df["revenue"].sum() / sample_df["order_id"].count())
    )


def test_compute_metric_definition_supports_agg_and_expression(sample_df: pd.DataFrame) -> None:
    grouped = compute_metric_definition(
        sample_df,
        {"name": "country_revenue", "metric_col": "revenue", "agg": "sum", "group_by": "country"},
    )
    assert grouped["name"] == "country_revenue"
    assert "India" in grouped["values"]

    expr = compute_metric_definition(sample_df, {"name": "rev_per_unit", "expression": "revenue/quantity"})
    assert expr["name"] == "rev_per_unit"
    assert math.isfinite(expr["value"])

    with pytest.raises(ValueError, match="metric_definition requires"):
        compute_metric_definition(sample_df, {"name": "bad"})


def test_segment_profile_and_cohort_retention(sample_df: pd.DataFrame) -> None:
    seg = compute_segment_profile(sample_df, segment_by="country", metric="revenue", agg="sum")
    assert seg["segment_by"] == "country"
    assert seg["total_value"] == pytest.approx(float(sample_df["revenue"].sum()))
    assert sum(r["share_pct"] for r in seg["rows"]) == pytest.approx(1.0)

    cohort = compute_cohort_retention(
        sample_df[["customer_id", "order_date"]], entity_col="customer_id", date_col="order_date", freq="M"
    )
    assert cohort["rows"], "Cohort rows should not be empty for valid data."
    assert cohort["matrix"], "Cohort matrix should not be empty for valid data."

    with pytest.raises(ValueError, match="entity_col not found"):
        compute_cohort_retention(sample_df, entity_col="unknown", date_col="order_date")
