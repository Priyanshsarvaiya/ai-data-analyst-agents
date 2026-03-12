from __future__ import annotations

import pandas as pd

from ai_data_analyst_agents.statistics.ab_testing import run_ab_test
from ai_data_analyst_agents.statistics.confidence_intervals import (
    difference_in_means_ci,
    difference_in_proportions_ci,
    single_mean_ci,
)
from ai_data_analyst_agents.statistics.models import ABTestRequest, HypothesisTestRequest, RegressionRequest
from ai_data_analyst_agents.statistics.regression import run_ols_regression
from ai_data_analyst_agents.statistics.selector import run_hypothesis_test, select_hypothesis_method


def test_selector_prefers_two_proportion_for_binary_experiment(ab_test_df: pd.DataFrame) -> None:
    selection = select_hypothesis_method(
        ab_test_df,
        HypothesisTestRequest(group_col="variant", metric="conversion", group_a="treatment", group_b="control"),
    )
    assert selection.method == "two_proportion_z_test"


def test_selector_prefers_welch_for_independent_numeric_groups(ab_test_df: pd.DataFrame) -> None:
    selection = select_hypothesis_method(
        ab_test_df,
        HypothesisTestRequest(group_col="variant", metric="revenue", group_a="treatment", group_b="control"),
    )
    assert selection.method == "welch_t_test"


def test_selector_falls_back_to_mann_whitney_for_small_non_normal_samples() -> None:
    df = pd.DataFrame(
        {
            "segment": ["A"] * 8 + ["B"] * 8,
            "metric": [1, 1, 1, 1, 2, 2, 20, 25, 1, 1, 1, 1, 1, 1, 2, 3],
        }
    )
    selection = select_hypothesis_method(
        df,
        HypothesisTestRequest(group_col="segment", metric="metric", group_a="A", group_b="B"),
    )
    assert selection.method == "mann_whitney_u"
    assert selection.fallback_used is True


def test_chi_square_output_includes_effect_size() -> None:
    df = pd.DataFrame(
        {
            "country": ["US", "US", "US", "IN", "IN", "IN", "IN", "DE", "DE", "DE"],
            "device": ["mobile", "desktop", "mobile", "mobile", "mobile", "desktop", "desktop", "desktop", "mobile", "desktop"],
        }
    )
    _, result = run_hypothesis_test(
        df,
        HypothesisTestRequest(group_col="country", metric="device"),
        analysis_id="T_STAT_CHI",
    )
    assert result.method == "chi_square_independence"
    assert result.effect_sizes[0].name == "cramers_v"
    assert "contingency_table" in result.metrics


def test_confidence_interval_helpers_return_ordered_bounds() -> None:
    mean_ci = single_mean_ci(pd.Series([1, 2, 3, 4, 5]), confidence_level=0.95, parameter="mean_metric")
    diff_ci = difference_in_means_ci(pd.Series([5, 6, 7]), pd.Series([1, 2, 3]), confidence_level=0.95)
    prop_ci = difference_in_proportions_ci(35, 100, 22, 100, confidence_level=0.95)

    assert mean_ci.lower_bound <= mean_ci.point_estimate <= mean_ci.upper_bound
    assert diff_ci.lower_bound <= diff_ci.point_estimate <= diff_ci.upper_bound
    assert prop_ci.lower_bound <= prop_ci.point_estimate <= prop_ci.upper_bound


def test_ab_workflow_returns_lift_ci_and_business_readout(ab_test_df: pd.DataFrame) -> None:
    result = run_ab_test(
        ab_test_df,
        ABTestRequest(
            group_col="variant",
            control="control",
            treatment="treatment",
            metric="conversion",
            metric_type="binary",
        ),
        analysis_id="T_AB",
    )
    effect_names = {eff.name for eff in result.effect_sizes}
    assert result.analysis_type == "ab_test"
    assert result.method == "two_proportion_z_test"
    assert "relative_lift" in effect_names
    assert any("conversion_rate" in ci.parameter for ci in result.confidence_intervals)
    assert "Treatment=treatment, control=control" in result.plain_language


def test_ols_regression_returns_coefficients_and_diagnostics(regression_df: pd.DataFrame) -> None:
    result, coeffs, diagnostics = run_ols_regression(
        regression_df,
        RegressionRequest(target="revenue", predictors=["marketing_spend", "sessions", "discount_pct"]),
        analysis_id="T_REG",
    )
    assert result.analysis_type == "regression"
    assert result.method == "ols"
    assert result.metrics["r_squared"] > 0.95
    assert {"term", "coefficient", "p_value", "standardized_beta"}.issubset(coeffs.columns)
    assert "vif" in diagnostics
    assert any(ci.parameter.startswith("coefficient:") for ci in result.confidence_intervals)


def test_regression_marks_rank_deficient_design_as_not_reliable(regression_df: pd.DataFrame) -> None:
    df = regression_df.copy()
    df["marketing_spend_copy"] = df["marketing_spend"]
    result, coeffs, diagnostics = run_ols_regression(
        df,
        RegressionRequest(target="revenue", predictors=["marketing_spend", "marketing_spend_copy"]),
        analysis_id="T_REG_BAD",
    )
    assert result.status == "not_reliable"
    assert result.decision == "not_reliable"
    assert coeffs.empty
    assert diagnostics["encoded_columns"] == ["marketing_spend", "marketing_spend_copy"]
