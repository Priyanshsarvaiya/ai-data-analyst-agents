from __future__ import annotations

import math
from typing import Any

import pandas as pd
from scipy import stats

from ai_data_analyst_agents.statistics.assumptions import (
    check_binary_values,
    check_equal_variance,
    check_expected_counts,
    check_group_balance,
    check_multiple_comparisons,
    check_normality,
    check_outlier_sensitivity,
    check_sample_size,
    summarize_failures,
    to_numeric_series,
)
from ai_data_analyst_agents.statistics.confidence_intervals import (
    difference_in_means_ci,
    difference_in_proportions_ci,
    single_proportion_ci,
)
from ai_data_analyst_agents.statistics.effect_sizes import cohens_d, cramers_v, odds_ratio, relative_lift
from ai_data_analyst_agents.statistics.models import AssumptionCheck, StatisticalResult


def _decision(p_value: float | None, alpha: float, *, reliable: bool = True) -> str:
    if not reliable:
        return "not_reliable"
    if p_value is None:
        return "not_reliable"
    return "reject_null" if p_value < alpha else "fail_to_reject_null"


def _plain_language(decision: str, *, alpha: float, label: str, p_value: float | None) -> str:
    if decision == "not_reliable":
        return f"The {label} result is not reliable enough to support a defensible claim."
    if p_value is None:
        return f"The {label} result could not be estimated."
    if decision == "reject_null":
        return f"The difference is statistically detectable at alpha={alpha:.2f} (p={p_value:.4f})."
    return f"The evidence is insufficient at alpha={alpha:.2f} (p={p_value:.4f})."


def welch_t_test(
    *,
    analysis_id: str,
    group_a: pd.Series,
    group_b: pd.Series,
    label_a: str,
    label_b: str,
    metric: str,
    alpha: float,
    alternative: str = "two-sided",
) -> StatisticalResult:
    a, a_missing = to_numeric_series(group_a, name=f"{metric}:{label_a}")
    b, b_missing = to_numeric_series(group_b, name=f"{metric}:{label_b}")
    assumptions = [
        a_missing,
        b_missing,
        check_sample_size(a, min_n=20, label=label_a),
        check_sample_size(b, min_n=20, label=label_b),
        check_group_balance(int(a.shape[0]), int(b.shape[0]), label_a=label_a, label_b=label_b),
        check_normality(a, alpha=alpha, label=label_a),
        check_normality(b, alpha=alpha, label=label_b),
        check_equal_variance(a, b, alpha=alpha),
        check_outlier_sensitivity(a, label=label_a),
        check_outlier_sensitivity(b, label=label_b),
        check_multiple_comparisons(1),
    ]
    statistic, p_value = stats.ttest_ind(a.to_numpy(), b.to_numpy(), equal_var=False, alternative=alternative)
    mean_diff = float(a.mean() - b.mean())
    ci = difference_in_means_ci(a, b, confidence_level=1.0 - alpha, parameter=f"mean({label_a}) - mean({label_b})")
    effect = cohens_d(a, b)
    warnings = summarize_failures(assumptions)
    reliable = min(int(a.shape[0]), int(b.shape[0])) >= 2
    decision = _decision(float(p_value), alpha, reliable=reliable)
    interpretation = (
        f"Welch's t-test compared {metric} between {label_a} and {label_b}. "
        f"Observed mean difference={mean_diff:.4f}, statistic={float(statistic):.4f}, p-value={float(p_value):.4f}."
    )
    return StatisticalResult(
        analysis_id=analysis_id,
        analysis_type="hypothesis_test",
        method="welch_t_test",
        method_reason="Independent numeric groups with a conservative unequal-variance default.",
        null_hypothesis=f"Mean {metric} is equal in {label_a} and {label_b}.",
        alternative_hypothesis=f"Mean {metric} differs between {label_a} and {label_b}.",
        sample_sizes={label_a: int(a.shape[0]), label_b: int(b.shape[0])},
        alpha=alpha,
        test_statistic=float(statistic),
        p_value=float(p_value),
        decision=decision,
        interpretation=interpretation,
        plain_language=_plain_language(decision, alpha=alpha, label="Welch t-test", p_value=float(p_value)),
        assumptions=assumptions,
        warnings=warnings,
        confidence_intervals=[ci],
        effect_sizes=[effect],
        limitations=["This test assesses association in observed groups and does not establish causality."],
        metrics={
            "mean_difference": mean_diff,
            f"mean_{label_a}": float(a.mean()),
            f"mean_{label_b}": float(b.mean()),
        },
    )


def paired_t_test(
    *,
    analysis_id: str,
    paired: pd.DataFrame,
    left_col: str,
    right_col: str,
    label_left: str,
    label_right: str,
    metric: str,
    alpha: float,
    alternative: str = "two-sided",
) -> StatisticalResult:
    left, left_missing = to_numeric_series(paired[left_col], name=f"{metric}:{label_left}")
    right, right_missing = to_numeric_series(paired[right_col], name=f"{metric}:{label_right}")
    aligned = pd.concat([left.reset_index(drop=True), right.reset_index(drop=True)], axis=1).dropna()
    diffs = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    assumptions = [
        left_missing,
        right_missing,
        check_sample_size(diffs, min_n=20, label="paired differences"),
        check_normality(diffs, alpha=alpha, label="paired differences"),
        check_outlier_sensitivity(diffs, label="paired differences"),
        check_multiple_comparisons(1),
    ]
    statistic, p_value = stats.ttest_rel(aligned.iloc[:, 0], aligned.iloc[:, 1], alternative=alternative)
    ci = difference_in_means_ci(aligned.iloc[:, 0], aligned.iloc[:, 1], confidence_level=1.0 - alpha, parameter=f"paired mean difference {label_left}-{label_right}")
    effect = cohens_d(aligned.iloc[:, 0], aligned.iloc[:, 1])
    decision = _decision(float(p_value), alpha, reliable=int(aligned.shape[0]) >= 2)
    return StatisticalResult(
        analysis_id=analysis_id,
        analysis_type="hypothesis_test",
        method="paired_t_test",
        method_reason="Explicit pairing was requested, so differences were tested within matched units.",
        null_hypothesis=f"Mean paired difference in {metric} between {label_left} and {label_right} is zero.",
        alternative_hypothesis=f"Mean paired difference in {metric} between {label_left} and {label_right} is not zero.",
        sample_sizes={"pairs": int(aligned.shape[0])},
        alpha=alpha,
        test_statistic=float(statistic),
        p_value=float(p_value),
        decision=decision,
        interpretation=(
            f"Paired t-test on {metric} differences between {label_left} and {label_right}: "
            f"mean difference={float(diffs.mean()):.4f}, statistic={float(statistic):.4f}, p-value={float(p_value):.4f}."
        ),
        plain_language=_plain_language(decision, alpha=alpha, label="paired t-test", p_value=float(p_value)),
        assumptions=assumptions,
        warnings=summarize_failures(assumptions),
        confidence_intervals=[ci],
        effect_sizes=[effect],
        limitations=["Pairing quality depends on the supplied match key and observed data only."],
        metrics={"mean_difference": float(diffs.mean())},
    )


def mann_whitney_u_test(
    *,
    analysis_id: str,
    group_a: pd.Series,
    group_b: pd.Series,
    label_a: str,
    label_b: str,
    metric: str,
    alpha: float,
    alternative: str = "two-sided",
) -> StatisticalResult:
    a, a_missing = to_numeric_series(group_a, name=f"{metric}:{label_a}")
    b, b_missing = to_numeric_series(group_b, name=f"{metric}:{label_b}")
    assumptions = [
        a_missing,
        b_missing,
        check_sample_size(a, min_n=8, label=label_a),
        check_sample_size(b, min_n=8, label=label_b),
        check_group_balance(int(a.shape[0]), int(b.shape[0]), label_a=label_a, label_b=label_b),
        check_outlier_sensitivity(a, label=label_a),
        check_outlier_sensitivity(b, label=label_b),
        check_multiple_comparisons(1),
    ]
    statistic, p_value = stats.mannwhitneyu(a.to_numpy(), b.to_numpy(), alternative=alternative)
    ci = difference_in_means_ci(a, b, confidence_level=1.0 - alpha, parameter=f"mean({label_a}) - mean({label_b})")
    effect = cohens_d(a, b)
    decision = _decision(float(p_value), alpha, reliable=min(int(a.shape[0]), int(b.shape[0])) >= 2)
    return StatisticalResult(
        analysis_id=analysis_id,
        analysis_type="hypothesis_test",
        method="mann_whitney_u",
        method_reason="A nonparametric fallback was used because normality/sample-size checks were weak.",
        null_hypothesis=f"The {metric} distributions for {label_a} and {label_b} are equal.",
        alternative_hypothesis=f"The {metric} distributions for {label_a} and {label_b} differ.",
        sample_sizes={label_a: int(a.shape[0]), label_b: int(b.shape[0])},
        alpha=alpha,
        test_statistic=float(statistic),
        p_value=float(p_value),
        decision=decision,
        interpretation=(
            f"Mann-Whitney U compared {metric} between {label_a} and {label_b}: "
            f"U={float(statistic):.4f}, p-value={float(p_value):.4f}."
        ),
        plain_language=_plain_language(decision, alpha=alpha, label="Mann-Whitney U", p_value=float(p_value)),
        assumptions=assumptions,
        warnings=summarize_failures(assumptions),
        confidence_intervals=[ci],
        effect_sizes=[effect],
        limitations=["The Mann-Whitney result reflects distributional shift, not necessarily a pure mean difference."],
        metrics={
            f"median_{label_a}": float(a.median()),
            f"median_{label_b}": float(b.median()),
            "mean_difference": float(a.mean() - b.mean()),
        },
    )


def chi_square_independence_test(
    *,
    analysis_id: str,
    left: pd.Series,
    right: pd.Series,
    left_name: str,
    right_name: str,
    alpha: float,
) -> StatisticalResult:
    table = pd.crosstab(left.astype(str), right.astype(str))
    stat, p_value, dof, _ = stats.chi2_contingency(table.to_numpy())
    expected_check, expected_df = check_expected_counts(table)
    assumptions = [expected_check, check_multiple_comparisons(1)]
    effect = cramers_v(float(stat), table)
    decision = _decision(float(p_value), alpha, reliable=bool(expected_check.passed or expected_check.passed is None))
    warnings = summarize_failures(assumptions)
    return StatisticalResult(
        analysis_id=analysis_id,
        analysis_type="hypothesis_test",
        method="chi_square_independence",
        method_reason="Both compared fields are categorical, so association was tested via contingency table.",
        null_hypothesis=f"{left_name} and {right_name} are independent.",
        alternative_hypothesis=f"{left_name} and {right_name} are associated.",
        sample_sizes={"n_rows": int(table.to_numpy().sum())},
        alpha=alpha,
        test_statistic=float(stat),
        p_value=float(p_value),
        decision=decision,
        interpretation=(
            f"Chi-square test of independence between {left_name} and {right_name}: "
            f"chi-square={float(stat):.4f}, dof={int(dof)}, p-value={float(p_value):.4f}."
        ),
        plain_language=_plain_language(decision, alpha=alpha, label="chi-square", p_value=float(p_value)),
        assumptions=assumptions,
        warnings=warnings,
        effect_sizes=[effect],
        limitations=["Chi-square detects association structure but not directional or causal effect."],
        metrics={
            "degrees_of_freedom": int(dof),
            "contingency_table": table.to_dict(),
            "expected_counts": expected_df.round(4).to_dict(),
        },
    )


def two_proportion_z_test(
    *,
    analysis_id: str,
    group_a: pd.Series,
    group_b: pd.Series,
    label_a: str,
    label_b: str,
    metric: str,
    success_value: Any,
    alpha: float,
) -> StatisticalResult:
    validity_a = check_binary_values(group_a, success_value=success_value)
    validity_b = check_binary_values(group_b, success_value=success_value)
    a = group_a.dropna().map(lambda x: 1 if x == success_value or x is True else 0).astype(int)
    b = group_b.dropna().map(lambda x: 1 if x == success_value or x is True else 0).astype(int)
    n_a = int(a.shape[0])
    n_b = int(b.shape[0])
    success_a = int(a.sum())
    success_b = int(b.sum())
    p_a = success_a / n_a if n_a else 0.0
    p_b = success_b / n_b if n_b else 0.0
    pooled = (success_a + success_b) / max(1, (n_a + n_b))
    se = math.sqrt(max(1e-12, pooled * (1.0 - pooled) * ((1.0 / max(1, n_a)) + (1.0 / max(1, n_b)))))
    z = (p_a - p_b) / se if se else 0.0
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    assumptions = [
        validity_a,
        validity_b,
        check_sample_size(a, min_n=20, label=label_a),
        check_sample_size(b, min_n=20, label=label_b),
        check_group_balance(n_a, n_b, label_a=label_a, label_b=label_b),
        AssumptionCheck(
            name="normal_approximation",
            passed=min(success_a, n_a - success_a, success_b, n_b - success_b) >= 5,
            detail=(
                f"Success/failure counts are {label_a}: {success_a}/{n_a - success_a}, "
                f"{label_b}: {success_b}/{n_b - success_b}."
            ),
            severity="warn",
        ),
        check_multiple_comparisons(1),
    ]
    ci = difference_in_proportions_ci(success_a, n_a, success_b, n_b, confidence_level=1.0 - alpha, parameter=f"conversion_rate({label_a}) - conversion_rate({label_b})")
    control_ci = single_proportion_ci(success_b, n_b, confidence_level=1.0 - alpha, parameter=f"conversion_rate({label_b})")
    effect_sizes = [
        relative_lift(p_a, p_b),
        odds_ratio(success_a, n_a - success_a, success_b, n_b - success_b),
    ]
    decision = _decision(p_value, alpha, reliable=n_a > 0 and n_b > 0 and validity_a.passed is not False and validity_b.passed is not False)
    warnings = summarize_failures(assumptions)
    return StatisticalResult(
        analysis_id=analysis_id,
        analysis_type="hypothesis_test",
        method="two_proportion_z_test",
        method_reason="Binary metric comparison across two independent groups.",
        null_hypothesis=f"Conversion rate for {metric} is equal in {label_a} and {label_b}.",
        alternative_hypothesis=f"Conversion rate for {metric} differs between {label_a} and {label_b}.",
        sample_sizes={label_a: n_a, label_b: n_b},
        alpha=alpha,
        test_statistic=float(z),
        p_value=p_value,
        decision=decision,
        interpretation=(
            f"Two-proportion z-test on {metric}: {label_a} rate={p_a:.4f}, {label_b} rate={p_b:.4f}, "
            f"z={z:.4f}, p-value={p_value:.4f}."
        ),
        plain_language=_plain_language(decision, alpha=alpha, label="two-proportion z-test", p_value=p_value),
        assumptions=assumptions,
        warnings=warnings,
        confidence_intervals=[ci, control_ci],
        effect_sizes=effect_sizes,
        limitations=["Interpret the result with the confidence interval because statistically detectable changes can still be small in practice."],
        metrics={
            f"rate_{label_a}": p_a,
            f"rate_{label_b}": p_b,
            "absolute_difference": p_a - p_b,
            "successes": {label_a: success_a, label_b: success_b},
        },
    )
