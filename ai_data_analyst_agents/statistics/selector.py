from __future__ import annotations

from typing import Any

import pandas as pd

from ai_data_analyst_agents.statistics.assumptions import check_normality
from ai_data_analyst_agents.statistics.hypothesis_tests import (
    chi_square_independence_test,
    mann_whitney_u_test,
    paired_t_test,
    two_proportion_z_test,
    welch_t_test,
)
from ai_data_analyst_agents.statistics.models import (
    HypothesisTestRequest,
    StatisticalResult,
    StatisticalSelection,
)


BINARY_TOKENS = {"converted", "conversion", "is_conversion", "converted_flag", "clicked", "purchased", "is_purchase"}


def _unique_groups(series: pd.Series) -> list[Any]:
    return [x for x in series.dropna().astype(str).unique().tolist() if str(x).strip()]


def _is_binary_metric(series: pd.Series, *, metric_name: str, success_value: Any) -> bool:
    clean = series.dropna()
    if clean.empty:
        return False
    unique_vals = set(clean.unique().tolist())
    if unique_vals.issubset({0, 1, True, False, success_value}):
        return True
    return metric_name.lower() in BINARY_TOKENS


def prepare_two_groups(df: pd.DataFrame, request: HypothesisTestRequest) -> tuple[pd.Series, pd.Series, str, str]:
    dff = df.dropna(subset=[request.group_col, request.metric]).copy()
    dff[request.group_col] = dff[request.group_col].astype(str)
    groups = _unique_groups(dff[request.group_col])
    if not groups:
        raise ValueError(f"No non-null groups found in column '{request.group_col}'.")

    group_a = str(request.group_a) if request.group_a is not None else None
    group_b = str(request.group_b) if request.group_b is not None else None

    if group_a is None and len(groups) == 2:
        group_a = groups[0]
        group_b = groups[1]
    elif group_a is None:
        raise ValueError("A comparison group must be specified when more than two groups exist.")

    if request.compare_to_rest:
        a = dff.loc[dff[request.group_col] == group_a, request.metric]
        b = dff.loc[dff[request.group_col] != group_a, request.metric]
        if a.empty or b.empty:
            raise ValueError(f"Could not build '{group_a}' versus rest comparison from {request.group_col}.")
        return a, b, group_a, "rest"

    if group_b is None:
        remaining = [g for g in groups if g != group_a]
        if len(remaining) == 1:
            group_b = remaining[0]
        else:
            raise ValueError("Both comparison groups must be specified when more than two groups exist.")

    a = dff.loc[dff[request.group_col] == group_a, request.metric]
    b = dff.loc[dff[request.group_col] == group_b, request.metric]
    if a.empty or b.empty:
        raise ValueError(f"One of the requested groups has no rows: {group_a}, {group_b}.")
    return a, b, group_a, group_b


def select_hypothesis_method(df: pd.DataFrame, request: HypothesisTestRequest) -> StatisticalSelection:
    if request.group_col not in df.columns:
        raise ValueError(f"Column not found: {request.group_col}")
    if request.metric not in df.columns:
        raise ValueError(f"Column not found: {request.metric}")

    metric_series = df[request.metric]
    if request.paired:
        return StatisticalSelection(
            analysis_type="hypothesis_test",
            method="paired_t_test",
            reason="Pairing was explicitly requested.",
        )

    if _is_binary_metric(metric_series, metric_name=request.metric, success_value=request.success_value):
        return StatisticalSelection(
            analysis_type="hypothesis_test",
            method="two_proportion_z_test",
            reason="Metric looks binary, so a rate comparison is more appropriate than a mean test.",
        )

    numeric = pd.to_numeric(metric_series, errors="coerce")
    if numeric.notna().sum() >= max(3, int(metric_series.notna().sum() * 0.6)):
        a, b, label_a, label_b = prepare_two_groups(df, request)
        normal_a = check_normality(pd.to_numeric(a, errors="coerce").dropna(), alpha=request.alpha, label=label_a)
        normal_b = check_normality(pd.to_numeric(b, errors="coerce").dropna(), alpha=request.alpha, label=label_b)
        min_n = min(int(pd.to_numeric(a, errors="coerce").dropna().shape[0]), int(pd.to_numeric(b, errors="coerce").dropna().shape[0]))
        if min_n < 20 and (normal_a.passed is False or normal_b.passed is False):
            return StatisticalSelection(
                analysis_type="hypothesis_test",
                method="mann_whitney_u",
                reason="Small-sample numeric comparison with weak normality checks; using nonparametric fallback.",
                fallback_used=True,
            )
        return StatisticalSelection(
            analysis_type="hypothesis_test",
            method="welch_t_test",
            reason="Independent numeric groups default to Welch's t-test for robustness to unequal variance.",
        )

    return StatisticalSelection(
        analysis_type="hypothesis_test",
        method="chi_square_independence",
        reason="Compared columns are categorical or not safely numeric, so contingency-table association is used.",
    )


def run_hypothesis_test(df: pd.DataFrame, request: HypothesisTestRequest, *, analysis_id: str) -> tuple[StatisticalSelection, StatisticalResult]:
    selection = select_hypothesis_method(df, request)
    if selection.method == "paired_t_test":
        if not request.pair_id_col or request.pair_id_col not in df.columns:
            raise ValueError("Paired t-test requires pair_id_col present in the dataframe.")
        dff = df.dropna(subset=[request.group_col, request.metric, request.pair_id_col]).copy()
        dff[request.group_col] = dff[request.group_col].astype(str)
        groups = _unique_groups(dff[request.group_col])
        if request.group_a is None or request.group_b is None:
            if len(groups) != 2:
                raise ValueError("Paired test requires exactly two groups or explicit group_a/group_b.")
            group_a, group_b = groups[0], groups[1]
        else:
            group_a, group_b = str(request.group_a), str(request.group_b)
        subset = dff[dff[request.group_col].isin([group_a, group_b])]
        pivot = (
            subset.groupby([request.pair_id_col, request.group_col])[request.metric]
            .mean()
            .unstack(request.group_col)
            .dropna(subset=[group_a, group_b])
        )
        if pivot.empty:
            raise ValueError("No complete matched pairs were available for the paired test.")
        return selection, paired_t_test(
            analysis_id=analysis_id,
            paired=pivot.reset_index(drop=True),
            left_col=group_a,
            right_col=group_b,
            label_left=group_a,
            label_right=group_b,
            metric=request.metric,
            alpha=request.alpha,
            alternative=request.alternative,
        )

    if selection.method == "chi_square_independence":
        dff = df.dropna(subset=[request.group_col, request.metric]).copy()
        return selection, chi_square_independence_test(
            analysis_id=analysis_id,
            left=dff[request.group_col],
            right=dff[request.metric],
            left_name=request.group_col,
            right_name=request.metric,
            alpha=request.alpha,
        )

    group_a, group_b, label_a, label_b = prepare_two_groups(df, request)
    if selection.method == "two_proportion_z_test":
        return selection, two_proportion_z_test(
            analysis_id=analysis_id,
            group_a=group_a,
            group_b=group_b,
            label_a=label_a,
            label_b=label_b,
            metric=request.metric,
            success_value=request.success_value,
            alpha=request.alpha,
        )
    if selection.method == "mann_whitney_u":
        return selection, mann_whitney_u_test(
            analysis_id=analysis_id,
            group_a=group_a,
            group_b=group_b,
            label_a=label_a,
            label_b=label_b,
            metric=request.metric,
            alpha=request.alpha,
            alternative=request.alternative,
        )
    return selection, welch_t_test(
        analysis_id=analysis_id,
        group_a=group_a,
        group_b=group_b,
        label_a=label_a,
        label_b=label_b,
        metric=request.metric,
        alpha=request.alpha,
        alternative=request.alternative,
    )
