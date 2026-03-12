from __future__ import annotations

import pandas as pd

from ai_data_analyst_agents.statistics.effect_sizes import percent_change, relative_lift
from ai_data_analyst_agents.statistics.limitations import build_statistical_limitations
from ai_data_analyst_agents.statistics.models import ABTestRequest, HypothesisTestRequest, StatisticalResult
from ai_data_analyst_agents.statistics.selector import run_hypothesis_test


def run_ab_test(df: pd.DataFrame, request: ABTestRequest, *, analysis_id: str) -> StatisticalResult:
    base_request = HypothesisTestRequest(
        group_col=request.group_col,
        metric=request.metric,
        group_a=request.treatment,
        group_b=request.control,
        success_value=request.success_value,
        alpha=request.alpha,
    )
    selection, result = run_hypothesis_test(df, base_request, analysis_id=analysis_id)

    dff = df.dropna(subset=[request.group_col, request.metric]).copy()
    dff[request.group_col] = dff[request.group_col].astype(str)
    treatment = dff.loc[dff[request.group_col] == str(request.treatment), request.metric]
    control = dff.loc[dff[request.group_col] == str(request.control), request.metric]

    treatment_numeric = pd.to_numeric(treatment, errors="coerce").dropna()
    control_numeric = pd.to_numeric(control, errors="coerce").dropna()
    if not treatment_numeric.empty and not control_numeric.empty:
        treat_mean = float(treatment_numeric.mean())
        control_mean = float(control_numeric.mean())
        result.effect_sizes.append(relative_lift(treat_mean, control_mean))
        result.effect_sizes.append(percent_change(treat_mean, control_mean))
        result.metrics["treatment_mean"] = treat_mean
        result.metrics["control_mean"] = control_mean
        result.metrics["absolute_difference"] = treat_mean - control_mean

    min_n = min(int(treatment.shape[0]), int(control.shape[0]))
    if selection.method == "two_proportion_z_test" and min_n < 100:
        result.warnings.append(
            f"A/B test may be underpowered for a binary metric because one or both groups have fewer than 100 observations (min_n={min_n})."
        )
    elif selection.method != "two_proportion_z_test" and min_n < 30:
        result.warnings.append(
            f"A/B test may be underpowered for a continuous metric because one or both groups have fewer than 30 observations (min_n={min_n})."
        )

    n_treat = int(treatment.shape[0])
    n_control = int(control.shape[0])
    if min(n_treat, n_control) > 0:
        imbalance_ratio = max(n_treat, n_control) / min(n_treat, n_control)
        if imbalance_ratio > 3.0:
            result.warnings.append(f"Treatment/control imbalance ratio is {imbalance_ratio:.2f}, which can reduce stability.")
        result.metrics["imbalance_ratio"] = imbalance_ratio

    result.analysis_type = "ab_test"
    result.method_reason = f"A/B workflow selected {result.method} based on metric type and group structure. {selection.reason}"
    result.plain_language = (
        f"A/B result for {request.metric}: {result.plain_language} "
        f"Treatment={request.treatment}, control={request.control}."
    )
    result.limitations = build_statistical_limitations(result)
    return result
