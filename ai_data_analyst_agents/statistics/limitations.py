from __future__ import annotations

from ai_data_analyst_agents.statistics.models import StatisticalResult


def build_statistical_limitations(result: StatisticalResult) -> list[str]:
    lines: list[str] = []
    for check in result.assumptions:
        if check.passed is False and check.severity in {"warn", "fail"}:
            lines.append(check.detail)
    for warning in result.warnings:
        if warning not in lines:
            lines.append(warning)
    for limit in result.limitations:
        if limit not in lines:
            lines.append(limit)
    if not lines:
        if result.analysis_type == "ab_test":
            lines.append(
                "Causal interpretation depends on randomized assignment, no interference, and valid exposure measurement."
            )
        elif result.analysis_type == "regression":
            lines.append("Regression estimates conditional associations and does not establish causality.")
        else:
            lines.append("Observed group differences may reflect confounding and do not establish causality.")
    return lines[:8]
