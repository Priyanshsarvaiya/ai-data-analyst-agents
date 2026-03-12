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
        lines.append("No major statistical caveats were triggered beyond standard observational limitations.")
    return lines[:8]
