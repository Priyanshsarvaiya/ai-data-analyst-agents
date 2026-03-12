from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from ai_data_analyst_agents.core.artifacts import ArtifactStore
from ai_data_analyst_agents.statistics.models import StatisticalArtifactBundle, StatisticalResult


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (text or "analysis").strip().lower())
    return cleaned.strip("_") or "analysis"


def _result_markdown(result: StatisticalResult) -> str:
    lines = [
        f"# Statistical Analysis: {result.analysis_id}",
        "",
        f"- Analysis type: {result.analysis_type}",
        f"- Method: {result.method}",
        f"- Method rationale: {result.method_reason}",
        f"- Decision: {result.decision}",
        f"- P-value: {result.p_value if result.p_value is not None else 'n/a'}",
        f"- Plain-language summary: {result.plain_language}",
        "",
        "## Assumptions",
    ]
    for check in result.assumptions:
        status = "pass" if check.passed else ("warn" if check.passed is None else "fail")
        lines.append(f"- {check.name} ({status}): {check.detail}")
    lines.extend(["", "## Confidence Intervals"])
    if result.confidence_intervals:
        for ci in result.confidence_intervals:
            lines.append(
                f"- {ci.parameter}: {ci.point_estimate:.4f} [{ci.lower_bound:.4f}, {ci.upper_bound:.4f}] at {ci.confidence_level:.0%}"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Effect Sizes"])
    if result.effect_sizes:
        for eff in result.effect_sizes:
            lines.append(f"- {eff.name}: {eff.value:.4f} ({eff.interpretation})")
    else:
        lines.append("- None")
    lines.extend(["", "## Limitations"])
    for line in result.limitations or ["- None recorded."]:
        if line.startswith("-"):
            lines.append(line)
        else:
            lines.append(f"- {line}")
    return "\n".join(lines) + "\n"


def write_statistical_artifacts(
    store: ArtifactStore,
    *,
    task_id: str,
    result: StatisticalResult,
    coefficients: pd.DataFrame | None = None,
    diagnostics: dict[str, object] | None = None,
) -> StatisticalArtifactBundle:
    base = Path("statistics") / f"{task_id}_{_slug(result.method)}"
    summary_path = str(base / "summary.json")
    assumptions_path = str(base / "assumptions.json")
    results_path = str(base / "results.md")
    diagnostics_path = str(base / "diagnostics.json") if diagnostics is not None else None
    coefficients_path = str(base / "coefficients.csv") if coefficients is not None and not coefficients.empty else None

    store.write_json(summary_path, result.to_dict())
    store.write_json(assumptions_path, [check.to_dict() for check in result.assumptions])
    store.write_text(results_path, _result_markdown(result))

    if diagnostics_path is not None:
        store.write_json(diagnostics_path, diagnostics)
    if coefficients_path is not None and coefficients is not None:
        coeff_p = store.path(coefficients_path)
        coeff_p.parent.mkdir(parents=True, exist_ok=True)
        coefficients.to_csv(coeff_p, index=False)
        store.register_file(coefficients_path)

    return StatisticalArtifactBundle(
        summary_path=summary_path,
        assumptions_path=assumptions_path,
        results_path=results_path,
        diagnostics_path=diagnostics_path,
        coefficients_path=coefficients_path,
    )
