from __future__ import annotations

from typing import Any


def run_reporting(cfg, plan: dict[str, Any], profile: dict[str, Any], qa: dict[str, Any], eda: dict[str, Any], store, logger) -> str:
    # Phase 1: deterministic report. Next step: have OpenRouter write narrative using these artifacts.
    lines = []
    lines.append("# Final Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Business question: **{plan['business_question']}**")
    lines.append(f"- Rows/Cols: **{profile['n_rows']} x {profile['n_cols']}**")
    lines.append(f"- Duplicate rate: **{qa['duplicate_rate']:.2%}**")
    lines.append(f"- Warnings: **{len(qa.get('warnings', []))}** (see `quality_warnings.md`)")
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append("- See `data_profile.json`")
    lines.append("")
    lines.append("## Data Quality Findings")
    if qa.get("warnings"):
        for w in qa["warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("- No major issues detected by Phase 1 checks.")
    lines.append("")
    lines.append("## EDA Summary")
    if eda.get("numeric_columns"):
        lines.append(f"- Numeric columns: {', '.join(eda['numeric_columns'][:10])}")
    if eda.get("charts"):
        lines.append(f"- Charts generated: {', '.join(eda['charts'])}")
    lines.append("")
    lines.append("## Artifacts Index")
    lines.append("- `analysis_plan.json`")
    lines.append("- `data_profile.json`")
    lines.append("- `quality_report.json`")
    lines.append("- `quality_warnings.md`")
    lines.append("- `cleaned.csv`")
    lines.append("- `feature_log.json`")
    lines.append("- `eda_summary.json`")
    lines.append("- `charts/`")
    report_md = "\n".join(lines)

    store.write_text("final_report.md", report_md)
    logger.info("Wrote final_report.md")
    return report_md