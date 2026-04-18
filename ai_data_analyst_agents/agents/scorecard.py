from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.core.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    validate_run_scorecard_contract,
)


EV_TAG_PATTERN = re.compile(r"\[\[(?:EV:)?(EV-[a-f0-9]{10})\]\]")
EVIDENCE_TABLE_PATTERN = re.compile(r"\|\s*\[(\d{1,4})\]\s*\|\s*(EV-[a-f0-9]{10})\s*\|")


def _framing_completeness(framing: Dict[str, Any], analysis_type: str | None) -> float:
    keys = [
        "target_metric",
        "aggregation_level",
        "metric_aggregation",
        "time_column",
        "segment_columns",
        "comparison_logic",
        "success_criterion",
        "analysis_limitations",
    ]
    if not framing:
        return 0.0

    present = 0
    for k in keys:
        if k not in framing:
            continue
        value = framing.get(k)
        if k == "time_column":
            # time column can legitimately be null for non-trend analyses.
            if value is not None or analysis_type not in {"trend", "forecasting_unsupported"}:
                present += 1
            continue
        if k == "segment_columns":
            if isinstance(value, list):
                present += 1
            continue
        if k == "analysis_limitations":
            if isinstance(value, list) and value:
                present += 1
            continue
        if value is not None and str(value).strip():
            present += 1
    return round(present / float(len(keys)), 4)


def _report_referenced_evidence(report_text: str) -> set[str]:
    refs = set(EV_TAG_PATTERN.findall(report_text or ""))
    for _, ev_id in EVIDENCE_TABLE_PATTERN.findall(report_text or ""):
        refs.add(ev_id)
    return refs


def _task_type_by_id(tasks: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for t in tasks:
        tid = str(t.get("id", "")).strip()
        ttype = str(t.get("type", "")).strip()
        if tid and ttype:
            out[tid] = ttype
    return out


class ScorecardAgent(Agent):
    name = "scorecard"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        store = ctx["store"]
        memory = ctx["memory"]
        evidence_store = ctx["evidence"]

        intake_out = memory.get("result.intake") or {}
        plan_out = memory.get("result.planner") or {}
        metrics_out = memory.get("result.metrics") or {}
        review_out = memory.get("result.reviewer") or {}
        report_meta = memory.get("result.reporting_metadata") or {}

        tasks = list(plan_out.get("tasks", []) or [])
        task_types = [str(t.get("type", "")).strip() for t in tasks if str(t.get("type", "")).strip()]
        type_counts = Counter(task_types)
        type_by_id = _task_type_by_id(tasks)
        failed_reasons = Counter(
            str(x.get("reason", "")).strip()
            for x in [*(metrics_out.get("failed", []) or []), *(metrics_out.get("skipped", []) or [])]
            if str(x.get("reason", "")).strip()
        )

        planned = len(tasks)
        computed = len(metrics_out.get("computed", []) or [])
        failed = len(metrics_out.get("failed", []) or [])
        skipped = len(metrics_out.get("skipped", []) or [])
        duplicate_task_count = max(0, planned - len({str(t.get("provenance", {}).get("semantic_key", "")) for t in tasks}))

        computed_by_type = Counter(
            type_by_id.get(str(item.get("task_id", "")).strip(), "unknown")
            for item in (metrics_out.get("computed", []) or [])
        )
        failed_by_type = Counter(
            type_by_id.get(str(item.get("task_id", "")).strip(), "unknown")
            for item in [*(metrics_out.get("failed", []) or []), *(metrics_out.get("skipped", []) or [])]
        )

        report_path = store.path("final_report.md")
        report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        evidence_ids = set(evidence_store.all().keys())
        referenced_ids = _report_referenced_evidence(report_text)

        evidence_coverage_ratio = float(len(referenced_ids & evidence_ids) / len(evidence_ids)) if evidence_ids else 0.0
        task_success_rate = float(computed / planned) if planned else 0.0
        quality_status = "pass"
        if str(review_out.get("status", "fail")) != "pass":
            quality_status = "fail"
        if int(report_meta.get("unresolved_ev_placeholders", 1)) > 0:
            quality_status = "fail"
        if int(report_meta.get("unsupported_numeric_claim_lines", 1)) > 0:
            quality_status = "fail"
        if not bool(report_meta.get("section_completeness_ok", False)):
            quality_status = "fail"

        analysis_type = (
            plan_out.get("analysis_type")
            or intake_out.get("analysis_type")
            or (memory.get("result.analysis_plan") or {}).get("analysis_type")
        )
        plan_framing = (
            (memory.get("result.analysis_plan") or {}).get("framing")
            or plan_out.get("planning_contract")
            or intake_out.get("framing")
            or {}
        )

        scorecard = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "analysis_type": analysis_type,
            "feasibility_status": (
                plan_out.get("feasibility_status")
                or intake_out.get("feasibility_status")
            ),
            "routing_confidence": intake_out.get("routing_confidence"),
            "framing_completeness_pct": _framing_completeness(plan_framing, analysis_type),
            "task_summary": {
                "planned_tasks": planned,
                "computed_tasks": computed,
                "failed_tasks": failed,
                "skipped_tasks": skipped,
                "task_success_rate": round(task_success_rate, 4),
                "duplicate_task_count": duplicate_task_count,
                "planned_by_type": dict(type_counts),
                "computed_by_type": dict(computed_by_type),
                "failed_by_type": dict(failed_by_type),
                "failure_reasons_top": [
                    {"reason": reason, "count": count}
                    for reason, count in failed_reasons.most_common(10)
                ],
            },
            "evidence_coverage": {
                "evidence_ids_total": len(evidence_ids),
                "evidence_ids_referenced": len(referenced_ids & evidence_ids),
                "coverage_ratio": round(evidence_coverage_ratio, 4),
            },
            "quality_gates": {
                "reviewer_status": str(review_out.get("status", "fail")),
                "reviewer_violation_count": len(review_out.get("violations", []) or []),
                "report_unresolved_placeholders": int(report_meta.get("unresolved_ev_placeholders", 0)),
                "report_unsupported_numeric_claim_lines": int(report_meta.get("unsupported_numeric_claim_lines", 0)),
                "report_section_completeness_ok": bool(report_meta.get("section_completeness_ok", False)),
                "report_contradiction_count": int(report_meta.get("contradiction_count", 0)),
            },
            "final_quality_status": quality_status,
        }
        scorecard = validate_run_scorecard_contract(scorecard).model_dump()
        store.write_json("run_scorecard.json", scorecard)
        memory.set("result.scorecard", scorecard)
        return scorecard
