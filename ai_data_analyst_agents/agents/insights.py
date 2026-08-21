from __future__ import annotations

from typing import Any, Dict

from ai_data_analyst_agents.agents.planner import MANDATORY_TASK_TYPES_BY_ANALYSIS_TYPE
from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.core.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    validate_analysis_readiness_contract,
)


GUIDANCE_BY_ANALYSIS_TYPE: dict[str, list[str]] = {
    "descriptive": ["Describe observed levels and rankings; do not infer causes."],
    "trend": ["Describe historical movement only; do not present it as a forecast."],
    "segment_comparison": ["Quantify absolute and relative segment differences."],
    "diagnostic": [
        "Lead with the arithmetic gap decomposition when available.",
        "Call correlations and segment contrasts associations or observed contributors, not causes.",
        "Do not call mix a leading contributor unless a cited artifact quantifies its contribution.",
    ],
    "experiment_ab": ["State uncertainty, confidence intervals, effect size, and the experiment's causal scope."],
    "forecasting_unsupported": ["Clearly separate the historical baseline from any future-looking request."],
    "impossible": ["State the missing requirements and avoid manufacturing a proxy answer."],
}


class InsightsAgent(Agent):
    """Deterministic question-coverage gate between computation and prose."""

    name = "insights"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        store = ctx["store"]
        memory = ctx["memory"]
        logger = ctx["logger"]
        plan = memory.get("result.planner") or {}
        metrics = memory.get("result.metrics") or {}
        analysis_type = str(plan.get("analysis_type") or "descriptive")
        tasks = list(plan.get("tasks", []) or [])
        by_id = {str(t.get("id")): str(t.get("type")) for t in tasks}

        computed = {
            by_id.get(str(item.get("task_id")), str(item.get("task_type") or ""))
            for item in (metrics.get("computed", []) or [])
        }
        computed.discard("")
        failed_or_skipped = {
            by_id.get(str(item.get("task_id")), "")
            for item in [*(metrics.get("failed", []) or []), *(metrics.get("skipped", []) or [])]
        }
        failed_or_skipped.discard("")
        required = set(MANDATORY_TASK_TYPES_BY_ANALYSIS_TYPE.get(analysis_type, set()))
        missing = required - computed
        blocked_requirements = list(plan.get("blocked_requirements", []) or [])

        if blocked_requirements or analysis_type == "impossible":
            answer_status = "blocked"
        elif missing:
            answer_status = "partial"
        else:
            answer_status = "ready"

        guidance = list(GUIDANCE_BY_ANALYSIS_TYPE.get(analysis_type, []))
        if missing:
            guidance.append(
                "State that the answer is partial and name these unclosed computation gaps: "
                + ", ".join(sorted(missing))
                + "."
            )
        if failed_or_skipped:
            guidance.append("Disclose failed or skipped capabilities that matter to the answer.")

        output = validate_analysis_readiness_contract(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "analysis_type": analysis_type,
                "answer_status": answer_status,
                "required_capabilities": sorted(required),
                "computed_capabilities": sorted(computed),
                "missing_capabilities": sorted(missing),
                "failed_or_skipped_capabilities": sorted(failed_or_skipped),
                "reporting_guidance": guidance,
            }
        ).model_dump()
        store.write_json("analysis_readiness.json", output)
        logger.info(
            "[Insights] answer_status=%s missing=%s",
            answer_status,
            sorted(missing),
        )
        return output
