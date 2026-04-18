from __future__ import annotations

from typing import Any, Dict, List
import re

import pandas as pd

from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.core.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    validate_analysis_plan_contract,
)
from ai_data_analyst_agents.core.kpi_templates import (
    default_agg_for_metric,
    detect_business_domain,
)


_TREND_KEYWORDS = {"trend", "over time", "monthly", "weekly", "daily", "season", "retention", "cohort", "recency"}
_SEGMENT_COMPARE_KEYWORDS = {"vs", "versus", "compare", "difference", "higher", "lower", "between", "by"}
_DIAGNOSTIC_KEYWORDS = {"why", "driver", "drivers", "cause", "reason", "reasons", "diagnostic"}
_EXPERIMENT_KEYWORDS = {"experiment", "a/b", "ab test", "treatment", "control", "lift", "variant"}
_FORECAST_KEYWORDS = {"forecast", "predict", "next month", "next quarter", "next year", "projection", "project"}
_EXTERNAL_DATA_KEYWORDS = {"competitor", "industry benchmark", "macro", "economic", "sentiment", "weather"}

_SEGMENT_CANDIDATES = [
    "country",
    "region",
    "market",
    "product_category",
    "category",
    "segment",
    "channel",
    "customer_id",
]
_EXPERIMENT_GROUP_COL_CANDIDATES = [
    "variant",
    "experiment_group",
    "treatment_group",
    "treatment",
    "group",
    "arm",
    "bucket",
]
_METRIC_PREFERENCE = [
    "revenue",
    "sales",
    "amount",
    "total_amount",
    "gmv",
    "profit",
    "price",
    "quantity",
    "orders",
]


def _contains_any(text: str, words: set[str]) -> bool:
    for w in words:
        word = (w or "").strip().lower()
        if not word:
            continue
        if " " in word or "-" in word:
            if word in text:
                return True
        elif re.search(rf"\b{re.escape(word)}\b", text):
            return True
    return False


def _infer_numeric_cols(df: pd.DataFrame) -> List[str]:
    return [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _infer_time_cols(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    for c in df.columns:
        cstr = str(c)
        cl = cstr.lower()
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            out.append(cstr)
            continue
        if any(tok in cl for tok in ("date", "time", "timestamp", "week", "month", "year")):
            out.append(cstr)
    return out


def _choose_metric(question: str, numeric_cols: List[str]) -> str | None:
    q = (question or "").lower()
    for col in numeric_cols:
        if re.search(rf"\b{re.escape(col.lower())}\b", q):
            return col
    for m in _METRIC_PREFERENCE:
        if m in numeric_cols:
            return m
    return numeric_cols[0] if numeric_cols else None


def _choose_segment_cols(question: str, schema_cols: List[str]) -> List[str]:
    q = (question or "").lower()
    out: List[str] = []
    for c in _SEGMENT_CANDIDATES:
        if c in schema_cols and (c in q or len(out) < 2):
            out.append(c)
    dedup: List[str] = []
    seen = set()
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup[:2]


def _choose_experiment_group_col(schema_cols: List[str]) -> str | None:
    for c in _EXPERIMENT_GROUP_COL_CANDIDATES:
        if c in schema_cols:
            return c
    return None


def _build_comparison_logic(
    analysis_type: str,
    metric: str | None,
    segment_cols: List[str],
    experiment_group_col: str | None,
    time_col: str | None,
) -> str:
    if analysis_type == "experiment_ab":
        return (
            f"Compare treatment vs control by {metric} across {experiment_group_col} "
            "and estimate statistical significance."
        )
    if analysis_type == "trend":
        return f"Analyze period-over-period movement of {metric} on {time_col}."
    if analysis_type == "segment_comparison":
        seg = segment_cols[0] if segment_cols else "available segments"
        return f"Compare {metric} across {seg} and quantify absolute and relative deltas."
    if analysis_type == "diagnostic":
        seg = segment_cols[0] if segment_cols else "major dimensions"
        return f"Decompose {metric} by {seg}, volume, and mix to identify likely drivers."
    if analysis_type == "forecasting_unsupported":
        return f"Provide historical baseline for {metric} without forecasting future periods."
    if analysis_type == "impossible":
        return "State infeasibility and list missing requirements without unsupported claims."
    return f"Summarize overall {metric} and key segment breakdowns."


def _build_success_criterion(analysis_type: str) -> str:
    if analysis_type == "experiment_ab":
        return "Report lift and p-value at alpha=0.05 using valid experiment groups."
    if analysis_type == "trend":
        return "Report direction and magnitude of change with enough historical periods."
    if analysis_type == "segment_comparison":
        return "Report ranked segment differences with absolute and percentage deltas."
    if analysis_type == "diagnostic":
        return "Report top contributors/drivers with artifact-backed evidence."
    if analysis_type == "forecasting_unsupported":
        return "Report historical performance and explicitly flag forecasting as unsupported."
    if analysis_type == "impossible":
        return "Return a clear infeasibility summary with specific missing data requirements."
    return "Return accurate descriptive KPIs grounded in computed artifacts."


def _classify(
    question: str,
    *,
    time_cols: List[str],
    segment_cols: List[str],
    metric: str | None,
    experiment_group_col: str | None,
) -> tuple[str, str, float, List[str]]:
    q = (question or "").lower()
    blocked: List[str] = []

    if _contains_any(q, _FORECAST_KEYWORDS):
        if not time_cols:
            blocked.append("missing_time_column")
        if metric is None:
            blocked.append("missing_numeric_metric")
        return (
            "forecasting_unsupported",
            "Question asks for forward-looking prediction; this pipeline only supports historical analysis.",
            0.98,
            blocked,
        )

    if _contains_any(q, _EXTERNAL_DATA_KEYWORDS):
        blocked.append("requires_external_data")
        return (
            "impossible",
            "Question requires external context not present in the provided dataset.",
            0.9,
            blocked,
        )

    if _contains_any(q, _EXPERIMENT_KEYWORDS):
        if not experiment_group_col:
            blocked.append("missing_experiment_group_column")
        if metric is None:
            blocked.append("missing_numeric_metric")
        if blocked:
            return (
                "impossible",
                "Experiment question detected but required experiment columns are unavailable.",
                0.9,
                blocked,
            )
        return ("experiment_ab", "Experiment language detected (treatment/control/A-B).", 0.95, blocked)

    if _contains_any(q, _DIAGNOSTIC_KEYWORDS):
        if metric is None:
            blocked.append("missing_numeric_metric")
        if not segment_cols:
            blocked.append("missing_segment_column")
        return ("diagnostic", "Driver/why question detected.", 0.88, blocked)

    if _contains_any(q, _TREND_KEYWORDS):
        if not time_cols:
            blocked.append("missing_time_column")
            return (
                "impossible",
                "Time-trend question detected but no usable time column is available.",
                0.9,
                blocked,
            )
        if metric is None:
            blocked.append("missing_numeric_metric")
        return ("trend", "Temporal/trend intent detected.", 0.86, blocked)

    if _contains_any(q, _SEGMENT_COMPARE_KEYWORDS):
        if not segment_cols:
            blocked.append("missing_segment_column")
            return (
                "impossible",
                "Segment comparison intent detected but no segment column is available.",
                0.82,
                blocked,
            )
        if metric is None:
            blocked.append("missing_numeric_metric")
        return ("segment_comparison", "Comparison intent detected.", 0.8, blocked)

    if metric is None:
        blocked.append("missing_numeric_metric")
    return ("descriptive", "Default descriptive analysis route.", 0.55, blocked)


def _build_limitations(analysis_type: str, blocked: List[str]) -> List[str]:
    limits: List[str] = []
    if analysis_type == "forecasting_unsupported":
        limits.append("Forecasting is not executed; only historical analysis is provided.")
    if analysis_type == "impossible":
        limits.append("Current dataset is insufficient to answer the question directly.")

    blocked_messages = {
        "missing_time_column": "No usable time column is available for temporal analysis.",
        "missing_segment_column": "No clear segment column is available for comparison.",
        "missing_numeric_metric": "No numeric metric candidate is available for quantitative analysis.",
        "missing_experiment_group_column": "No experiment group column (control/treatment variant) is available.",
        "requires_external_data": "Question requires external benchmark/context not present in the dataset.",
    }
    for b in blocked:
        msg = blocked_messages.get(b)
        if msg and msg not in limits:
            limits.append(msg)

    limits.append("All outputs are constrained to provided dataset columns and computed artifacts.")
    return limits


class IntakeAgent(Agent):
    name = "intake"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        store = ctx["store"]
        logger = ctx["logger"]
        question = ctx["business_question"]
        df = ctx["df"]
        source = ctx.get("source", {"type": "csv"})
        schema_cols = [str(c) for c in df.columns]
        domain = detect_business_domain(question, schema_cols)

        numeric_cols = _infer_numeric_cols(df)
        time_cols = _infer_time_cols(df)
        segment_cols = _choose_segment_cols(question, schema_cols)
        metric = _choose_metric(question, numeric_cols)
        experiment_group_col = _choose_experiment_group_col(schema_cols)

        analysis_type, routing_reason, routing_confidence, blocked = _classify(
            question,
            time_cols=time_cols,
            segment_cols=segment_cols,
            metric=metric,
            experiment_group_col=experiment_group_col,
        )
        feasibility_status = "infeasible" if analysis_type == "impossible" else ("partially_feasible" if blocked else "feasible")
        time_col = time_cols[0] if time_cols else None
        agg = default_agg_for_metric(metric or "metric")
        aggregation_level = "overall"
        if analysis_type == "trend" and time_col:
            aggregation_level = f"time:{time_col}"
        elif analysis_type == "experiment_ab" and experiment_group_col:
            aggregation_level = f"group:{experiment_group_col}"
        elif segment_cols:
            aggregation_level = f"segment:{segment_cols[0]}"

        limitations = _build_limitations(analysis_type, blocked)
        framing = {
            "target_metric": metric,
            "aggregation_level": aggregation_level,
            "metric_aggregation": agg,
            "time_column": time_col,
            "segment_columns": segment_cols,
            "comparison_logic": _build_comparison_logic(
                analysis_type,
                metric=metric,
                segment_cols=segment_cols,
                experiment_group_col=experiment_group_col,
                time_col=time_col,
            ),
            "success_criterion": _build_success_criterion(analysis_type),
            "analysis_limitations": limitations,
            "analysis_type": analysis_type,
            "feasibility_status": feasibility_status,
        }

        plan = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "business_question": question,
            "source_type": source.get("type", "csv"),
            "suggested_domain": domain,
            "analysis_type": analysis_type,
            "routing_reason": routing_reason,
            "routing_confidence": routing_confidence,
            "blocked_requirements": blocked,
            "feasibility_status": feasibility_status,
            "framing": framing,
            "assumptions": [
                "All computations are based only on provided dataset artifacts.",
                "If time columns exist, they may need parsing for time-based insights.",
            ],
            "suggested_slices": segment_cols or ["country", "product_category"],
            "requested_metrics": [x for x in [metric, "revenue", "orders", "avg_order_value"] if x],
        }

        plan = validate_analysis_plan_contract(plan).model_dump()

        store.write_json("analysis_plan.json", plan)
        logger.info("Wrote analysis_plan.json")
        return plan
