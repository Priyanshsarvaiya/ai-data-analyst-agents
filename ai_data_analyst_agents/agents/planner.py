from __future__ import annotations

from typing import Any, Dict, List
import json
import re

from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.core.kpi_templates import (
    default_agg_for_metric,
    detect_business_domain,
    is_agg_allowed_for_metric,
    pick_cohort_columns,
)
from ai_data_analyst_agents.core.openrouter_client import OpenRouterClient
from ai_data_analyst_agents.core.security import validate_read_only_sql
from ai_data_analyst_agents.core.sql_source import (
    build_groupby_query,
    choose_primary_table,
    column_table_index,
)


SYSTEM = """You are a senior data analyst planner.

Your job: convert business question + dataframe schema (+ optional SQL schema) into concrete computation tasks.

IMPORTANT OUTPUT RULES
- Return ONLY valid JSON (no markdown, no commentary, no extra keys).
- Use ONLY columns that exist in provided schema context.
- Use exact param keys.

JSON OUTPUT SCHEMA
{
  "tasks": [
    {
      "id": "T1",
      "type": "groupby_agg" | "groupby2_agg" | "filter_agg" | "correlation" | "distribution" | "group_distribution" | "recency_by_group" | "topk" | "timeseries_agg" | "sql_query" | "sql_join_profile" | "kpi_template_apply" | "metric_definition" | "segment_analysis" | "cohort_analysis" | "statistical_test" | "ab_test" | "ols_regression",
      "params": {}
    }
  ],
  "notes": "short"
}

TASK PARAM SCHEMAS
1) groupby_agg: {group_by, metric, agg?, limit?}
2) groupby2_agg: {group_by_1, group_by_2, metric, agg?, limit?}
3) filter_agg: {filter_col, filter_val, metric, agg?}
4) correlation: {x, y}
5) distribution: {column, quantiles?}
6) group_distribution: {group_by, metric, agg?, quantiles?}
7) recency_by_group: {group_by, date_col}
8) topk: {by, metric, agg?, k?}
9) timeseries_agg: {date_col, metric, freq?, agg?}
10) sql_query: {query, limit?, output?} where output in {"rows","mapping"}
11) sql_join_profile: {fact_table, dimension_table?}
12) kpi_template_apply: {domain, segment_by?}
13) metric_definition: {name, metric_col?, expression?, agg?, group_by?}
14) segment_analysis: {segment_by, metric, agg?, limit?}
15) cohort_analysis: {entity_col, date_col, freq?}
16) statistical_test: {group_col, metric, group_a?, group_b?, compare_to_rest?, paired?, pair_id_col?, success_value?, alpha?, alternative?}
17) ab_test: {group_col, control, treatment, metric, metric_type?, success_value?, alpha?}
18) ols_regression: {target, predictors, alpha?}

HEURISTICS
- Always include direct answer tasks first.
- For "why/driver/difference": include contrast + volume + average + mix + driver checks.
- For SQL multi-table schemas: include schema-aware sql_query and join-profile tasks when relevant columns are in different tables.
- For business context: include kpi_template_apply for best-fit domain and add segment/cohort tasks if question implies segmentation or retention.

Return ONLY the JSON object.
"""


WHY_KEYWORDS = {"why", "driver", "drivers", "cause", "caused", "reason", "reasons"}
COMPARE_KEYWORDS = {"less", "lower", "higher", "difference", "vs", "versus", "compare"}
TOTAL_KEYWORDS = {"total", "sum", "overall", "how much"}
COUNT_KEYWORDS = {"count", "how many", "number of", "orders", "transactions", "volume"}
CUSTOMER_KEYWORDS = {"customer", "customers", "high-value", "high value", "vip", "rfm", "retention", "churn"}
PRODUCT_KEYWORDS = {"product", "category", "categories", "sku", "item", "items"}
TIME_KEYWORDS = {"trend", "over time", "monthly", "weekly", "daily", "season", "cohort", "recency", "lifetime"}
SEGMENT_KEYWORDS = {"segment", "segments", "percentile", "quantile", "top", "bottom", "distribution"}
SQL_KEYWORDS = {"sql", "database", "table", "join", "joins", "postgres", "postgresql", "sqlite"}
STAT_KEYWORDS = {
    "significant",
    "statistically",
    "confidence",
    "interval",
    "uncertainty",
    "p-value",
    "p value",
    "hypothesis",
    "experiment",
    "a/b",
    "ab test",
    "test",
    "lift",
}
AB_KEYWORDS = {"experiment", "treatment", "control", "variant", "a/b", "ab test", "lift", "conversion"}
REGRESSION_KEYWORDS = {"regression", "predict", "predictor", "predictors", "association", "associated", "relationship"}
PREFERRED_METRICS = [
    "revenue",
    "sales",
    "amount",
    "profit",
    "gmv",
    "value",
    "price",
]
PREFERRED_SEGMENTS = [
    "country",
    "region",
    "market",
    "product_category",
    "category",
    "segment",
    "channel",
    "customer_id",
]
DRIVER_COLS = ["quantity", "unit_price", "discount_pct", "discount", "price", "units"]
AB_GROUP_COL_CANDIDATES = ["variant", "experiment_group", "treatment_group", "treatment", "group", "arm", "bucket"]
PAIR_ID_CANDIDATES = ["customer_id", "user_id", "account_id", "session_id", "order_id"]
CONVERSION_COL_CANDIDATES = ["converted", "conversion", "is_conversion", "converted_flag", "purchased", "clicked"]
ALLOWED_TASK_TYPES = {
    "groupby_agg",
    "groupby2_agg",
    "filter_agg",
    "correlation",
    "distribution",
    "group_distribution",
    "recency_by_group",
    "topk",
    "timeseries_agg",
    "sql_query",
    "sql_join_profile",
    "kpi_template_apply",
    "metric_definition",
    "segment_analysis",
    "cohort_analysis",
    "statistical_test",
    "ab_test",
    "ols_regression",
}

SUPPORTED_ANALYSIS_TYPES = {
    "descriptive",
    "trend",
    "segment_comparison",
    "diagnostic",
    "experiment_ab",
    "forecasting_unsupported",
    "impossible",
}

ALLOWED_TASKS_BY_ANALYSIS_TYPE: Dict[str, set[str]] = {
    "descriptive": {
        "groupby_agg",
        "groupby2_agg",
        "distribution",
        "topk",
        "kpi_template_apply",
        "segment_analysis",
        "metric_definition",
        "sql_query",
        "sql_join_profile",
    },
    "trend": {
        "timeseries_agg",
        "groupby_agg",
        "distribution",
        "kpi_template_apply",
        "segment_analysis",
        "cohort_analysis",
        "sql_query",
        "sql_join_profile",
    },
    "segment_comparison": {
        "groupby_agg",
        "groupby2_agg",
        "filter_agg",
        "group_distribution",
        "topk",
        "distribution",
        "kpi_template_apply",
        "segment_analysis",
        "sql_query",
        "sql_join_profile",
    },
    "diagnostic": {
        "groupby_agg",
        "groupby2_agg",
        "filter_agg",
        "correlation",
        "distribution",
        "group_distribution",
        "timeseries_agg",
        "kpi_template_apply",
        "segment_analysis",
        "metric_definition",
        "statistical_test",
        "ols_regression",
        "sql_query",
        "sql_join_profile",
    },
    "experiment_ab": {
        "ab_test",
        "statistical_test",
        "groupby_agg",
        "distribution",
        "kpi_template_apply",
        "segment_analysis",
        "sql_query",
        "sql_join_profile",
    },
    "forecasting_unsupported": {
        "timeseries_agg",
        "groupby_agg",
        "distribution",
        "kpi_template_apply",
        "segment_analysis",
        "cohort_analysis",
        "sql_query",
        "sql_join_profile",
    },
    "impossible": {
        "kpi_template_apply",
        "distribution",
        "groupby_agg",
    },
}


def _is_numeric_dtype(dtype: str) -> bool:
    d = (dtype or "").lower()
    return any(token in d for token in ["int", "float", "double", "decimal", "number"])


def _is_datetime_dtype(dtype: str) -> bool:
    d = (dtype or "").lower()
    return any(token in d for token in ["datetime", "timestamp", "date", "time"])


def _contains_any(text: str, words: set[str]) -> bool:
    for w in words:
        w_norm = w.strip().lower()
        if not w_norm:
            continue
        if " " in w_norm or "-" in w_norm:
            if w_norm in text:
                return True
            continue
        if re.search(rf"\b{re.escape(w_norm)}\b", text):
            return True
    return False


def _repair_json_candidate(text: str) -> str:
    repaired = re.sub(r",\s*([}\]])", r"\1", text.strip())
    repaired = repaired.replace("“", "\"").replace("”", "\"")
    repaired = repaired.replace("’", "'").replace("‘", "'")
    return repaired


def _extract_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("LLM returned empty response (no JSON to parse).")

    for candidate in [raw]:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        candidate = _repair_json_candidate(m.group(1))
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    m = re.search(r"(\{.*\})", raw, re.DOTALL)
    if m:
        candidate = _repair_json_candidate(m.group(1))
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Could not extract JSON from LLM output.\nRaw output:\n{raw}")


def _add_task(tasks: List[Dict[str, Any]], seen: set[str], ttype: str, params: Dict[str, Any]) -> None:
    key = json.dumps({"type": ttype, "params": params}, sort_keys=True, ensure_ascii=False)
    if key in seen:
        return
    seen.add(key)
    tasks.append({"type": ttype, "params": params})


def _extract_focus_value(question: str, schema_cols: List[str], column_profiles: List[Dict[str, Any]]) -> str | None:
    q = question.lower()

    for prof in column_profiles:
        col = str(prof.get("name", ""))
        if col.lower() not in {"country", "region", "market", "segment", "channel"}:
            continue
        for ex in prof.get("example_values", []) or []:
            exs = str(ex).strip()
            if exs and exs.lower() in q:
                return exs

    return None


def _choose_metric(question: str, schema_cols: List[str], numeric_cols: List[str]) -> str | None:
    q = question.lower()
    for col in schema_cols:
        if col.lower() in q and col in numeric_cols:
            return col

    for name in PREFERRED_METRICS:
        if name in schema_cols and name in numeric_cols:
            return name

    return numeric_cols[0] if numeric_cols else None


def _choose_primary_dim(question: str, schema_cols: List[str], categorical_cols: List[str]) -> str | None:
    q = question.lower()
    priority: List[str] = []

    if _contains_any(q, CUSTOMER_KEYWORDS):
        priority.extend(["customer_id", "customer", "customer_name"])
    if _contains_any(q, {"country", "region", "market", "geography", "geo"}):
        priority.extend(["country", "region", "market"])
    if _contains_any(q, PRODUCT_KEYWORDS):
        priority.extend(["product_category", "category", "product", "sku"])
    if _contains_any(q, {"channel", "segment"}):
        priority.extend(["channel", "segment"])

    priority.extend(PREFERRED_SEGMENTS)
    for col in priority:
        if col in schema_cols:
            return col
    return categorical_cols[0] if categorical_cols else None


def _choose_secondary_dim(question: str, schema_cols: List[str], primary_dim: str | None) -> str | None:
    q = question.lower()
    candidates: List[str] = []
    if _contains_any(q, PRODUCT_KEYWORDS):
        candidates.extend(["product_category", "category", "product"])
    if _contains_any(q, CUSTOMER_KEYWORDS):
        candidates.extend(["customer_id"])
    if _contains_any(q, {"country", "region", "market"}):
        candidates.extend(["country", "region", "market"])

    candidates.extend(["product_category", "country", "channel", "segment", "customer_id"])
    for c in candidates:
        if c in schema_cols and c != primary_dim:
            return c
    return None


def _choose_experiment_group_col(
    schema_cols: List[str],
    column_profiles: List[Dict[str, Any]],
) -> str | None:
    for cand in AB_GROUP_COL_CANDIDATES:
        if cand in schema_cols:
            return cand
    for prof in column_profiles:
        name = str(prof.get("name", ""))
        examples = {str(v).strip().lower() for v in prof.get("example_values", []) or []}
        if {"control", "treatment"} & examples or {"a", "b"} <= examples:
            if name in schema_cols:
                return name
    return None


def _extract_experiment_labels(column_profiles: List[Dict[str, Any]], group_col: str | None) -> tuple[str | None, str | None]:
    if not group_col:
        return None, None
    for prof in column_profiles:
        if str(prof.get("name")) != group_col:
            continue
        values = [str(v).strip() for v in (prof.get("example_values") or []) if str(v).strip()]
        values_lc = [v.lower() for v in values]
        if "control" in values_lc and "treatment" in values_lc:
            return values[values_lc.index("control")], values[values_lc.index("treatment")]
        if "a" in values_lc and "b" in values_lc:
            return values[values_lc.index("a")], values[values_lc.index("b")]
        if len(values) >= 2:
            return values[0], values[1]
    return None, None


def _choose_binary_metric(schema_cols: List[str], question: str, numeric_cols: List[str]) -> str | None:
    q = question.lower()
    for cand in CONVERSION_COL_CANDIDATES:
        if cand in schema_cols:
            return cand
    for col in schema_cols:
        if col.lower() in q and any(tok in col.lower() for tok in ["convert", "click", "purchase", "signup", "churn"]):
            return col
    for col in numeric_cols:
        if any(tok in col.lower() for tok in ["flag", "binary", "indicator"]):
            return col
    return None


def _choose_regression_predictors(numeric_cols: List[str], target: str | None) -> List[str]:
    out: List[str] = []
    for cand in DRIVER_COLS + ["discount_pct", "discount", "customer_age", "tenure_days", "sessions"]:
        if cand in numeric_cols and cand != target and cand not in out:
            out.append(cand)
    for col in numeric_cols:
        if col != target and col not in out:
            out.append(col)
        if len(out) >= 4:
            break
    return out[:4]


def _classify_analysis_type(
    question: str,
    *,
    metric: str | None,
    time_col: str | None,
    segment_cols: List[str],
    experiment_group_col: str | None,
) -> tuple[str, List[str], str]:
    q = (question or "").lower()
    blocked: List[str] = []

    if _contains_any(q, {"forecast", "predict", "projection", "next month", "next quarter", "next year"}):
        if not time_col:
            blocked.append("missing_time_column")
        if not metric:
            blocked.append("missing_numeric_metric")
        return "forecasting_unsupported", blocked, "Forecast-like request; pipeline is limited to historical analysis."

    if _contains_any(q, AB_KEYWORDS):
        if not experiment_group_col:
            blocked.append("missing_experiment_group_column")
        if not metric:
            blocked.append("missing_numeric_metric")
        if blocked:
            return "impossible", blocked, "Experiment intent detected but required columns are unavailable."
        return "experiment_ab", blocked, "Experiment intent detected."

    if _contains_any(q, REGRESSION_KEYWORDS):
        if not metric:
            blocked.append("missing_numeric_metric")
        return "diagnostic", blocked, "Regression/association intent detected."

    if _contains_any(q, WHY_KEYWORDS):
        if not segment_cols:
            blocked.append("missing_segment_column")
        if not metric:
            blocked.append("missing_numeric_metric")
        return "diagnostic", blocked, "Diagnostic/driver intent detected."

    if _contains_any(q, TIME_KEYWORDS):
        if not time_col:
            blocked.append("missing_time_column")
            return "impossible", blocked, "Trend intent detected but no time column is available."
        if not metric:
            blocked.append("missing_numeric_metric")
        return "trend", blocked, "Trend/time intent detected."

    if _contains_any(q, COMPARE_KEYWORDS):
        if not segment_cols:
            blocked.append("missing_segment_column")
            return "impossible", blocked, "Comparison intent detected but no segment column is available."
        if not metric:
            blocked.append("missing_numeric_metric")
        return "segment_comparison", blocked, "Segment comparison intent detected."

    if not metric:
        blocked.append("missing_numeric_metric")
    return "descriptive", blocked, "Default descriptive route."


def _route_comparison_logic(
    analysis_type: str,
    *,
    metric: str | None,
    time_col: str | None,
    segment_cols: List[str],
    experiment_group_col: str | None,
) -> str:
    if analysis_type == "experiment_ab":
        return (
            f"Compare treatment vs control on {metric} using {experiment_group_col}; "
            "estimate lift and significance."
        )
    if analysis_type == "trend":
        return f"Evaluate period-over-period movement of {metric} on {time_col}."
    if analysis_type == "segment_comparison":
        seg = segment_cols[0] if segment_cols else "segments"
        return f"Compare {metric} across {seg} with absolute and relative deltas."
    if analysis_type == "diagnostic":
        seg = segment_cols[0] if segment_cols else "major dimensions"
        return f"Decompose {metric} by {seg}, volume, and mix to identify drivers."
    if analysis_type == "forecasting_unsupported":
        return f"Provide historical baseline for {metric}; do not forecast future values."
    if analysis_type == "impossible":
        return "Return infeasibility summary with missing requirements."
    return f"Summarize overall {metric} and key segment breakdowns."


def _route_success_criterion(analysis_type: str) -> str:
    if analysis_type == "experiment_ab":
        return "Report lift and p-value at alpha=0.05 with valid control/treatment groups."
    if analysis_type == "trend":
        return "Report trend direction and magnitude with sufficient time periods."
    if analysis_type == "segment_comparison":
        return "Report ranked segment deltas with absolute and percentage difference."
    if analysis_type == "diagnostic":
        return "Report top contributors with artifact-backed evidence."
    if analysis_type == "forecasting_unsupported":
        return "Report historical trend only and explicitly flag forecasting as unsupported."
    if analysis_type == "impossible":
        return "Return a precise infeasibility explanation with missing requirements."
    return "Return descriptive KPIs grounded in computed artifacts."


def _build_planning_contract(
    *,
    question: str,
    schema_cols: List[str],
    date_cols: List[str],
    metric_hint: str | None,
    primary_dim_hint: str | None,
    secondary_dim_hint: str | None,
    analysis_type: str,
    blocked_requirements: List[str],
    intake_plan: Dict[str, Any] | None,
    experiment_group_col: str | None,
    control_label: str | None,
    treatment_label: str | None,
) -> Dict[str, Any]:
    intake = dict((intake_plan or {}).get("framing", {}) or {})

    target_metric = intake.get("target_metric")
    if target_metric not in schema_cols:
        target_metric = metric_hint if metric_hint in schema_cols else None

    time_col = intake.get("time_column")
    if time_col not in schema_cols:
        time_col = date_cols[0] if date_cols else None

    segment_columns = intake.get("segment_columns")
    if not isinstance(segment_columns, list):
        segment_columns = []
    segment_columns = [str(c) for c in segment_columns if str(c) in schema_cols]
    if not segment_columns:
        for c in [primary_dim_hint, secondary_dim_hint]:
            if c and c in schema_cols and c not in segment_columns:
                segment_columns.append(c)

    aggregation_level = str(intake.get("aggregation_level", "")).strip()
    if not aggregation_level:
        if analysis_type == "trend" and time_col:
            aggregation_level = f"time:{time_col}"
        elif analysis_type == "experiment_ab" and experiment_group_col:
            aggregation_level = f"group:{experiment_group_col}"
        elif segment_columns:
            aggregation_level = f"segment:{segment_columns[0]}"
        else:
            aggregation_level = "overall"

    metric_aggregation = default_agg_for_metric(str(target_metric or "metric"), preferred=intake.get("metric_aggregation"))
    comparison_logic = str(intake.get("comparison_logic", "")).strip() or _route_comparison_logic(
        analysis_type,
        metric=target_metric,
        time_col=time_col,
        segment_cols=segment_columns,
        experiment_group_col=experiment_group_col,
    )
    success_criterion = str(intake.get("success_criterion", "")).strip() or _route_success_criterion(analysis_type)

    limitations = intake.get("analysis_limitations")
    if not isinstance(limitations, list):
        limitations = []
    for req in blocked_requirements:
        if req == "missing_time_column":
            limitations.append("No usable time column is available for time-based analysis.")
        elif req == "missing_segment_column":
            limitations.append("No suitable segment column is available for segment comparison.")
        elif req == "missing_numeric_metric":
            limitations.append("No numeric metric candidate is available for quantitative analysis.")
        elif req == "missing_experiment_group_column":
            limitations.append("No experiment group column (control/treatment) is available.")
    if analysis_type == "forecasting_unsupported":
        limitations.append("Forecasting is unsupported; only historical analysis is provided.")
    if analysis_type == "impossible":
        limitations.append("Question cannot be fully answered with available data constraints.")
    if not limitations:
        limitations.append("All conclusions are limited to provided columns and computed artifacts.")
    limitations = list(dict.fromkeys(str(x) for x in limitations))

    feasibility_status = str(intake.get("feasibility_status", "")).strip()
    if not feasibility_status:
        feasibility_status = "infeasible" if analysis_type == "impossible" else ("partially_feasible" if blocked_requirements else "feasible")

    contract = {
        "analysis_type": analysis_type,
        "target_metric": target_metric,
        "aggregation_level": aggregation_level,
        "metric_aggregation": metric_aggregation,
        "time_column": time_col,
        "segment_columns": segment_columns,
        "comparison_logic": comparison_logic,
        "success_criterion": success_criterion,
        "analysis_limitations": limitations,
        "feasibility_status": feasibility_status,
        "experiment_group_column": experiment_group_col,
        "experiment_control_label": control_label,
        "experiment_treatment_label": treatment_label,
    }
    return contract


def _route_seed_tasks(planning_contract: Dict[str, Any], business_domain: str) -> List[Dict[str, Any]]:
    analysis_type = str(planning_contract.get("analysis_type", "descriptive"))
    metric = planning_contract.get("target_metric")
    agg = str(planning_contract.get("metric_aggregation", "sum"))
    time_col = planning_contract.get("time_column")
    segments = planning_contract.get("segment_columns") or []
    group_col = planning_contract.get("experiment_group_column")
    control = planning_contract.get("experiment_control_label") or "control"
    treatment = planning_contract.get("experiment_treatment_label") or "treatment"

    seeded: List[Dict[str, Any]] = [{"type": "kpi_template_apply", "params": {"domain": business_domain}}]
    if metric:
        seeded.append({"type": "distribution", "params": {"column": metric, "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95]}})
    if segments and metric:
        seeded.append(
            {
                "type": "segment_analysis",
                "params": {"segment_by": segments[0], "metric": metric, "agg": agg, "limit": 100},
            }
        )
    if analysis_type in {"trend", "forecasting_unsupported"} and time_col and metric:
        seeded.append({"type": "timeseries_agg", "params": {"date_col": time_col, "metric": metric, "freq": "M", "agg": agg}})
    if analysis_type == "segment_comparison" and segments and metric:
        seeded.append({"type": "groupby_agg", "params": {"group_by": segments[0], "metric": metric, "agg": agg, "limit": 1000}})
    if analysis_type == "experiment_ab" and group_col and metric:
        seeded.append(
            {
                "type": "ab_test",
                "params": {
                    "group_col": group_col,
                    "control": control,
                    "treatment": treatment,
                    "metric": metric,
                    "metric_type": "auto",
                    "alpha": 0.05,
                },
            }
        )
    return seeded


def _filter_tasks_by_analysis_type(tasks: List[Dict[str, Any]], analysis_type: str) -> List[Dict[str, Any]]:
    allowed = ALLOWED_TASKS_BY_ANALYSIS_TYPE.get(analysis_type, ALLOWED_TASK_TYPES)
    return [t for t in tasks if str(t.get("type", "")).strip() in allowed]


def _resolve_metric_agg(metric: str, agg: str | None, *, fallback: str | None = None) -> str:
    candidate = str(agg or "").strip().lower() if agg is not None else ""
    if candidate and is_agg_allowed_for_metric(metric, candidate):
        return candidate
    return default_agg_for_metric(metric, preferred=fallback or candidate or None)


def _sanitize_and_number_tasks(
    tasks: List[Dict[str, Any]],
    schema_cols: List[str],
    numeric_cols: List[str],
    planning_contract: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    clean: List[Dict[str, Any]] = []
    contract_metric = str((planning_contract or {}).get("target_metric") or "")
    contract_agg = str((planning_contract or {}).get("metric_aggregation") or "").strip().lower()

    for task in tasks:
        ttype = str(task.get("type", "")).strip()
        p = dict(task.get("params", {}) or {})

        if ttype not in ALLOWED_TASK_TYPES:
            continue

        if ttype == "groupby_agg":
            if p.get("group_by") not in schema_cols or p.get("metric") not in schema_cols:
                continue
            metric = str(p.get("metric") or contract_metric)
            p["agg"] = _resolve_metric_agg(metric, p.get("agg"), fallback=contract_agg or "sum")
            p.setdefault("limit", 50)

        elif ttype == "groupby2_agg":
            if p.get("group_by_1") not in schema_cols or p.get("group_by_2") not in schema_cols:
                continue
            if p.get("metric") not in schema_cols:
                continue
            metric = str(p.get("metric") or contract_metric)
            p["agg"] = _resolve_metric_agg(metric, p.get("agg"), fallback=contract_agg or "sum")
            p.setdefault("limit", 100)

        elif ttype == "filter_agg":
            if p.get("filter_col") not in schema_cols or p.get("metric") not in schema_cols:
                continue
            if "filter_val" not in p:
                continue
            metric = str(p.get("metric") or contract_metric)
            p["agg"] = _resolve_metric_agg(metric, p.get("agg"), fallback=contract_agg or "sum")

        elif ttype == "correlation":
            if p.get("x") not in numeric_cols or p.get("y") not in numeric_cols:
                continue
            if p.get("x") == p.get("y"):
                continue

        elif ttype == "distribution":
            if p.get("column") not in numeric_cols:
                continue
            p.setdefault("quantiles", [0.05, 0.25, 0.5, 0.75, 0.95])

        elif ttype == "group_distribution":
            if p.get("group_by") not in schema_cols:
                continue
            if p.get("metric") not in schema_cols:
                continue
            metric = str(p.get("metric") or contract_metric)
            p["agg"] = _resolve_metric_agg(metric, p.get("agg"), fallback=contract_agg or "sum")
            p.setdefault("quantiles", [0.5, 0.75, 0.9, 0.95, 0.99])

        elif ttype == "recency_by_group":
            if p.get("group_by") not in schema_cols:
                continue
            if p.get("date_col") not in schema_cols:
                continue

        elif ttype == "topk":
            if p.get("by") not in schema_cols or p.get("metric") not in numeric_cols:
                continue
            metric = str(p.get("metric") or contract_metric)
            p["agg"] = _resolve_metric_agg(metric, p.get("agg"), fallback=contract_agg or "sum")
            p.setdefault("k", 10)

        elif ttype == "timeseries_agg":
            if p.get("date_col") not in schema_cols or p.get("metric") not in schema_cols:
                continue
            p.setdefault("freq", "M")
            metric = str(p.get("metric") or contract_metric)
            p["agg"] = _resolve_metric_agg(metric, p.get("agg"), fallback=contract_agg or "sum")

        elif ttype == "sql_query":
            query = str(p.get("query", "")).strip()
            if not query:
                continue
            try:
                p["query"] = validate_read_only_sql(query)
            except Exception:
                continue
            p.setdefault("limit", 1000)
            p.setdefault("output", "rows")
            try:
                p["limit"] = int(p.get("limit", 1000))
            except Exception:
                p["limit"] = 1000

        elif ttype == "sql_join_profile":
            fact = str(p.get("fact_table", "")).strip()
            if not fact:
                continue
            if "dimension_table" in p and p.get("dimension_table") is not None:
                p["dimension_table"] = str(p["dimension_table"]).strip()

        elif ttype == "kpi_template_apply":
            domain = str(p.get("domain", "")).strip().lower()
            if not domain:
                continue
            p["domain"] = domain

        elif ttype == "metric_definition":
            name = str(p.get("name", "")).strip()
            metric_col = str(p.get("metric_col", "")).strip() if p.get("metric_col") is not None else ""
            expression = str(p.get("expression", "")).strip() if p.get("expression") is not None else ""
            if not name:
                continue
            if not metric_col and not expression:
                continue
            if metric_col:
                p["agg"] = _resolve_metric_agg(metric_col, p.get("agg"), fallback=contract_agg or "sum")
            else:
                p.setdefault("agg", contract_agg or "sum")
            if metric_col and metric_col not in schema_cols:
                continue
            if p.get("group_by") and p.get("group_by") not in schema_cols:
                continue

        elif ttype == "segment_analysis":
            if p.get("segment_by") not in schema_cols:
                continue
            if p.get("metric") not in schema_cols:
                continue
            metric = str(p.get("metric") or contract_metric)
            p["agg"] = _resolve_metric_agg(metric, p.get("agg"), fallback=contract_agg or "sum")
            p.setdefault("limit", 100)

        elif ttype == "cohort_analysis":
            if p.get("entity_col") not in schema_cols:
                continue
            if p.get("date_col") not in schema_cols:
                continue
            p.setdefault("freq", "M")

        elif ttype == "statistical_test":
            if p.get("group_col") not in schema_cols:
                continue
            if p.get("metric") not in schema_cols:
                continue
            if p.get("pair_id_col") and p.get("pair_id_col") not in schema_cols:
                continue
            p.setdefault("alpha", 0.05)
            p.setdefault("alternative", "two-sided")
            p["compare_to_rest"] = bool(p.get("compare_to_rest", False))
            p["paired"] = bool(p.get("paired", False))

        elif ttype == "ab_test":
            if p.get("group_col") not in schema_cols:
                continue
            if p.get("metric") not in schema_cols:
                continue
            if "control" not in p or "treatment" not in p:
                continue
            p.setdefault("metric_type", "auto")
            p.setdefault("alpha", 0.05)

        elif ttype == "ols_regression":
            if p.get("target") not in numeric_cols:
                continue
            predictors = p.get("predictors", [])
            if not isinstance(predictors, list):
                continue
            predictors = [str(x) for x in predictors if str(x) in schema_cols and str(x) != str(p["target"])]
            if not predictors:
                continue
            p["predictors"] = predictors
            p.setdefault("alpha", 0.05)

        clean.append({"type": ttype, "params": p})

    out: List[Dict[str, Any]] = []
    for i, task in enumerate(clean, start=1):
        out.append({"id": f"T{i}", "type": task["type"], "params": task["params"]})
    return out


def _build_heuristic_tasks(
    question: str,
    schema_cols: List[str],
    dtypes: Dict[str, Any],
    datetime_candidates: List[str],
    column_profiles: List[Dict[str, Any]],
    business_domain: str,
) -> List[Dict[str, Any]]:
    q = question.lower()
    numeric_cols = [c for c in schema_cols if _is_numeric_dtype(str(dtypes.get(c, "")))]
    date_cols = [c for c in schema_cols if _is_datetime_dtype(str(dtypes.get(c, "")))]
    for c in datetime_candidates:
        if c in schema_cols and c not in date_cols:
            date_cols.append(c)

    categorical_cols = [c for c in schema_cols if c not in numeric_cols]

    metric = _choose_metric(question, schema_cols, numeric_cols)
    if metric is None and numeric_cols:
        metric = numeric_cols[0]
    if metric is None:
        first_col = schema_cols[0] if schema_cols else "value"
        return [{"type": "distribution", "params": {"column": first_col}}]

    primary_dim = _choose_primary_dim(question, schema_cols, categorical_cols)
    secondary_dim = _choose_secondary_dim(question, schema_cols, primary_dim)

    why_intent = _contains_any(q, WHY_KEYWORDS) or _contains_any(q, COMPARE_KEYWORDS)
    total_intent = _contains_any(q, TOTAL_KEYWORDS)
    count_intent = _contains_any(q, COUNT_KEYWORDS)
    customer_intent = _contains_any(q, CUSTOMER_KEYWORDS)
    product_intent = _contains_any(q, PRODUCT_KEYWORDS)
    time_intent = _contains_any(q, TIME_KEYWORDS)
    segment_intent = _contains_any(q, SEGMENT_KEYWORDS) or customer_intent
    stat_intent = _contains_any(q, STAT_KEYWORDS)
    ab_intent = _contains_any(q, AB_KEYWORDS)
    regression_intent = _contains_any(q, REGRESSION_KEYWORDS)

    tasks: List[Dict[str, Any]] = []
    seen: set[str] = set()
    base_limit = 0 if customer_intent and "customer_id" in schema_cols else 1000
    mix_limit = 0 if customer_intent else 5000
    count_metric = "order_id" if "order_id" in schema_cols else ("transaction_id" if "transaction_id" in schema_cols else None)
    main_metric = count_metric if count_intent and count_metric else metric
    main_agg = "count" if count_intent and count_metric else "sum"

    focus_value = _extract_focus_value(question, schema_cols, column_profiles)
    if focus_value and primary_dim:
        _add_task(
            tasks,
            seen,
            "filter_agg",
            {"filter_col": primary_dim, "filter_val": focus_value, "metric": main_metric, "agg": main_agg},
        )

    if primary_dim:
        _add_task(
            tasks,
            seen,
            "groupby_agg",
            {"group_by": primary_dim, "metric": main_metric, "agg": main_agg, "limit": base_limit},
        )

    if secondary_dim:
        _add_task(
            tasks,
            seen,
            "groupby_agg",
            {"group_by": secondary_dim, "metric": main_metric, "agg": main_agg, "limit": base_limit},
        )

    if (why_intent or segment_intent) and primary_dim:
        _add_task(
            tasks,
            seen,
            "groupby_agg",
            {"group_by": primary_dim, "metric": metric, "agg": "count", "limit": base_limit},
        )
        _add_task(
            tasks,
            seen,
            "groupby_agg",
            {"group_by": primary_dim, "metric": metric, "agg": "mean", "limit": base_limit},
        )

    if (why_intent or segment_intent or product_intent) and primary_dim and secondary_dim:
        _add_task(
            tasks,
            seen,
            "groupby2_agg",
            {
                "group_by_1": primary_dim,
                "group_by_2": secondary_dim,
                "metric": main_metric,
                "agg": main_agg,
                "limit": mix_limit,
            },
        )

    _add_task(
        tasks,
        seen,
        "distribution",
        {"column": metric, "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95]},
    )

    if segment_intent and primary_dim:
        _add_task(
            tasks,
            seen,
            "group_distribution",
            {
                "group_by": primary_dim,
                "metric": main_metric,
                "agg": main_agg,
                "quantiles": [0.5, 0.75, 0.9, 0.95, 0.99],
            },
        )

    if not count_intent or why_intent:
        for driver in DRIVER_COLS:
            if driver in numeric_cols and driver != metric:
                _add_task(tasks, seen, "correlation", {"x": driver, "y": metric})
            if len([t for t in tasks if t["type"] == "correlation"]) >= 3:
                break

    if primary_dim and "quantity" in numeric_cols and "quantity" != metric and (why_intent or segment_intent):
        _add_task(
            tasks,
            seen,
            "groupby_agg",
            {"group_by": primary_dim, "metric": "quantity", "agg": "sum", "limit": base_limit},
        )

    discount_col = "discount_pct" if "discount_pct" in numeric_cols else None
    if primary_dim and discount_col and (why_intent or segment_intent):
        _add_task(
            tasks,
            seen,
            "groupby_agg",
            {"group_by": primary_dim, "metric": discount_col, "agg": "mean", "limit": base_limit},
        )

    if date_cols and (time_intent or why_intent or customer_intent):
        _add_task(
            tasks,
            seen,
            "timeseries_agg",
            {"date_col": date_cols[0], "metric": metric, "freq": "M", "agg": "sum"},
        )
        if customer_intent and "customer_id" in schema_cols:
            _add_task(
                tasks,
                seen,
                "recency_by_group",
                {"group_by": "customer_id", "date_col": date_cols[0]},
            )

    if customer_intent and "customer_id" in schema_cols:
        _add_task(
            tasks,
            seen,
            "topk",
            {"by": "customer_id", "metric": main_metric, "agg": main_agg, "k": 100},
        )

    if not why_intent and not total_intent and primary_dim and not customer_intent:
        _add_task(tasks, seen, "topk", {"by": primary_dim, "metric": main_metric, "agg": main_agg, "k": 25})

    # Phase 3 business context tasks
    _add_task(tasks, seen, "kpi_template_apply", {"domain": business_domain})
    if primary_dim:
        _add_task(
            tasks,
            seen,
            "segment_analysis",
            {"segment_by": primary_dim, "metric": main_metric, "agg": main_agg, "limit": 100},
        )

    entity_col, cohort_date_col = pick_cohort_columns(business_domain, schema_cols)
    if entity_col and cohort_date_col and (_contains_any(q, TIME_KEYWORDS) or customer_intent or segment_intent):
        _add_task(
            tasks,
            seen,
            "cohort_analysis",
            {"entity_col": entity_col, "date_col": cohort_date_col, "freq": "M"},
        )

    experiment_group_col = _choose_experiment_group_col(schema_cols, column_profiles)
    control_label, treatment_label = _extract_experiment_labels(column_profiles, experiment_group_col)
    binary_metric = _choose_binary_metric(schema_cols, question, numeric_cols)
    if ab_intent and experiment_group_col and metric:
        _add_task(
            tasks,
            seen,
            "ab_test",
            {
                "group_col": experiment_group_col,
                "control": control_label or "control",
                "treatment": treatment_label or "treatment",
                "metric": binary_metric or metric,
                "metric_type": "binary" if binary_metric else "continuous",
                "success_value": 1,
                "alpha": 0.05,
            },
        )

    if stat_intent and primary_dim and metric:
        stat_params: Dict[str, Any] = {
            "group_col": primary_dim,
            "metric": binary_metric or metric,
            "alpha": 0.05,
            "alternative": "two-sided",
        }
        if focus_value is not None:
            stat_params["group_a"] = focus_value
            stat_params["compare_to_rest"] = True
        _add_task(tasks, seen, "statistical_test", stat_params)

    if regression_intent and metric:
        predictors = _choose_regression_predictors(numeric_cols, metric)
        if predictors:
            _add_task(
                tasks,
                seen,
                "ols_regression",
                {"target": metric, "predictors": predictors, "alpha": 0.05},
            )

    return tasks


def _choose_sql_metric(sql_schema: Dict[str, Any], fallback_metric: str | None = None) -> str | None:
    idx = column_table_index(sql_schema)
    if fallback_metric and fallback_metric in idx:
        return fallback_metric
    for name in PREFERRED_METRICS:
        if name in idx:
            return name
    numeric_like = {"amount", "price", "revenue", "sales", "profit", "quantity", "units", "cost", "spend"}
    for col in idx.keys():
        if any(tok in col.lower() for tok in numeric_like):
            return col
    return next(iter(idx.keys()), None)


def _choose_sql_dim(
    question: str,
    sql_schema: Dict[str, Any],
    fallback_dim: str | None = None,
    exclude: set[str] | None = None,
) -> str | None:
    exclude = exclude or set()
    idx = column_table_index(sql_schema)
    q = question.lower()
    candidates: List[str] = []
    if _contains_any(q, {"country", "region", "market"}):
        candidates.extend(["country", "region", "market"])
    if _contains_any(q, PRODUCT_KEYWORDS):
        candidates.extend(["product_category", "category", "product"])
    if _contains_any(q, {"channel", "segment"}):
        candidates.extend(["channel", "segment"])
    if _contains_any(q, CUSTOMER_KEYWORDS):
        candidates.extend(["customer_id"])
    candidates.extend(PREFERRED_SEGMENTS)
    for c in candidates:
        if c in idx and c not in exclude:
            return c
    if fallback_dim and fallback_dim in idx and fallback_dim not in exclude:
        return fallback_dim
    for col in idx.keys():
        if col not in exclude:
            return col
    return None


def _build_sql_heuristic_tasks(
    *,
    question: str,
    sql_schema: Dict[str, Any],
    sql_engine: Any,
    metric_hint: str | None,
    primary_dim_hint: str | None,
    secondary_dim_hint: str | None,
    business_domain: str,
    main_agg: str,
) -> List[Dict[str, Any]]:
    if not isinstance(sql_schema, dict) or not (sql_schema.get("tables") or []):
        return []

    idx = column_table_index(sql_schema)
    if not idx:
        return []

    q = question.lower()
    why_intent = _contains_any(q, WHY_KEYWORDS) or _contains_any(q, COMPARE_KEYWORDS)
    segment_intent = _contains_any(q, SEGMENT_KEYWORDS) or _contains_any(q, CUSTOMER_KEYWORDS)
    sql_intent = _contains_any(q, SQL_KEYWORDS)

    fact_table = choose_primary_table(sql_schema)
    metric = _choose_sql_metric(sql_schema, metric_hint)
    primary_dim = _choose_sql_dim(question, sql_schema, primary_dim_hint)
    secondary_dim = _choose_sql_dim(question, sql_schema, secondary_dim_hint, exclude={str(primary_dim)})

    tasks: List[Dict[str, Any]] = []
    seen: set[str] = set()

    if fact_table:
        _add_task(tasks, seen, "sql_join_profile", {"fact_table": fact_table})

    if metric and primary_dim and fact_table:
        try:
            q1 = build_groupby_query(
                engine=sql_engine,
                schema=sql_schema,
                metric_col=metric,
                group_cols=[primary_dim],
                agg=main_agg,
                limit=1000,
                preferred_fact_table=fact_table,
            )
        except Exception:
            q1 = None
        if q1:
            _add_task(
                tasks,
                seen,
                "sql_query",
                {"query": q1["query"], "output": "mapping", "limit": 1000},
            )
            joined_tables = list(q1.get("joined_tables") or [])
            if len(joined_tables) > 1:
                _add_task(
                    tasks,
                    seen,
                    "sql_join_profile",
                    {"fact_table": joined_tables[0], "dimension_table": joined_tables[-1]},
                )

    if metric and primary_dim and secondary_dim and fact_table and (why_intent or segment_intent or sql_intent):
        try:
            q2 = build_groupby_query(
                engine=sql_engine,
                schema=sql_schema,
                metric_col=metric,
                group_cols=[primary_dim, secondary_dim],
                agg=main_agg,
                limit=5000,
                preferred_fact_table=fact_table,
            )
        except Exception:
            q2 = None
        if q2:
            _add_task(
                tasks,
                seen,
                "sql_query",
                {"query": q2["query"], "output": "rows", "limit": 5000},
            )

    _add_task(tasks, seen, "kpi_template_apply", {"domain": business_domain})
    if primary_dim and metric:
        _add_task(
            tasks,
            seen,
            "segment_analysis",
            {"segment_by": primary_dim, "metric": metric, "agg": main_agg, "limit": 200},
        )

    return tasks


def _merge_task_lists(first: List[Dict[str, Any]], second: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for task in [*first, *second]:
        ttype = str(task.get("type", "")).strip()
        params = dict(task.get("params", {}) or {})
        key = json.dumps({"type": ttype, "params": params}, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        merged.append({"type": ttype, "params": params})
    return merged


class PlannerAgent(Agent):
    name = "planner"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        cfg = ctx["cfg"]
        store = ctx["store"]
        logger = ctx["logger"]
        question = ctx["business_question"]

        profile = ctx["memory"].get("result.profiling") or {}
        schema_cols = profile.get("columns", []) or []
        schema_dtypes = profile.get("dtypes", {}) or {}
        datetime_candidates = profile.get("datetime_candidates", []) or []
        column_profiles = profile.get("column_profiles", []) or []
        sql_schema = ctx.get("sql_schema") or profile.get("sql_schema")
        sql_source = ctx.get("sql_source")
        intake_plan = ctx["memory"].get("result.intake", {}) or {}

        numeric_cols = [c for c in schema_cols if _is_numeric_dtype(str(schema_dtypes.get(c, "")))]
        date_cols = [c for c in schema_cols if _is_datetime_dtype(str(schema_dtypes.get(c, "")))]
        for c in datetime_candidates:
            if c in schema_cols and c not in date_cols:
                date_cols.append(c)
        categorical_cols = [c for c in schema_cols if c not in numeric_cols]
        metric_hint = _choose_metric(question, schema_cols, numeric_cols)
        primary_dim_hint = _choose_primary_dim(question, schema_cols, categorical_cols)
        secondary_dim_hint = _choose_secondary_dim(question, schema_cols, primary_dim_hint)
        experiment_group_col = _choose_experiment_group_col(schema_cols, column_profiles)
        control_label, treatment_label = _extract_experiment_labels(column_profiles, experiment_group_col)
        q_lc = question.lower()
        count_intent = _contains_any(q_lc, COUNT_KEYWORDS)
        count_metric = "order_id" if "order_id" in schema_cols else ("transaction_id" if "transaction_id" in schema_cols else None)
        main_metric = count_metric if count_intent and count_metric else metric_hint
        main_agg = "count" if count_intent and count_metric else default_agg_for_metric(str(main_metric or "metric"))
        business_domain = detect_business_domain(question, schema_cols)

        intake_analysis_type = str(intake_plan.get("analysis_type", "")).strip()
        intake_blocked = intake_plan.get("blocked_requirements")
        blocked_requirements = [str(x) for x in intake_blocked] if isinstance(intake_blocked, list) else []
        if intake_analysis_type not in SUPPORTED_ANALYSIS_TYPES:
            analysis_type, blocked_requirements, routing_reason = _classify_analysis_type(
                question,
                metric=main_metric,
                time_col=date_cols[0] if date_cols else None,
                segment_cols=[x for x in [primary_dim_hint, secondary_dim_hint] if x],
                experiment_group_col=experiment_group_col,
            )
        else:
            analysis_type = intake_analysis_type
            routing_reason = str(intake_plan.get("routing_reason", "")).strip() or "Using intake route."

        planning_contract = _build_planning_contract(
            question=question,
            schema_cols=schema_cols,
            date_cols=date_cols,
            metric_hint=main_metric,
            primary_dim_hint=primary_dim_hint,
            secondary_dim_hint=secondary_dim_hint,
            analysis_type=analysis_type,
            blocked_requirements=blocked_requirements,
            intake_plan=intake_plan,
            experiment_group_col=experiment_group_col,
            control_label=control_label,
            treatment_label=treatment_label,
        )

        heuristic_tasks = _build_heuristic_tasks(
            question=question,
            schema_cols=schema_cols,
            dtypes=schema_dtypes,
            datetime_candidates=datetime_candidates,
            column_profiles=column_profiles,
            business_domain=business_domain,
        )
        sql_heuristic_tasks: List[Dict[str, Any]] = []
        if isinstance(sql_schema, dict) and sql_source is not None:
            try:
                sql_heuristic_tasks = _build_sql_heuristic_tasks(
                    question=question,
                    sql_schema=sql_schema,
                    sql_engine=sql_source.engine,
                    metric_hint=main_metric,
                    primary_dim_hint=primary_dim_hint,
                    secondary_dim_hint=secondary_dim_hint,
                    business_domain=business_domain,
                    main_agg=main_agg,
                )
            except Exception as e:
                logger.warning(f"[Planner] SQL heuristic planning failed: {e}")

        raw = ""
        llm_tasks: List[Dict[str, Any]] = []
        payload = {
            "business_question": question,
            "schema": {"columns": schema_cols, "dtypes": schema_dtypes},
            "source": ctx.get("source", {"type": "csv"}),
            "suggested_domain": business_domain,
            "analysis_type": analysis_type,
            "planning_contract": planning_contract,
            "sql_schema": sql_schema if isinstance(sql_schema, dict) else None,
        }

        try:
            client = OpenRouterClient(timeout_s=cfg.llm.timeout_s)
            logger.info("[Planner] Calling OpenRouter for task plan...")
            raw = client.chat(
                model=cfg.llm.model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": json.dumps(payload, indent=2)},
                ],
                temperature=0.0,
                max_tokens=max(1200, min(cfg.llm.max_tokens, 4096)),
            )
        except Exception as e:
            logger.warning(f"[Planner] OpenRouter call failed: {e}. Continuing with deterministic plan.")

        store.write_text("planner_raw.txt", raw or "")

        if raw and raw.strip():
            try:
                parsed = _extract_json(raw)
                llm_tasks = list((parsed or {}).get("tasks", []) or [])
            except Exception as e:
                logger.warning(f"[Planner] Invalid JSON from LLM: {e}. Using deterministic plan.")
        else:
            logger.warning("[Planner] Empty LLM response. Using deterministic plan.")

        seeded_tasks = _route_seed_tasks(planning_contract, business_domain)
        merged_tasks = _merge_task_lists(_merge_task_lists(heuristic_tasks, sql_heuristic_tasks), llm_tasks)
        route_filtered = _filter_tasks_by_analysis_type(merged_tasks, analysis_type)
        routed_tasks = _merge_task_lists(seeded_tasks, route_filtered)
        final_tasks = _sanitize_and_number_tasks(
            routed_tasks,
            schema_cols,
            numeric_cols,
            planning_contract=planning_contract,
        )

        if not final_tasks:
            fallback_col = "revenue" if "revenue" in numeric_cols else (numeric_cols[0] if numeric_cols else None)
            if fallback_col:
                final_tasks = [
                    {
                        "id": "T1",
                        "type": "distribution",
                        "params": {"column": fallback_col, "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95]},
                    }
                ]
            else:
                fallback_group = schema_cols[0] if schema_cols else "unknown_col"
                final_tasks = [
                    {
                        "id": "T1",
                        "type": "groupby_agg",
                        "params": {"group_by": fallback_group, "metric": fallback_group, "agg": "count", "limit": 50},
                    }
                ]

        plan = {
            "analysis_type": analysis_type,
            "routing_reason": routing_reason,
            "blocked_requirements": blocked_requirements,
            "feasibility_status": planning_contract.get("feasibility_status"),
            "planning_contract": planning_contract,
            "tasks": final_tasks,
            "notes": (
                f"Deterministic-first plan with optional LLM enrichment. "
                f"Domain={business_domain}, route={analysis_type}."
            ),
        }
        store.write_json("analysis_tasks.json", plan)
        logger.info("[Planner] Wrote analysis_tasks.json")
        return plan
