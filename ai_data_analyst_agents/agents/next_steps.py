from __future__ import annotations

from typing import Any, Dict, List
import json
import re

from ai_data_analyst_agents.agents.metrics import MetricsAgent
from ai_data_analyst_agents.agents.planner import (
    AB_KEYWORDS,
    CONVERSION_COL_CANDIDATES,
    CUSTOMER_KEYWORDS,
    DRIVER_COLS,
    REGRESSION_KEYWORDS,
    SEGMENT_KEYWORDS,
    STAT_KEYWORDS,
    WHY_KEYWORDS,
    _choose_metric,
    _choose_primary_dim,
    _choose_secondary_dim,
    _contains_any,
    _is_datetime_dtype,
    _is_numeric_dtype,
    _sanitize_and_number_tasks,
)
from ai_data_analyst_agents.core.agent_base import Agent


AB_GROUP_COL_CANDIDATES = ["variant", "experiment_group", "treatment_group", "treatment", "group", "arm", "bucket"]


def _task_key(task: Dict[str, Any]) -> str:
    return json.dumps({"type": task.get("type"), "params": task.get("params", {})}, sort_keys=True, ensure_ascii=False)


def _has_task(planned: List[Dict[str, Any]], ttype: str, **params: Any) -> bool:
    for task in planned:
        if task.get("type") != ttype:
            continue
        p = task.get("params", {}) or {}
        if all(p.get(k) == v for k, v in params.items()):
            return True
    return False


def _choose_experiment_group_col(schema_cols: List[str], column_profiles: List[Dict[str, Any]]) -> str | None:
    for cand in AB_GROUP_COL_CANDIDATES:
        if cand in schema_cols:
            return cand
    for prof in column_profiles:
        name = str(prof.get("name", ""))
        examples = {str(v).strip().lower() for v in prof.get("example_values", []) or []}
        if {"control", "treatment"} <= examples or {"a", "b"} <= examples:
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


def _choose_binary_metric(schema_cols: List[str], question: str) -> str | None:
    q = question.lower()
    for cand in CONVERSION_COL_CANDIDATES:
        if cand in schema_cols:
            return cand
    for col in schema_cols:
        cl = col.lower()
        if cl in q and any(tok in cl for tok in ["convert", "click", "purchase", "signup", "churn"]):
            return col
    return None


def _choose_regression_predictors(numeric_cols: List[str], target: str | None) -> List[str]:
    out: List[str] = []
    for cand in DRIVER_COLS + ["customer_age", "tenure_days", "sessions", "marketing_spend"]:
        if cand in numeric_cols and cand != target and cand not in out:
            out.append(cand)
    for col in numeric_cols:
        if col != target and col not in out:
            out.append(col)
        if len(out) >= 4:
            break
    return out[:4]


def _build_followup_candidates(
    *,
    question: str,
    schema_cols: List[str],
    dtypes: Dict[str, Any],
    datetime_candidates: List[str],
    column_profiles: List[Dict[str, Any]],
    existing_tasks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    q = question.lower()
    numeric_cols = [c for c in schema_cols if _is_numeric_dtype(str(dtypes.get(c, "")))]
    date_cols = [c for c in schema_cols if _is_datetime_dtype(str(dtypes.get(c, "")))]
    for c in datetime_candidates:
        if c in schema_cols and c not in date_cols:
            date_cols.append(c)
    categorical_cols = [c for c in schema_cols if c not in numeric_cols]

    metric = _choose_metric(question, schema_cols, numeric_cols)
    primary_dim = _choose_primary_dim(question, schema_cols, categorical_cols)
    secondary_dim = _choose_secondary_dim(question, schema_cols, primary_dim)
    count_metric = "order_id" if "order_id" in schema_cols else ("transaction_id" if "transaction_id" in schema_cols else None)
    why_intent = _contains_any(q, WHY_KEYWORDS) or re.search(r"\b(compare|difference|lower|higher|less|more)\b", q) is not None
    customer_intent = _contains_any(q, CUSTOMER_KEYWORDS)
    segment_intent = _contains_any(q, SEGMENT_KEYWORDS) or customer_intent
    stat_intent = _contains_any(q, STAT_KEYWORDS)
    ab_intent = _contains_any(q, AB_KEYWORDS)
    regression_intent = _contains_any(q, REGRESSION_KEYWORDS)

    candidates: List[Dict[str, Any]] = []

    def add(ttype: str, params: Dict[str, Any]) -> None:
        task = {"type": ttype, "params": params}
        if _task_key(task) in {_task_key(t) for t in existing_tasks}:
            return
        if _task_key(task) in {_task_key(t) for t in candidates}:
            return
        candidates.append(task)

    if metric and primary_dim and why_intent:
        add("groupby_agg", {"group_by": primary_dim, "metric": metric, "agg": "count", "limit": 1000})
        add("groupby_agg", {"group_by": primary_dim, "metric": metric, "agg": "mean", "limit": 1000})
        if count_metric:
            add("groupby_agg", {"group_by": primary_dim, "metric": count_metric, "agg": "count", "limit": 1000})
        if secondary_dim:
            add("groupby2_agg", {"group_by_1": primary_dim, "group_by_2": secondary_dim, "metric": metric, "agg": "mean", "limit": 5000})
            if count_metric:
                add("groupby2_agg", {"group_by_1": primary_dim, "group_by_2": secondary_dim, "metric": count_metric, "agg": "count", "limit": 5000})
        for driver in DRIVER_COLS:
            if driver in numeric_cols and driver != metric:
                add("correlation", {"x": driver, "y": metric})
        if date_cols:
            add("timeseries_agg", {"date_col": date_cols[0], "metric": metric, "freq": "M", "agg": "sum"})

    if metric and segment_intent and primary_dim:
        add("segment_analysis", {"segment_by": primary_dim, "metric": metric, "agg": "mean", "limit": 200})
        add("group_distribution", {"group_by": primary_dim, "metric": metric, "agg": "sum", "quantiles": [0.5, 0.75, 0.9, 0.95, 0.99]})

    if customer_intent and "customer_id" in schema_cols:
        add("topk", {"by": "customer_id", "metric": metric or count_metric or "customer_id", "agg": "sum" if metric else "count", "k": 200})
        if date_cols:
            add("recency_by_group", {"group_by": "customer_id", "date_col": date_cols[0]})

    experiment_group_col = _choose_experiment_group_col(schema_cols, column_profiles)
    control_label, treatment_label = _extract_experiment_labels(column_profiles, experiment_group_col)
    binary_metric = _choose_binary_metric(schema_cols, question)
    if ab_intent and experiment_group_col and (binary_metric or metric):
        add(
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

    if stat_intent and primary_dim and (binary_metric or metric):
        add(
            "statistical_test",
            {
                "group_col": primary_dim,
                "metric": binary_metric or metric,
                "alpha": 0.05,
                "alternative": "two-sided",
            },
        )

    if regression_intent and metric:
        predictors = _choose_regression_predictors(numeric_cols, metric)
        if predictors:
            add("ols_regression", {"target": metric, "predictors": predictors, "alpha": 0.05})

    sanitized = _sanitize_and_number_tasks(candidates, schema_cols, numeric_cols)
    out: List[Dict[str, Any]] = []
    for idx, task in enumerate(sanitized, start=1):
        task["id"] = f"F{idx}"
        out.append(task)
    return out


class NextStepsAgent(Agent):
    name = "next_steps"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        store = ctx["store"]
        logger = ctx["logger"]
        memory = ctx["memory"]
        question = ctx.get("business_question", "")

        profile = memory.get("result.profiling") or {}
        planner_out = memory.get("result.planner") or {"tasks": []}
        metrics_out = memory.get("result.metrics") or {"computed": [], "failed": [], "skipped": []}

        existing_tasks = list(planner_out.get("tasks", []) or [])
        followup_tasks = _build_followup_candidates(
            question=question,
            schema_cols=profile.get("columns", []) or [],
            dtypes=profile.get("dtypes", {}) or {},
            datetime_candidates=profile.get("datetime_candidates", []) or [],
            column_profiles=profile.get("column_profiles", []) or [],
            existing_tasks=existing_tasks,
        )

        followup_plan = {
            "tasks": followup_tasks,
            "notes": "Automatically expanded tasks from coverage gaps and next-step heuristics.",
        }
        store.write_json("next_steps_plan.json", followup_plan)

        if not followup_tasks:
            logger.info("[NextSteps] No additional tasks generated.")
            return {"tasks": [], "execution": {"computed": [], "failed": [], "skipped": []}, "merged_task_count": len(existing_tasks)}

        analysis_tasks_path = store.path("analysis_tasks.json")
        initial_tasks_path = store.path("analysis_tasks_initial.json")
        if analysis_tasks_path.exists() and not initial_tasks_path.exists():
            store.write_text("analysis_tasks_initial.json", analysis_tasks_path.read_text(encoding="utf-8"))

        combined_plan = {
            "tasks": [*existing_tasks, *followup_tasks],
            "notes": str(planner_out.get("notes", "")).strip() + " + next_steps expansion",
        }
        store.write_json("analysis_tasks.json", combined_plan)
        memory.set("result.planner.followup_only", followup_plan)
        memory.set("result.planner", followup_plan)
        execution = MetricsAgent().run(ctx)
        store.write_json("next_steps_metrics_outputs.json", execution)
        memory.set("result.planner", combined_plan)

        merged = {
            "computed": [*(metrics_out.get("computed", []) or []), *(execution.get("computed", []) or [])],
            "failed": [*(metrics_out.get("failed", []) or []), *(execution.get("failed", []) or [])],
            "skipped": [*(metrics_out.get("skipped", []) or []), *(execution.get("skipped", []) or [])],
            "followup": {
                "tasks_added": len(followup_tasks),
                "followup_output_artifact": "next_steps_metrics_outputs.json",
            },
        }
        store.write_json("metrics_outputs.json", merged)
        memory.set("result.metrics", merged)

        logger.info(f"[NextSteps] Added {len(followup_tasks)} follow-up task(s).")
        return {"tasks": followup_tasks, "execution": execution, "merged_task_count": len(combined_plan["tasks"])}
