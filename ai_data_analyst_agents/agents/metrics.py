from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.core.kpi_templates import detect_business_domain, pick_template_dimension
from ai_data_analyst_agents.core.metric_engine import (
    compute_cohort_retention,
    compute_metric_definition,
    compute_segment_profile,
    compute_template_kpis,
)
from ai_data_analyst_agents.core.security import validate_read_only_sql
from ai_data_analyst_agents.core.sql_source import compute_join_profile
from ai_data_analyst_agents.statistics.ab_testing import run_ab_test
from ai_data_analyst_agents.statistics.artifacts import write_statistical_artifacts
from ai_data_analyst_agents.statistics.models import ABTestRequest, HypothesisTestRequest, RegressionRequest
from ai_data_analyst_agents.statistics.regression import run_ols_regression
from ai_data_analyst_agents.statistics.selector import run_hypothesis_test


# -----------------------------
# Task normalization utilities
# -----------------------------
def _first_present(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None and str(d[k]).strip() != "":
            return d[k]
    return None


def _normalize_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make planner outputs compatible with MetricsAgent expected schema.
    Converts common alternative keys into the canonical ones.

    Canonical schemas:
    - groupby_agg: {group_by, metric, agg?, limit?}
    - groupby2_agg: {group_by_1, group_by_2, metric, agg?, limit?}
    - filter_agg: {filter_col, filter_val, metric, agg?}
    - correlation: {x, y}
    - distribution: {column, quantiles?}
    - group_distribution: {group_by, metric, agg?, quantiles?}
    - recency_by_group: {group_by, date_col}
    - topk: {by, metric, agg?, k?}
    - timeseries_agg: {date_col, metric, freq?, agg?}
    - statistical_test: {group_col, metric, group_a?, group_b?, compare_to_rest?, paired?, pair_id_col?, success_value?, alpha?, alternative?}
    - ab_test: {group_col, control, treatment, metric, metric_type?, success_value?, alpha?}
    - ols_regression: {target, predictors, alpha?}
    """
    t: Dict[str, Any] = dict(task or {})
    p: Dict[str, Any] = dict(t.get("params", {}) or {})

    ttype = (t.get("type") or "").strip()

    if ttype == "groupby_agg":
        if "group_by" not in p:
            p["group_by"] = _first_present(p, ["group_by", "groupby", "group", "dimension", "by", "column", "col"])
        if "metric" not in p:
            p["metric"] = _first_present(p, ["metric", "value", "measure", "target", "y", "sum_col", "metric_col"])
        if "agg" not in p:
            p["agg"] = _first_present(p, ["agg", "aggregation", "op"]) or "sum"
        if "limit" not in p:
            p["limit"] = _first_present(p, ["limit", "top_k", "topk", "k"]) or 50

        # ensure int
        try:
            p["limit"] = int(p["limit"])
        except Exception:
            p["limit"] = 50

    elif ttype == "groupby2_agg":
        if "group_by_1" not in p:
            p["group_by_1"] = _first_present(
                p, ["group_by_1", "group1", "group_by", "groupby_1", "dim1", "dimension1"]
            )
        if "group_by_2" not in p:
            p["group_by_2"] = _first_present(
                p, ["group_by_2", "group2", "groupby_2", "dim2", "dimension2"]
            )
        if "metric" not in p:
            p["metric"] = _first_present(p, ["metric", "value", "measure", "target", "y", "metric_col"])
        if "agg" not in p:
            p["agg"] = _first_present(p, ["agg", "aggregation", "op"]) or "sum"
        if "limit" not in p:
            p["limit"] = _first_present(p, ["limit", "top_k", "topk", "k"]) or 100

        try:
            p["limit"] = int(p["limit"])
        except Exception:
            p["limit"] = 100

    elif ttype == "correlation":
        if "x" not in p:
            p["x"] = _first_present(p, ["x", "col1", "a", "feature_x", "feature1"])
        if "y" not in p:
            p["y"] = _first_present(p, ["y", "col2", "b", "feature_y", "feature2"])

    elif ttype == "distribution":
        if "column" not in p:
            p["column"] = _first_present(p, ["column", "metric", "col", "feature", "x"])
        if "quantiles" not in p or not isinstance(p.get("quantiles"), list):
            p["quantiles"] = [0.05, 0.25, 0.5, 0.75, 0.95]

    elif ttype == "group_distribution":
        if "group_by" not in p:
            p["group_by"] = _first_present(p, ["group_by", "groupby", "group", "dimension", "by", "column", "col"])
        if "metric" not in p:
            p["metric"] = _first_present(p, ["metric", "value", "measure", "target", "y", "metric_col"])
        if "agg" not in p:
            p["agg"] = _first_present(p, ["agg", "aggregation", "op"]) or "sum"
        if "quantiles" not in p or not isinstance(p.get("quantiles"), list):
            p["quantiles"] = [0.5, 0.75, 0.9, 0.95, 0.99]

    elif ttype == "recency_by_group":
        if "group_by" not in p:
            p["group_by"] = _first_present(p, ["group_by", "groupby", "group", "dimension", "by", "column", "col"])
        if "date_col" not in p:
            p["date_col"] = _first_present(p, ["date_col", "date", "time_col", "timestamp_col", "x"])

    elif ttype == "topk":
        if "by" not in p:
            p["by"] = _first_present(p, ["by", "group_by", "groupby", "group", "dimension", "column", "col"])
        if "metric" not in p:
            p["metric"] = _first_present(p, ["metric", "value", "measure", "target", "y", "metric_col"])
        if "agg" not in p:
            p["agg"] = _first_present(p, ["agg", "aggregation", "op"]) or "sum"
        if "k" not in p:
            p["k"] = _first_present(p, ["k", "top_k", "topk", "limit"]) or 10
        try:
            p["k"] = int(p["k"])
        except Exception:
            p["k"] = 10

    elif ttype == "timeseries_agg":
        if "date_col" not in p:
            p["date_col"] = _first_present(p, ["date_col", "date", "time_col", "timestamp_col", "x"])
        if "metric" not in p:
            p["metric"] = _first_present(p, ["metric", "value", "measure", "target", "y"])
        if "freq" not in p:
            p["freq"] = _first_present(p, ["freq", "granularity", "period"]) or "M"
        if "agg" not in p:
            p["agg"] = _first_present(p, ["agg", "aggregation", "op"]) or "sum"

    elif ttype == "filter_agg":
        # ✅ Handle nested where dict produced by some planners:
        # e.g. {"where": {"country": "India"}} or {"filter": {"country": "India"}}
        where = p.get("where")
        if isinstance(where, dict) and ("filter_col" not in p or "filter_val" not in p):
            if len(where) == 1:
                k = next(iter(where.keys()))
                p.setdefault("filter_col", k)
                p.setdefault("filter_val", where[k])

        filt = p.get("filter")
        if isinstance(filt, dict) and ("filter_col" not in p or "filter_val" not in p):
            if len(filt) == 1:
                k = next(iter(filt.keys()))
                p.setdefault("filter_col", k)
                p.setdefault("filter_val", filt[k])

        # ✅ Flat-key normalization (expanded variants)
        if "filter_col" not in p:
            p["filter_col"] = _first_present(
                p,
                ["filter_col", "where_col", "column", "field", "by", "dim", "dimension", "group_by", "groupby", "group"]
            )
        if "filter_val" not in p:
            p["filter_val"] = _first_present(
                p,
                ["filter_val", "where_val", "value", "equals", "match", "val", "target_value"]
            )

        # Some planners emit {"country": "India"} directly as params
        if ("filter_col" not in p or "filter_val" not in p):
            for candidate_col in ["country", "product_category", "customer_id"]:
                if candidate_col in p and p.get(candidate_col) is not None:
                    p.setdefault("filter_col", candidate_col)
                    p.setdefault("filter_val", p[candidate_col])
                    break

        if "metric" not in p:
            p["metric"] = _first_present(
                p,
                ["metric", "measure", "target", "metric_col", "value_col", "y", "value_metric"]
            )
        if "agg" not in p:
            p["agg"] = _first_present(p, ["agg", "aggregation", "op"]) or "sum"

    elif ttype == "sql_query":
        if "query" not in p:
            p["query"] = _first_present(p, ["query", "sql", "statement"])
        if "limit" not in p:
            p["limit"] = _first_present(p, ["limit", "max_rows", "top_k", "k"]) or 1000
        if "output" not in p:
            p["output"] = _first_present(p, ["output", "format", "mode"]) or "rows"
        try:
            p["limit"] = int(p["limit"])
        except Exception:
            p["limit"] = 1000

    elif ttype == "sql_join_profile":
        if "fact_table" not in p:
            p["fact_table"] = _first_present(p, ["fact_table", "from_table", "base_table"])
        if "dimension_table" not in p:
            p["dimension_table"] = _first_present(p, ["dimension_table", "to_table", "join_table"])

    elif ttype == "kpi_template_apply":
        if "domain" not in p:
            p["domain"] = _first_present(p, ["domain", "template", "kpi_domain"])
        if "segment_by" not in p:
            p["segment_by"] = _first_present(p, ["segment_by", "group_by", "dimension", "by"])

    elif ttype == "metric_definition":
        if "name" not in p:
            p["name"] = _first_present(p, ["name", "metric_name", "id"])
        if "metric_col" not in p:
            p["metric_col"] = _first_present(p, ["metric_col", "metric", "value_col", "column"])
        if "expression" not in p:
            p["expression"] = _first_present(p, ["expression", "expr", "formula"])
        if "group_by" not in p:
            p["group_by"] = _first_present(p, ["group_by", "segment_by", "dimension", "by"])
        if "agg" not in p:
            p["agg"] = _first_present(p, ["agg", "aggregation", "op"]) or "sum"

    elif ttype == "segment_analysis":
        if "segment_by" not in p:
            p["segment_by"] = _first_present(p, ["segment_by", "group_by", "by", "dimension"])
        if "metric" not in p:
            p["metric"] = _first_present(p, ["metric", "metric_col", "value", "measure"])
        if "agg" not in p:
            p["agg"] = _first_present(p, ["agg", "aggregation", "op"]) or "sum"
        if "limit" not in p:
            p["limit"] = _first_present(p, ["limit", "top_k", "k"]) or 100
        try:
            p["limit"] = int(p["limit"])
        except Exception:
            p["limit"] = 100

    elif ttype == "cohort_analysis":
        if "entity_col" not in p:
            p["entity_col"] = _first_present(p, ["entity_col", "entity", "id_col", "customer_col"])
        if "date_col" not in p:
            p["date_col"] = _first_present(p, ["date_col", "date", "time_col", "timestamp_col"])
        if "freq" not in p:
            p["freq"] = _first_present(p, ["freq", "period", "granularity"]) or "M"

    elif ttype == "statistical_test":
        if "group_col" not in p:
            p["group_col"] = _first_present(p, ["group_col", "group_by", "group", "segment_by", "by", "dimension"])
        if "metric" not in p:
            p["metric"] = _first_present(p, ["metric", "value", "measure", "target", "column"])
        if "group_a" not in p:
            p["group_a"] = _first_present(p, ["group_a", "label_a", "segment_a", "control", "treatment"])
        if "group_b" not in p:
            p["group_b"] = _first_present(p, ["group_b", "label_b", "segment_b"])
        if "pair_id_col" not in p:
            p["pair_id_col"] = _first_present(p, ["pair_id_col", "pair_col", "entity_col", "id_col"])
        if "alpha" not in p:
            p["alpha"] = _first_present(p, ["alpha", "significance_level"]) or 0.05
        if "alternative" not in p:
            p["alternative"] = _first_present(p, ["alternative"]) or "two-sided"
        p["compare_to_rest"] = bool(p.get("compare_to_rest", False))
        p["paired"] = bool(p.get("paired", False))

    elif ttype == "ab_test":
        if "group_col" not in p:
            p["group_col"] = _first_present(p, ["group_col", "variant_col", "group_by", "group"])
        if "control" not in p:
            p["control"] = _first_present(p, ["control", "group_b", "label_b"])
        if "treatment" not in p:
            p["treatment"] = _first_present(p, ["treatment", "group_a", "label_a"])
        if "metric" not in p:
            p["metric"] = _first_present(p, ["metric", "value", "measure", "target", "column"])
        if "metric_type" not in p:
            p["metric_type"] = _first_present(p, ["metric_type", "outcome_type"]) or "auto"
        if "alpha" not in p:
            p["alpha"] = _first_present(p, ["alpha", "significance_level"]) or 0.05

    elif ttype == "ols_regression":
        if "target" not in p:
            p["target"] = _first_present(p, ["target", "metric", "dependent_var", "y"])
        predictors = p.get("predictors")
        if not isinstance(predictors, list):
            maybe = _first_present(p, ["predictors", "features", "x"])
            if isinstance(maybe, list):
                p["predictors"] = maybe
            elif isinstance(maybe, str) and maybe.strip():
                p["predictors"] = [x.strip() for x in maybe.split(",") if x.strip()]
        if "alpha" not in p:
            p["alpha"] = _first_present(p, ["alpha", "significance_level"]) or 0.05

    t["params"] = p
    return t


def _require_params(tid: str, ttype: str, p: Dict[str, Any], required: list[str]) -> None:
    missing = []
    for k in required:
        if k not in p:
            missing.append(k)
        else:
            v = p.get(k)
            if v is None or str(v).strip() == "":
                missing.append(k)

    if missing:
        raise ValueError(f"{tid} ({ttype}) missing required params {missing}. Got params={p}")


def _ensure_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"Column not found: {col}")


def _safe_agg(series: pd.Series, agg: str) -> float:
    agg = (agg or "sum").lower().strip()
    if agg not in {"sum", "mean", "min", "max", "median", "count"}:
        raise ValueError(f"Unsupported agg '{agg}'. Use one of sum/mean/min/max/median/count.")

    if agg == "sum":
        return float(series.sum())
    if agg == "mean":
        return float(series.mean())
    if agg == "min":
        return float(series.min())
    if agg == "max":
        return float(series.max())
    if agg == "median":
        return float(series.median())
    if agg == "count":
        return float(series.count())

    # unreachable
    return float(series.sum())


class MetricsAgent(Agent):
    name = "metrics"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        cfg = ctx["cfg"]
        store = ctx["store"]
        logger = ctx["logger"]
        evidence = ctx["evidence"]
        sql_source = ctx.get("sql_source")
        sql_schema = ctx.get("sql_schema") if isinstance(ctx.get("sql_schema"), dict) else None

        df: pd.DataFrame = ctx["memory"].get("df.cleaned", ctx["df"])
        task_plan = ctx["memory"].get("result.planner", {}) or {}
        tasks = task_plan.get("tasks", []) or []

        outputs: Dict[str, Any] = {
            "computed": [],
            "failed": [],
            "skipped": [],
        }

        for t in tasks:
            # Keep a copy of what the planner produced (for debugging)
            tid = str(t.get("id", "T?"))
            ttype_raw = str(t.get("type", "")).strip()
            raw_params = dict(t.get("params", {}) or {})

            # Normalize to canonical schema
            t = _normalize_task(t)
            ttype = str(t.get("type", "")).strip()
            p: Dict[str, Any] = dict(t.get("params", {}) or {})

            # Debug log so you can SEE why keys are missing
            logger.info(
                f"[Metrics] {tid} type={ttype_raw} -> {ttype} | "
                f"raw_params={raw_params} | normalized_params={p}"
            )

            try:
                if ttype == "groupby_agg":
                    _require_params(tid, ttype, p, ["group_by", "metric"])
                    group = str(p["group_by"])
                    metric = str(p["metric"])
                    agg = str(p.get("agg", "sum"))
                    limit = int(p.get("limit", 50))

                    _ensure_col(df, group)
                    _ensure_col(df, metric)

                    res = df.groupby(group)[metric].agg(agg).sort_values(ascending=False)
                    if limit > 0:
                        res = res.head(limit)

                    artifact = f"{tid}_groupby_{group}_{metric}_{agg}.json"
                    store.write_json(artifact, res.to_dict())

                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer=None,
                        summary=f"{agg}({metric}) grouped by {group}" + (f" (top {limit})" if limit > 0 else " (all)"),
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "groupby2_agg":
                    _require_params(tid, ttype, p, ["group_by_1", "group_by_2", "metric"])
                    g1 = str(p["group_by_1"])
                    g2 = str(p["group_by_2"])
                    metric = str(p["metric"])
                    agg = str(p.get("agg", "sum"))
                    limit = int(p.get("limit", 100))

                    _ensure_col(df, g1)
                    _ensure_col(df, g2)
                    _ensure_col(df, metric)

                    res = df.groupby([g1, g2])[metric].agg(agg).sort_values(ascending=False)
                    if limit > 0:
                        res = res.head(limit)
                    rows = []
                    for idx, val in res.to_dict().items():
                        if isinstance(idx, tuple) and len(idx) == 2:
                            rows.append({g1: idx[0], g2: idx[1], "value": float(val)})
                        else:
                            rows.append({"group_key": str(idx), "value": float(val)})

                    artifact = f"{tid}_groupby2_{g1}_{g2}_{metric}_{agg}.json"
                    store.write_json(artifact, rows)

                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer=None,
                        summary=f"{agg}({metric}) grouped by ({g1}, {g2})" + (f" (top {limit})" if limit > 0 else " (all)"),
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "filter_agg":
                    _require_params(tid, ttype, p, ["filter_col", "filter_val", "metric"])
                    filt_col = str(p["filter_col"])
                    filt_val = p["filter_val"]
                    metric = str(p["metric"])
                    agg = str(p.get("agg", "sum"))

                    _ensure_col(df, filt_col)
                    _ensure_col(df, metric)

                    dff = df[df[filt_col].astype(str) == str(filt_val)]
                    val = _safe_agg(pd.to_numeric(dff[metric], errors="coerce"), agg)

                    artifact = f"{tid}_filter_{filt_col}_{metric}_{agg}.json"
                    store.write_json(artifact, {"filter": {filt_col: filt_val}, "value": val})

                    ev = evidence.add(
                        kind="metric",
                        artifact_path=artifact,
                        pointer="value",
                        summary=f"{agg}({metric}) where {filt_col} == {filt_val}",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "correlation":
                    _require_params(tid, ttype, p, ["x", "y"])
                    x = str(p["x"])
                    y = str(p["y"])

                    _ensure_col(df, x)
                    _ensure_col(df, y)

                    pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
                    if len(pair) < 3:
                        outputs["skipped"].append(
                            {"task_id": tid, "reason": f"Not enough numeric rows for correlation ({x} vs {y})."}
                        )
                        continue

                    corr_val = pair[x].corr(pair[y])
                    if pd.isna(corr_val):
                        outputs["skipped"].append(
                            {"task_id": tid, "reason": f"Correlation is NaN ({x} vs {y})."}
                        )
                        continue

                    corr = float(corr_val)

                    artifact = f"{tid}_corr_{x}_{y}.json"
                    store.write_json(artifact, {"x": x, "y": y, "corr": corr})

                    ev = evidence.add(
                        kind="metric",
                        artifact_path=artifact,
                        pointer="corr",
                        summary=f"Correlation between {x} and {y}",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "distribution":
                    _require_params(tid, ttype, p, ["column"])
                    col = str(p["column"])
                    _ensure_col(df, col)

                    q_list = p.get("quantiles", [0.05, 0.25, 0.5, 0.75, 0.95])
                    if not isinstance(q_list, list):
                        q_list = [0.05, 0.25, 0.5, 0.75, 0.95]
                    q_vals = []
                    for q in q_list:
                        try:
                            qf = float(q)
                            if 0.0 <= qf <= 1.0:
                                q_vals.append(qf)
                        except Exception:
                            continue
                    if not q_vals:
                        q_vals = [0.05, 0.25, 0.5, 0.75, 0.95]

                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    if s.empty:
                        outputs["skipped"].append(
                            {"task_id": tid, "reason": f"No numeric values available for distribution ({col})."}
                        )
                        continue

                    q = s.quantile(q_vals).to_dict()
                    quantiles = {str(k): float(v) for k, v in q.items()}
                    payload = {
                        "column": col,
                        "count": int(s.shape[0]),
                        "mean": float(s.mean()),
                        "std": float(s.std()) if s.shape[0] > 1 else 0.0,
                        "min": float(s.min()),
                        "max": float(s.max()),
                        "quantiles": quantiles,
                    }

                    artifact = f"{tid}_dist_{col}.json"
                    store.write_json(artifact, payload)
                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer=None,
                        summary=f"Distribution summary for {col}",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "group_distribution":
                    _require_params(tid, ttype, p, ["group_by", "metric"])
                    group = str(p["group_by"])
                    metric = str(p["metric"])
                    agg = str(p.get("agg", "sum"))
                    _ensure_col(df, group)
                    _ensure_col(df, metric)

                    grouped = df.groupby(group)[metric].agg(agg)
                    s = pd.to_numeric(grouped, errors="coerce").dropna()
                    if s.empty:
                        outputs["skipped"].append(
                            {"task_id": tid, "reason": f"No numeric grouped values for distribution ({group}, {metric})."}
                        )
                        continue

                    q_list = p.get("quantiles", [0.5, 0.75, 0.9, 0.95, 0.99])
                    if not isinstance(q_list, list):
                        q_list = [0.5, 0.75, 0.9, 0.95, 0.99]
                    q_vals = []
                    for q in q_list:
                        try:
                            qf = float(q)
                            if 0.0 <= qf <= 1.0:
                                q_vals.append(qf)
                        except Exception:
                            continue
                    if not q_vals:
                        q_vals = [0.5, 0.75, 0.9, 0.95, 0.99]

                    q = s.quantile(q_vals).to_dict()
                    payload = {
                        "group_by": group,
                        "metric": metric,
                        "agg": agg,
                        "n_groups": int(s.shape[0]),
                        "quantiles": {str(k): float(v) for k, v in q.items()},
                        "max_group_value": float(s.max()),
                        "min_group_value": float(s.min()),
                        "mean_group_value": float(s.mean()),
                    }

                    artifact = f"{tid}_group_dist_{group}_{metric}_{agg}.json"
                    store.write_json(artifact, payload)
                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer=None,
                        summary=f"Distribution of {agg}({metric}) across {group}",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "recency_by_group":
                    _require_params(tid, ttype, p, ["group_by", "date_col"])
                    group = str(p["group_by"])
                    date_col = str(p["date_col"])
                    _ensure_col(df, group)
                    _ensure_col(df, date_col)

                    dff = df[[group, date_col]].copy()
                    dff[date_col] = pd.to_datetime(dff[date_col], errors="coerce")
                    dff = dff.dropna(subset=[group, date_col])
                    if dff.empty:
                        outputs["skipped"].append(
                            {"task_id": tid, "reason": f"No valid rows for recency computation ({group}, {date_col})."}
                        )
                        continue

                    latest = dff[date_col].max()
                    rec = dff.groupby(group)[date_col].max()
                    out = {str(k): int((latest - v).days) for k, v in rec.items()}

                    artifact = f"{tid}_recency_{group}_{date_col}.json"
                    store.write_json(artifact, {"group_by": group, "reference_date": latest.isoformat(), "recency_days": out})
                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer="recency_days",
                        summary=f"Recency days by {group} using {date_col}",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "topk":
                    _require_params(tid, ttype, p, ["by", "metric"])
                    by = str(p["by"])
                    metric = str(p["metric"])
                    agg = str(p.get("agg", "sum"))
                    k = int(p.get("k", 10))

                    _ensure_col(df, by)
                    _ensure_col(df, metric)

                    res = (
                        df.groupby(by)[metric]
                        .agg(agg)
                        .sort_values(ascending=False)
                        .head(k)
                    )

                    artifact = f"{tid}_topk_{by}_{metric}_{agg}.json"
                    store.write_json(artifact, res.to_dict())
                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer=None,
                        summary=f"Top {k} {by} by {agg}({metric})",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "timeseries_agg":
                    _require_params(tid, ttype, p, ["date_col", "metric"])
                    date_col = str(p["date_col"])
                    metric = str(p["metric"])
                    freq = str(p.get("freq", "M")).upper()
                    agg = str(p.get("agg", "sum"))
                    freq_alias = {"M": "ME", "Q": "QE"}.get(freq, freq)

                    _ensure_col(df, date_col)
                    _ensure_col(df, metric)

                    dff = df[[date_col, metric]].copy()
                    dff[date_col] = pd.to_datetime(dff[date_col], errors="coerce")
                    dff[metric] = pd.to_numeric(dff[metric], errors="coerce")
                    dff = dff.dropna(subset=[date_col, metric])
                    if dff.empty:
                        outputs["skipped"].append(
                            {"task_id": tid, "reason": f"No valid rows for timeseries aggregation ({date_col}, {metric})."}
                        )
                        continue

                    res = (
                        dff.set_index(date_col)[metric]
                        .resample(freq_alias)
                        .agg(agg)
                        .dropna()
                    )
                    out = {}
                    for k, v in res.to_dict().items():
                        key = k.isoformat() if hasattr(k, "isoformat") else str(k)
                        out[key] = float(v)

                    artifact = f"{tid}_timeseries_{date_col}_{metric}_{freq}_{agg}.json"
                    store.write_json(artifact, out)
                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer=None,
                        summary=f"{agg}({metric}) resampled by {freq} on {date_col}",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "sql_query":
                    _require_params(tid, ttype, p, ["query"])
                    if sql_source is None:
                        outputs["skipped"].append({"task_id": tid, "reason": "SQL source unavailable in current run."})
                        continue

                    query = str(p["query"])
                    if getattr(cfg, "security", None) is not None and cfg.security.enforce_read_only_sql:
                        query = validate_read_only_sql(query)
                    else:
                        query = query.strip().rstrip(";")
                    limit = int(p.get("limit", cfg.sql.default_query_row_limit))
                    output_mode = str(p.get("output", "rows")).lower()

                    store.write_text(f"{tid}_query.sql", query.strip() + "\n")
                    qdf = sql_source.execute_query(query, limit=limit)

                    payload: Dict[str, Any]
                    if output_mode == "mapping" and qdf.shape[1] >= 2:
                        key_col = qdf.columns[0]
                        val_col = qdf.columns[1]
                        mapping: Dict[str, float] = {}
                        for _, row in qdf.iterrows():
                            try:
                                mapping[str(row[key_col])] = float(row[val_col])
                            except Exception:
                                continue
                        payload = mapping
                    else:
                        payload = {
                            "columns": [str(c) for c in qdf.columns.tolist()],
                            "rows": qdf.to_dict(orient="records"),
                            "n_rows": int(qdf.shape[0]),
                        }

                    artifact = f"{tid}_sql_query.json"
                    store.write_json(artifact, payload)
                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer=None,
                        summary=f"SQL query result ({qdf.shape[0]} rows)",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "sql_join_profile":
                    _require_params(tid, ttype, p, ["fact_table"])
                    if sql_source is None:
                        outputs["skipped"].append({"task_id": tid, "reason": "SQL source unavailable in current run."})
                        continue

                    profile = compute_join_profile(
                        engine=sql_source.engine,
                        schema=sql_schema or {},
                        fact_table=str(p["fact_table"]),
                        dimension_table=str(p["dimension_table"]) if p.get("dimension_table") else None,
                    )
                    artifact = f"{tid}_sql_join_profile.json"
                    store.write_json(artifact, profile)
                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer=None,
                        summary="SQL join path and row coverage profile",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "kpi_template_apply":
                    _require_params(tid, ttype, p, ["domain"])
                    domain = str(p["domain"]).lower()
                    if not domain:
                        domain = detect_business_domain(ctx.get("business_question", ""), list(df.columns))
                    payload = compute_template_kpis(df, domain)

                    segment_by = p.get("segment_by")
                    if not segment_by:
                        segment_by = pick_template_dimension(domain, list(df.columns))
                    if segment_by and "kpis" in payload:
                        # Add one segment view using the first available base KPI column.
                        seg_metric_col = None
                        for name, col in payload.get("resolved_columns", {}).items():
                            if col is not None:
                                seg_metric_col = str(col)
                                break
                        if seg_metric_col and seg_metric_col in df.columns:
                            payload["segment_sample"] = compute_segment_profile(
                                df=df,
                                segment_by=str(segment_by),
                                metric=seg_metric_col,
                                agg="sum",
                                limit=20,
                            )

                    artifact = f"{tid}_kpi_template_{domain}.json"
                    store.write_json(artifact, payload)
                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer=None,
                        summary=f"KPI template metrics for domain={domain}",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "metric_definition":
                    _require_params(tid, ttype, p, ["name"])
                    payload = compute_metric_definition(df, p)
                    artifact = f"{tid}_metric_definition_{str(p.get('name')).lower().replace(' ', '_')}.json"
                    store.write_json(artifact, payload)
                    pointer = "value" if isinstance(payload, dict) and "value" in payload else None
                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer=pointer,
                        summary=f"Metric definition output: {p.get('name')}",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "segment_analysis":
                    _require_params(tid, ttype, p, ["segment_by", "metric"])
                    payload = compute_segment_profile(
                        df=df,
                        segment_by=str(p["segment_by"]),
                        metric=str(p["metric"]),
                        agg=str(p.get("agg", "sum")),
                        limit=int(p.get("limit", 100)),
                    )
                    artifact = f"{tid}_segment_{p['segment_by']}_{p['metric']}.json"
                    store.write_json(artifact, payload)
                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer="values",
                        summary=f"Segment analysis by {p['segment_by']} on {p['metric']}",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "cohort_analysis":
                    _require_params(tid, ttype, p, ["entity_col", "date_col"])
                    payload = compute_cohort_retention(
                        df=df,
                        entity_col=str(p["entity_col"]),
                        date_col=str(p["date_col"]),
                        freq=str(p.get("freq", "M")).upper(),
                    )
                    artifact = f"{tid}_cohort_{p['entity_col']}_{p['date_col']}.json"
                    store.write_json(artifact, payload)
                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer=None,
                        summary=f"Cohort retention by {p['entity_col']} over {p['date_col']}",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "statistical_test":
                    _require_params(tid, ttype, p, ["group_col", "metric"])
                    req = HypothesisTestRequest(
                        group_col=str(p["group_col"]),
                        metric=str(p["metric"]),
                        group_a=p.get("group_a"),
                        group_b=p.get("group_b"),
                        compare_to_rest=bool(p.get("compare_to_rest", False)),
                        paired=bool(p.get("paired", False)),
                        pair_id_col=str(p["pair_id_col"]) if p.get("pair_id_col") else None,
                        success_value=p.get("success_value", 1),
                        alpha=float(p.get("alpha", 0.05)),
                        alternative=str(p.get("alternative", "two-sided")),
                    )
                    selection, result = run_hypothesis_test(df, req, analysis_id=tid)
                    bundle = write_statistical_artifacts(store, task_id=tid, result=result)
                    ev = evidence.add(
                        kind="json",
                        artifact_path=bundle.summary_path,
                        pointer=None,
                        summary=f"Statistical test ({result.method}) for {req.metric} by {req.group_col}",
                    )
                    outputs["computed"].append(
                        {
                            "task_id": tid,
                            "task_type": ttype,
                            "artifact": bundle.summary_path,
                            "artifact_bundle": bundle.to_dict(),
                            "evidence_id": ev.id,
                            "method": result.method,
                            "selection": selection.to_dict(),
                        }
                    )

                elif ttype == "ab_test":
                    _require_params(tid, ttype, p, ["group_col", "control", "treatment", "metric"])
                    req = ABTestRequest(
                        group_col=str(p["group_col"]),
                        control=p["control"],
                        treatment=p["treatment"],
                        metric=str(p["metric"]),
                        metric_type=str(p.get("metric_type", "auto")),
                        success_value=p.get("success_value", 1),
                        alpha=float(p.get("alpha", 0.05)),
                    )
                    result = run_ab_test(df, req, analysis_id=tid)
                    bundle = write_statistical_artifacts(store, task_id=tid, result=result)
                    ev = evidence.add(
                        kind="json",
                        artifact_path=bundle.summary_path,
                        pointer=None,
                        summary=(
                            f"A/B test ({result.method}) comparing treatment={req.treatment} "
                            f"vs control={req.control} on {req.metric}"
                        ),
                    )
                    outputs["computed"].append(
                        {
                            "task_id": tid,
                            "task_type": ttype,
                            "artifact": bundle.summary_path,
                            "artifact_bundle": bundle.to_dict(),
                            "evidence_id": ev.id,
                            "method": result.method,
                        }
                    )

                elif ttype == "ols_regression":
                    _require_params(tid, ttype, p, ["target", "predictors"])
                    predictors = p.get("predictors", [])
                    if not isinstance(predictors, list):
                        raise ValueError(f"{tid} ({ttype}) predictors must be a list. Got params={p}")
                    req = RegressionRequest(
                        target=str(p["target"]),
                        predictors=[str(x) for x in predictors],
                        alpha=float(p.get("alpha", 0.05)),
                    )
                    result, coeff_table, diagnostics = run_ols_regression(df, req, analysis_id=tid)
                    bundle = write_statistical_artifacts(
                        store,
                        task_id=tid,
                        result=result,
                        coefficients=coeff_table,
                        diagnostics=diagnostics,
                    )
                    ev = evidence.add(
                        kind="json",
                        artifact_path=bundle.summary_path,
                        pointer=None,
                        summary=f"OLS regression for target={req.target} with predictors={req.predictors}",
                    )
                    outputs["computed"].append(
                        {
                            "task_id": tid,
                            "task_type": ttype,
                            "artifact": bundle.summary_path,
                            "artifact_bundle": bundle.to_dict(),
                            "evidence_id": ev.id,
                            "method": result.method,
                        }
                    )

                else:
                    outputs["failed"].append({"task_id": tid, "reason": f"Unknown task type: {ttype}", "task": t})

            except Exception as e:
                logger.exception(f"Task {tid} failed")
                outputs["failed"].append({"task_id": tid, "reason": str(e), "task": t})

        store.write_json("metrics_outputs.json", outputs)
        logger.info("Wrote metrics_outputs.json")
        return outputs
