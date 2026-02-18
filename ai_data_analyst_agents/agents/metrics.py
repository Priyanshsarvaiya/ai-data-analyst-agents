from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from ai_data_analyst_agents.core.agent_base import Agent


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
    - filter_agg: {filter_col, filter_val, metric, agg?}
    - correlation: {x, y}
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

    elif ttype == "correlation":
        if "x" not in p:
            p["x"] = _first_present(p, ["x", "col1", "a", "feature_x", "feature1"])
        if "y" not in p:
            p["y"] = _first_present(p, ["y", "col2", "b", "feature_y", "feature2"])

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
        store = ctx["store"]
        logger = ctx["logger"]
        evidence = ctx["evidence"]

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

                    res = (
                        df.groupby(group)[metric]
                        .agg(agg)
                        .sort_values(ascending=False)
                        .head(limit)
                    )

                    artifact = f"{tid}_groupby_{group}_{metric}_{agg}.json"
                    store.write_json(artifact, res.to_dict())

                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer=None,
                        summary=f"{agg}({metric}) grouped by {group} (top {limit})",
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

                else:
                    outputs["failed"].append({"task_id": tid, "reason": f"Unknown task type: {ttype}", "task": t})

            except Exception as e:
                logger.exception(f"Task {tid} failed")
                outputs["failed"].append({"task_id": tid, "reason": str(e), "task": t})

        store.write_json("metrics_outputs.json", outputs)
        logger.info("Wrote metrics_outputs.json")
        return outputs