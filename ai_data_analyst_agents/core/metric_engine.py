from __future__ import annotations

from typing import Any, Dict, List
import ast
import operator as op

import numpy as np
import pandas as pd

from ai_data_analyst_agents.core.kpi_templates import KPI_TEMPLATE_LIBRARY


def _pick_column(candidates: List[str], columns: List[str]) -> str | None:
    colset = set(columns)
    for c in candidates:
        if c in colset:
            return c
    return None


def _safe_agg(series: pd.Series, agg: str) -> float:
    s = pd.to_numeric(series, errors="coerce")
    agg = (agg or "sum").lower().strip()
    if agg == "sum":
        return float(s.sum())
    if agg == "mean":
        return float(s.mean())
    if agg == "count":
        return float(series.count())
    if agg == "min":
        return float(s.min())
    if agg == "max":
        return float(s.max())
    if agg == "median":
        return float(s.median())
    raise ValueError(f"Unsupported agg: {agg}")


def _safe_eval_expr(expr: str, vars_map: Dict[str, float]) -> float:
    """
    Evaluate a basic arithmetic expression safely.
    Allowed: +, -, *, /, **, unary +/-, numeric constants, variable names.
    """
    allowed_bin = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
    }
    allowed_unary = {
        ast.UAdd: op.pos,
        ast.USub: op.neg,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in vars_map:
                raise ValueError(f"Unknown variable in expression: {node.id}")
            return float(vars_map[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_bin:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                return float("nan")
            return float(allowed_bin[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary:
            return float(allowed_unary[type(node.op)](_eval(node.operand)))
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree)


def compute_template_kpis(df: pd.DataFrame, domain: str) -> Dict[str, Any]:
    spec = KPI_TEMPLATE_LIBRARY.get(domain)
    if not spec:
        raise ValueError(f"Unknown KPI domain: {domain}")

    out: Dict[str, Any] = {
        "domain": domain,
        "resolved_columns": {},
        "kpis": {},
        "derived_kpis": {},
    }

    cols = df.columns.tolist()
    base_vals: Dict[str, float] = {}
    for metric_def in spec.get("metric_defs", []):
        name = str(metric_def.get("name"))
        agg = str(metric_def.get("agg", "sum"))
        col = _pick_column(list(metric_def.get("candidates", [])), cols)
        if not name:
            continue
        out["resolved_columns"][name] = col
        if col is None:
            out["kpis"][name] = None
            continue
        val = _safe_agg(df[col], agg)
        out["kpis"][name] = val
        base_vals[name] = val

    for d in spec.get("derived_defs", []):
        name = str(d.get("name", "")).strip()
        expr = str(d.get("expr", "")).strip()
        if not name or not expr:
            continue
        try:
            val = _safe_eval_expr(expr, base_vals)
            if np.isfinite(val):
                out["derived_kpis"][name] = float(val)
            else:
                out["derived_kpis"][name] = None
        except Exception:
            out["derived_kpis"][name] = None

    return out


def compute_metric_definition(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supported forms:
    1) direct agg: {name, metric_col, agg?, group_by?}
    2) expression on grouped dataframe: {name, expression, group_by?}
       expression can reference column names directly (e.g., "revenue/quantity").
    """
    name = str(params.get("name", "metric")).strip() or "metric"
    metric_col = params.get("metric_col")
    agg = str(params.get("agg", "sum"))
    group_by = params.get("group_by")
    expression = params.get("expression")

    payload: Dict[str, Any] = {"name": name, "group_by": group_by, "agg": agg}

    if expression:
        expr = str(expression)
        if group_by:
            if str(group_by) not in df.columns:
                raise ValueError(f"group_by column not found: {group_by}")
            grp = df.groupby(str(group_by)).apply(
                lambda x: pd.to_numeric(x.eval(expr), errors="coerce").mean()
            )
            payload["values"] = {str(k): float(v) for k, v in grp.dropna().to_dict().items()}
        else:
            val = pd.to_numeric(df.eval(expr), errors="coerce").mean()
            payload["value"] = float(val)
        return payload

    if not metric_col:
        raise ValueError("metric_definition requires either 'metric_col' or 'expression'.")
    metric_col = str(metric_col)
    if metric_col not in df.columns:
        raise ValueError(f"metric_col not found: {metric_col}")

    if group_by:
        group_by = str(group_by)
        if group_by not in df.columns:
            raise ValueError(f"group_by column not found: {group_by}")
        grp = df.groupby(group_by)[metric_col].agg(agg).sort_values(ascending=False)
        payload["values"] = {str(k): float(v) for k, v in grp.to_dict().items()}
    else:
        payload["value"] = _safe_agg(df[metric_col], agg)
    return payload


def compute_segment_profile(
    df: pd.DataFrame,
    segment_by: str,
    metric: str,
    agg: str = "sum",
    limit: int = 100,
) -> Dict[str, Any]:
    if segment_by not in df.columns:
        raise ValueError(f"segment_by column not found: {segment_by}")
    if metric not in df.columns:
        raise ValueError(f"metric column not found: {metric}")

    res = df.groupby(segment_by)[metric].agg(agg).sort_values(ascending=False)
    if limit > 0:
        res = res.head(limit)
    total = float(res.sum()) if len(res) else 0.0
    rows = []
    for key, val in res.to_dict().items():
        fval = float(val)
        share = (fval / total) if total else 0.0
        rows.append({"segment": str(key), "value": fval, "share_pct": share})

    return {
        "segment_by": segment_by,
        "metric": metric,
        "agg": agg,
        "total_value": total,
        "rows": rows,
        "values": {r["segment"]: r["value"] for r in rows},
    }


def compute_cohort_retention(
    df: pd.DataFrame,
    entity_col: str,
    date_col: str,
    freq: str = "M",
) -> Dict[str, Any]:
    if entity_col not in df.columns:
        raise ValueError(f"entity_col not found: {entity_col}")
    if date_col not in df.columns:
        raise ValueError(f"date_col not found: {date_col}")

    dff = df[[entity_col, date_col]].copy()
    dff[date_col] = pd.to_datetime(dff[date_col], errors="coerce")
    dff = dff.dropna(subset=[entity_col, date_col])
    if dff.empty:
        return {"entity_col": entity_col, "date_col": date_col, "rows": [], "matrix": []}

    period = dff[date_col].dt.to_period(freq)
    dff = dff.assign(event_period=period)
    cohort = dff.groupby(entity_col)["event_period"].min().rename("cohort_period")
    dff = dff.join(cohort, on=entity_col)
    dff["period_number"] = (dff["event_period"] - dff["cohort_period"]).apply(lambda x: int(x.n))

    grouped = dff.groupby(["cohort_period", "period_number"])[entity_col].nunique().reset_index()
    grouped.columns = ["cohort_period", "period_number", "active_entities"]

    cohort_size = grouped[grouped["period_number"] == 0][["cohort_period", "active_entities"]].rename(
        columns={"active_entities": "cohort_size"}
    )
    merged = grouped.merge(cohort_size, on="cohort_period", how="left")
    merged["retention_rate"] = merged["active_entities"] / merged["cohort_size"]

    rows = []
    for _, r in merged.sort_values(["cohort_period", "period_number"]).iterrows():
        rows.append(
            {
                "cohort_period": str(r["cohort_period"]),
                "period_number": int(r["period_number"]),
                "active_entities": int(r["active_entities"]),
                "cohort_size": int(r["cohort_size"]),
                "retention_rate": float(r["retention_rate"]),
            }
        )

    matrix = (
        merged.pivot(index="cohort_period", columns="period_number", values="retention_rate")
        .fillna(0.0)
        .sort_index()
    )
    matrix_rows = []
    for cohort_idx, row in matrix.iterrows():
        row_dict: Dict[str, Any] = {"cohort_period": str(cohort_idx)}
        for k, v in row.to_dict().items():
            row_dict[str(int(k))] = float(v)
        matrix_rows.append(row_dict)

    return {
        "entity_col": entity_col,
        "date_col": date_col,
        "freq": freq,
        "rows": rows,
        "matrix": matrix_rows,
    }
