from __future__ import annotations
from typing import Any, Dict
import pandas as pd

from ai_data_analyst_agents.core.agent_base import Agent

def _ensure_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"Column not found: {col}")

class MetricsAgent(Agent):
    name = "metrics"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        store = ctx["store"]
        logger = ctx["logger"]
        evidence = ctx["evidence"]

        df: pd.DataFrame = ctx["memory"].get("df.cleaned", ctx["df"])
        task_plan = ctx["memory"].get("result.planner", {})  # output from PlannerAgent
        tasks = task_plan.get("tasks", [])

        outputs: Dict[str, Any] = {"computed": [], "failed": []}

        for t in tasks:
            tid = t.get("id", "T?")
            ttype = t.get("type")
            p = t.get("params", {})

            try:
                if ttype == "groupby_agg":
                    group = p["group_by"]
                    metric = p["metric"]
                    agg = p.get("agg", "sum")
                    _ensure_col(df, group); _ensure_col(df, metric)

                    res = (
                        df.groupby(group)[metric]
                        .agg(agg)
                        .sort_values(ascending=False)
                        .head(p.get("limit", 50))
                    )
                    artifact = f"{tid}_groupby_{group}_{metric}_{agg}.json"
                    store.write_json(artifact, res.to_dict())

                    ev = evidence.add(
                        kind="json",
                        artifact_path=artifact,
                        pointer=None,
                        summary=f"{agg}({metric}) grouped by {group}",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "filter_agg":
                    filt_col = p["filter_col"]
                    filt_val = p["filter_val"]
                    metric = p["metric"]
                    agg = p.get("agg", "sum")
                    _ensure_col(df, filt_col); _ensure_col(df, metric)

                    dff = df[df[filt_col].astype(str) == str(filt_val)]
                    val = float(getattr(dff[metric], agg)())
                    artifact = f"{tid}_filter_{filt_col}_{metric}_{agg}.json"
                    store.write_json(artifact, {"filter": {filt_col: filt_val}, "value": val})

                    ev = evidence.add(
                        kind="metric",
                        artifact_path=artifact,
                        pointer="value",
                        summary=f"{agg}({metric}) where {filt_col}={filt_val}",
                    )
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                elif ttype == "correlation":
                    x = p["x"]; y = p["y"]
                    _ensure_col(df, x); _ensure_col(df, y)
                    pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
                    if len(pair) < 3:
                        raise ValueError(f"Not enough numeric data to compute correlation for {x} vs {y}")

                    corr_val = pair[x].corr(pair[y])  # returns float | NaN
                    if pd.isna(corr_val):
                        raise ValueError(f"Correlation is NaN for {x} vs {y}")

                    corr = float(corr_val)

                    artifact = f"{tid}_corr_{x}_{y}.json"
                    store.write_json(artifact, {"x": x, "y": y, "corr": corr})

                    ev = evidence.add(kind="metric", artifact_path=artifact, pointer="corr",
                                     summary=f"Correlation between {x} and {y}")
                    outputs["computed"].append({"task_id": tid, "artifact": artifact, "evidence_id": ev.id})

                else:
                    outputs["failed"].append({"task_id": tid, "reason": f"Unknown task type: {ttype}"})

            except Exception as e:
                logger.exception(f"Task {tid} failed")
                outputs["failed"].append({"task_id": tid, "reason": str(e)})

        store.write_json("metrics_outputs.json", outputs)
        logger.info("Wrote metrics_outputs.json")
        return outputs