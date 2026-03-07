from __future__ import annotations

from typing import Any, Dict
import json
import pandas as pd

from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.tools.plotting_tools import (
    save_histogram,
    save_bar_top_categories,
    save_bar_from_mapping,
    save_line_from_mapping,
    save_heatmap_from_rows,
)


class EDAAgent(Agent):
    name = "eda"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        cfg = ctx["cfg"]
        store = ctx["store"]
        logger = ctx["logger"]

        df: pd.DataFrame = ctx["memory"].get("df.cleaned", ctx["df"])
        metrics_out = ctx["memory"].get("result.metrics", {}) or {}
        charts_dir = store.run_dir / "charts"

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("string")]

        charts: list[str] = []
        question_aware_charts: list[str] = []

        # Build question-aware visuals from computed artifacts first.
        for item in (metrics_out or {}).get("computed", []):
            if len(charts) >= cfg.eda.max_plots:
                break

            artifact = str(item.get("artifact", ""))
            if not artifact.endswith(".json"):
                continue

            p = store.path(artifact)
            if not p.exists():
                continue

            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue

            chart_name = None
            title = artifact.replace(".json", "")
            task_id = str(item.get("task_id", "t"))

            if isinstance(payload, dict) and "filter" in payload and "value" in payload:
                f = payload.get("filter", {})
                if isinstance(f, dict) and len(f) == 1:
                    k = next(iter(f.keys()))
                    v = f[k]
                    try:
                        chart_name = save_bar_from_mapping(
                            {str(v): float(payload.get("value"))},
                            charts_dir,
                            f"qa_filter_{task_id}",
                            f"{k} = {v}",
                            xlabel="value",
                            ylabel=k,
                            top_k=1,
                        )
                    except Exception:
                        chart_name = None

            elif isinstance(payload, dict) and "quantiles" in payload and isinstance(payload.get("quantiles"), dict):
                try:
                    q = {str(k): float(v) for k, v in payload.get("quantiles", {}).items()}
                    chart_name = save_bar_from_mapping(
                        q,
                        charts_dir,
                        f"qa_quantiles_{task_id}",
                        f"Quantiles: {payload.get('column', payload.get('metric', 'metric'))}",
                        xlabel="value",
                        ylabel="quantile",
                        top_k=12,
                    )
                except Exception:
                    chart_name = None

            elif isinstance(payload, dict) and "values" in payload and isinstance(payload.get("values"), dict):
                vals = payload.get("values", {})
                try:
                    chart_name = save_bar_from_mapping(
                        {str(k): float(v) for k, v in vals.items()},
                        charts_dir,
                        f"qa_values_{task_id}",
                        title,
                        xlabel="value",
                        ylabel="segment",
                        top_k=25,
                    )
                except Exception:
                    chart_name = None

            elif isinstance(payload, dict) and "kpis" in payload and isinstance(payload.get("kpis"), dict):
                merged = {}
                for k, v in payload.get("kpis", {}).items():
                    if isinstance(v, (int, float)):
                        merged[str(k)] = float(v)
                for k, v in payload.get("derived_kpis", {}).items():
                    if isinstance(v, (int, float)):
                        merged[str(k)] = float(v)
                if merged:
                    chart_name = save_bar_from_mapping(
                        merged,
                        charts_dir,
                        f"qa_kpis_{task_id}",
                        f"KPI template: {payload.get('domain', '')}",
                        xlabel="value",
                        ylabel="kpi",
                        top_k=25,
                    )

            elif isinstance(payload, dict) and "rows" in payload and isinstance(payload.get("rows"), list):
                rows = payload.get("rows", [])
                if rows and isinstance(rows[0], dict) and "retention_rate" in rows[0]:
                    chart_name = save_heatmap_from_rows(
                        rows,
                        charts_dir,
                        f"qa_cohort_{task_id}",
                        title,
                        x_col="cohort_period",
                        y_col="period_number",
                        value_col="retention_rate",
                    )

            elif isinstance(payload, dict) and all(isinstance(v, (int, float)) for v in payload.values()):
                if "_timeseries_" in artifact:
                    chart_name = save_line_from_mapping(
                        {str(k): float(v) for k, v in payload.items()},
                        charts_dir,
                        f"qa_ts_{task_id}",
                        title,
                        xlabel="time",
                        ylabel="value",
                    )
                else:
                    chart_name = save_bar_from_mapping(
                        {str(k): float(v) for k, v in payload.items()},
                        charts_dir,
                        f"qa_group_{task_id}",
                        title,
                        xlabel="value",
                        ylabel="group",
                        top_k=20,
                    )

            elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
                if "value" in payload[0]:
                    dims = [k for k in payload[0].keys() if k != "value"]
                    if len(dims) >= 2:
                        chart_name = save_heatmap_from_rows(
                            payload,
                            charts_dir,
                            f"qa_heat_{task_id}",
                            title,
                            x_col=dims[0],
                            y_col=dims[1],
                            value_col="value",
                        )
                    elif len(dims) == 1:
                        chart_name = save_bar_from_mapping(
                            {str(r[dims[0]]): float(r["value"]) for r in payload},
                            charts_dir,
                            f"qa_rows_{task_id}",
                            title,
                            xlabel="value",
                            ylabel=dims[0],
                            top_k=25,
                        )

            if chart_name:
                charts.append(chart_name)
                question_aware_charts.append(chart_name)

        # Generic fallback only when no question-aware visuals were generated.
        if not question_aware_charts:
            for col in numeric_cols[:3]:
                name = save_histogram(df, col, charts_dir)
                if name:
                    charts.append(name)

            for col in cat_cols:
                nunique = int(df[col].nunique(dropna=True))
                if 1 < nunique <= cfg.eda.max_unique_for_category:
                    name = save_bar_top_categories(df, col, charts_dir, top_k=15)
                    if name:
                        charts.append(name)
                if len(charts) >= cfg.eda.max_plots:
                    break

        summary = {
            "numeric_columns": numeric_cols,
            "categorical_candidates": cat_cols,
            "charts": charts,
            "question_aware_charts": question_aware_charts,
        }

        store.write_json("eda_summary.json", summary)
        logger.info("Wrote eda_summary.json")
        return summary
