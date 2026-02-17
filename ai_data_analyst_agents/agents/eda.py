from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.tools.plotting_tools import save_histogram, save_bar_top_categories

class EDAAgent(Agent):
    name = "eda"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        cfg = ctx["cfg"]
        store = ctx["store"]
        logger = ctx["logger"]

        df: pd.DataFrame = ctx["memory"].get("df.cleaned", ctx["df"])
        charts_dir = store.run_dir / "charts"

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype).startswith("string")]

        charts = []

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
        }

        store.write_json("eda_summary.json", summary)
        logger.info("Wrote eda_summary.json")
        return summary