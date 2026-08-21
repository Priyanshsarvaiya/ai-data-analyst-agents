from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from ai_data_analyst_agents.core.agent_base import Agent

class WranglingAgent(Agent):
    name = "wrangling"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        store = ctx["store"]
        logger = ctx["logger"]
        df: pd.DataFrame = ctx["df"]

        clean = df.copy()
        before = len(clean)
        clean = clean.drop_duplicates()
        after = len(clean)

        feature_log = {
            "steps": [
                {
                    "action": "drop_duplicates",
                    "before_rows": before,
                    "after_rows": after,
                    "rows_removed": before - after,
                }
            ]
        }
        profile = ctx["memory"].get("result.profiling") or {}
        for col in profile.get("datetime_candidates", []) or []:
            if col not in clean.columns:
                continue
            parsed = pd.to_datetime(clean[col], errors="coerce")
            valid_rate = float(parsed.notna().mean()) if len(parsed) else 0.0
            if valid_rate >= 0.8:
                clean[col] = parsed
                feature_log["steps"].append(
                    {"action": "parse_datetime", "column": str(col), "valid_rate": valid_rate}
                )

        clean.to_csv(store.path("cleaned.csv"), index=False)
        store.register_file("cleaned.csv")
        store.write_json("feature_log.json", feature_log)

        # Make cleaned df available to later agents
        ctx["memory"].set("df.cleaned", clean)

        logger.info("Wrote cleaned.csv + feature_log.json")
        return {"rows_before": before, "rows_after": after, "feature_log": feature_log}
