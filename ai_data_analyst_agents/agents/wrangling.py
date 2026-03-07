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
            "steps": [{"action": "drop_duplicates", "before_rows": before, "after_rows": after}]
        }

        clean.to_csv(store.path("cleaned.csv"), index=False)
        store.register_file("cleaned.csv")
        store.write_json("feature_log.json", feature_log)

        # Make cleaned df available to later agents
        ctx["memory"].set("df.cleaned", clean)

        logger.info("Wrote cleaned.csv + feature_log.json")
        return {"rows_before": before, "rows_after": after, "feature_log": feature_log}
