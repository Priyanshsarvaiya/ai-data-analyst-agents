from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.tools.pandas_tools import basic_dataset_summary, infer_column_profiles, detect_probable_datetime_columns

class ProfilingAgent(Agent):
    name = "profiling"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        store = ctx["store"]
        logger = ctx["logger"]
        df: pd.DataFrame = ctx["df"]

        profile = {
            **basic_dataset_summary(df),
            "datetime_candidates": detect_probable_datetime_columns(df),
            "column_profiles": infer_column_profiles(df),
        }

        store.write_json("data_profile.json", profile)
        logger.info("Wrote data_profile.json")
        return profile