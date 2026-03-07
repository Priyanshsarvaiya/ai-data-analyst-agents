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
        evidence = ctx["evidence"]
        df: pd.DataFrame = ctx["df"]

        profile: Dict[str, Any] = {
            **basic_dataset_summary(df),
            "datetime_candidates": detect_probable_datetime_columns(df),
            "column_profiles": infer_column_profiles(df),
        }

        sql_schema = ctx.get("sql_schema")
        if isinstance(sql_schema, dict):
            profile["sql_schema"] = {
                "dialect": sql_schema.get("dialect"),
                "table_count": len(sql_schema.get("tables", []) or []),
                "tables": [
                    {
                        "name": t.get("name"),
                        "n_rows": t.get("n_rows"),
                        "columns": [c.get("name") for c in (t.get("columns", []) or [])],
                        "primary_key": t.get("primary_key", []),
                    }
                    for t in (sql_schema.get("tables", []) or [])
                ],
                "relationship_count": len(sql_schema.get("relationships", []) or []),
            }
            evidence.add(
                kind="json",
                artifact_path="db_schema.json",
                pointer=None,
                summary="Database schema and table relationships",
            )

        store.write_json("data_profile.json", profile)
        evidence.add(
            kind="metric",
            artifact_path="data_profile.json",
            pointer="n_rows",
            summary="Row count in dataset",
        )
        evidence.add(
            kind="metric",
            artifact_path="data_profile.json",
            pointer="n_cols",
            summary="Column count in dataset",
        )
        logger.info("Wrote data_profile.json")
        return profile
