from __future__ import annotations
from typing import Any, Dict
import json

from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.core.openrouter_client import OpenRouterClient

SYSTEM = """You are a senior data analyst planner.
Given:
- business_question
- dataset schema (columns + dtypes)
Return ONLY valid JSON matching the schema below. No extra text.

JSON schema:
{
  "tasks": [
    {
      "id": "T1",
      "type": "groupby_agg" | "filter_agg" | "topk" | "timeseries_agg" | "correlation" | "distribution",
      "params": { ... }
    }
  ],
  "notes": "short"
}

Rules:
- Prefer simple, reliable tasks.
- Only reference columns that exist in schema.
- If question asks 'why', include tasks that compare segments and find drivers (topk + groupby + correlation).
"""

class PlannerAgent(Agent):
    name = "planner"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        cfg = ctx["cfg"]
        store = ctx["store"]
        logger = ctx["logger"]
        question = ctx["business_question"]
        profile = ctx["memory"].get("result.profiling")  # from profiling agent

        schema = {
            "columns": profile.get("columns", []),
            "dtypes": profile.get("dtypes", {}),
        }

        client = OpenRouterClient(timeout_s=cfg.llm.timeout_s)

        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"business_question: {question}"},
            {"role": "user", "content": "schema:\n" + json.dumps(schema, indent=2)},
        ]

        logger.info("[OpenRouter] Planning tasks...")
        raw = client.chat(
            model=cfg.llm.model,
            messages=msgs,
            temperature=0.1,
            max_tokens=800,
        )

        plan = json.loads(raw)  # strict JSON only
        store.write_json("analysis_tasks.json", plan)
        logger.info("Wrote analysis_tasks.json")
        return plan