from __future__ import annotations
from typing import Any, Dict
from ai_data_analyst_agents.core.agent_base import Agent

class IntakeAgent(Agent):
    name = "intake"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        store = ctx["store"]
        logger = ctx["logger"]
        question = ctx["business_question"]

        plan = {
            "business_question": question,
            "assumptions": [
                "All computations are based only on provided dataset artifacts.",
                "If time columns exist, they may need parsing for time-based insights.",
            ],
            "suggested_slices": ["country", "product_category"],
            "requested_metrics": ["revenue", "orders", "avg_order_value"],
        }

        store.write_json("analysis_plan.json", plan)
        logger.info("Wrote analysis_plan.json")
        return plan