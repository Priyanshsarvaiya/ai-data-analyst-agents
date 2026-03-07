from __future__ import annotations
from typing import Any, Dict
from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.core.kpi_templates import detect_business_domain

class IntakeAgent(Agent):
    name = "intake"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        store = ctx["store"]
        logger = ctx["logger"]
        question = ctx["business_question"]
        df = ctx["df"]
        source = ctx.get("source", {"type": "csv"})
        domain = detect_business_domain(question, list(df.columns))

        plan = {
            "business_question": question,
            "source_type": source.get("type", "csv"),
            "suggested_domain": domain,
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
