from __future__ import annotations
from typing import Any, Dict
import json

from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.core.openrouter_client import OpenRouterClient

SYSTEM_PROMPT = """You are a strict data analyst.

Rules:
- Use ONLY the provided artifacts context.
- Do NOT invent metrics, numbers, columns, or trends not present in the context.
- When stating a numeric fact or concrete claim, include an evidence tag like [[EV:EV-xxxxxxxxxx]].
- If you cannot support a claim with evidence, write: "Not computed in artifacts."

Write a report with sections:
1) Executive Summary
2) Dataset Overview
3) Data Quality Findings
4) EDA Insights
5) Limitations
6) Next Steps
7) Artifacts Index
"""

class ReportingAgent(Agent):
    name = "reporting"

    def run(self, ctx: Dict[str, Any]) -> str:
        cfg = ctx["cfg"]
        store = ctx["store"]
        logger = ctx["logger"]
        evidence_store = ctx["evidence"]

        plan = ctx["memory"].get("result.intake")
        profile = ctx["memory"].get("result.profiling")
        qa = ctx["memory"].get("result.quality")
        eda = ctx["memory"].get("result.eda")

        evidence_map = {
            ev_id: {
                "kind": ev.kind,
                "artifact_path": ev.artifact_path,
                "pointer": ev.pointer,
                "summary": ev.summary,
            }
            for ev_id, ev in evidence_store.all().items()
        }

        artifacts_context = {
            "analysis_plan": plan,
            "data_profile": profile,
            "quality_report": qa,
            "eda_summary": eda,
            "evidence_map": evidence_map,
        }

        client = OpenRouterClient(timeout_s=cfg.llm.timeout_s)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Generate the report now."},
            {"role": "user", "content": "Artifacts Context (JSON):\n" + json.dumps(artifacts_context, indent=2)},
        ]

        logger.info("[OpenRouter] Calling LLM for report generation...")
        report_md = client.chat(
            model=cfg.llm.model,
            messages=messages,
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_tokens,
        )
        logger.info("[OpenRouter] Report generated.")

        store.write_text("final_report.md", report_md)
        return report_md