from __future__ import annotations
from typing import Any, Dict
import json

from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.core.openrouter_client import OpenRouterClient

SYSTEM_PROMPT = """You are a strict, evidence-grounded data analyst.

NON-NEGOTIABLE RULES
- Use ONLY the provided artifacts context.
- Do NOT invent metrics, numbers, columns, entities, or trends not present in the context.
- Every numeric value or concrete claim MUST include an evidence tag like [[EV:EV-xxxxxxxxxx]].
- If you cannot support a claim with evidence, write EXACTLY: "Not computed in artifacts."
- Prefer computed metric artifacts (metrics_outputs.json, *_filter_*.json, *_groupby_*.json, *_corr_*.json) over descriptive summaries.
- Do not include generic boilerplate. Every sentence must either (a) answer the business question, or (b) justify limitations/next steps.

GOAL
- Directly answer the business question: "{business_question}"
- If the question is a "why/driver" question, you MUST summarize drivers using evidence:
  - compare across segments (e.g., country/category)
  - reference distributions/quantiles if available
  - reference correlations if available
  - if none computed, clearly state what is missing and what needs to be computed.

REPORT FORMAT (STRICT)
Produce exactly these sections in this order:

# Data Analysis Report

## 1) Executive Summary
- Restate the business question in one sentence.
- Answer it in 1-3 bullets using evidence.
- If the answer is not computable from artifacts: say "Not computed in artifacts." and name which missing computations would be required (e.g., "revenue by country").

## 2) Question Answer (Evidence)
- Provide a short, direct answer paragraph.
- Then include a compact bullet list of the key evidence items (each bullet must include an evidence tag).
- If the question is about a total/aggregation, show the computed value(s).
- If the question is about "why", list the top 2–5 quantitative drivers/contrasts (e.g., top segments, strongest correlations).

## 3) Dataset Overview
- Rows, columns, and key fields relevant to the question (do NOT list every column unless needed).
- Include data types ONLY if relevant to computations performed.

## 4) Data Quality Findings
- Missingness, duplicates, and any warnings. If not present: "Not computed in artifacts."
- Only mention outliers if an outlier artifact exists.

## 5) Analysis Outputs
- Summarize the outputs that were actually computed:
  - groupby tables (top categories/countries)
  - filter aggregations (e.g., India revenue)
  - correlations (if computed)
  - distributions/quantiles (if computed)
- Each bullet must cite evidence.
- If metrics tasks failed, state that and reference the failure entries (if present).

## 6) Limitations
- Must be specific to missing artifacts/failed computations.
- Include at most 5 bullets.
- If everything needed exists, keep this section minimal.

## 7) Next Steps
- Must be actionable computations or checks that would strengthen the answer.
- Tie each next step to the business question.

## 8) Artifacts Index
- Provide a compact table: Evidence ID | Kind | Artifact | Pointer | Summary
- Only include evidence IDs that were referenced in the report.
"""

class ReportingAgent(Agent):
    name = "reporting"

    def run(self, ctx: Dict[str, Any]) -> str:
        cfg = ctx["cfg"]
        store = ctx["store"]
        logger = ctx["logger"]
        evidence_store = ctx["evidence"]
        business_question: str = ctx.get("business_question", "").strip()

        # Core artifacts from earlier agents
        plan = ctx["memory"].get("result.intake")
        profile = ctx["memory"].get("result.profiling")
        qa = ctx["memory"].get("result.quality")
        eda = ctx["memory"].get("result.eda")

        # ✅ NEW: planner + metrics outputs (Phase 2)
        planner_out = ctx["memory"].get("result.planner")
        metrics_out = ctx["memory"].get("result.metrics")

        # Evidence map (only what exists)
        evidence_map = {
            ev_id: {
                "kind": ev.kind,
                "artifact_path": ev.artifact_path,
                "pointer": ev.pointer,
                "summary": ev.summary,
            }
            for ev_id, ev in evidence_store.all().items()
        }

        # ✅ Include everything the reporter needs to answer the question
        artifacts_context = {
            "business_question": business_question,
            "analysis_plan": plan,
            "data_profile": profile,
            "quality_report": qa,
            "eda_summary": eda,
            "planner_output": planner_out,        # includes tasks list
            "metrics_outputs": metrics_out,       # includes computed/failed/skipped
            "evidence_map": evidence_map,
        }

        # Inject the business question into the system prompt (safer than relying on user msg)
        system_prompt = SYSTEM_PROMPT.format(business_question=business_question)

        client = OpenRouterClient(timeout_s=cfg.llm.timeout_s)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Business Question:\n{business_question}\n\n"
                    "Generate the report now. Answer the business question first, "
                    "then support with evidence tags."
                ),
            },
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