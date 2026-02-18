from __future__ import annotations

from typing import Any, Dict
import json
import re

from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.core.openrouter_client import OpenRouterClient


SYSTEM = """You are a senior data analyst planner.

Your job: convert the business question + dataset schema into a SMALL set of concrete computation tasks
that will produce artifacts needed to answer the question with evidence.

IMPORTANT OUTPUT RULES
- Return ONLY valid JSON (no markdown, no commentary, no extra keys).
- Use ONLY columns that exist in the provided schema.
- Use the EXACT param names specified below (case-sensitive).

JSON OUTPUT SCHEMA
{
  "tasks": [
    {
      "id": "T1",
      "type": "groupby_agg" | "filter_agg" | "correlation" | "distribution" | "topk" | "timeseries_agg",
      "params": { }
    }
  ],
  "notes": "short"
}

TASK PARAM SCHEMAS (MUST FOLLOW)
1) groupby_agg
  params MUST include:
    - group_by: <single column name>
    - metric: <single column name>
  params MAY include:
    - agg: "sum" | "mean" | "count" | "min" | "max" | "median"   (default "sum")
    - limit: integer (default 50)

2) filter_agg
  params MUST include:
    - filter_col: <column name>
    - filter_val: <string or number value>
    - metric: <column name>
  params MAY include:
    - agg: "sum" | "mean" | "count" | "min" | "max" | "median" (default "sum")

3) correlation
  params MUST include:
    - x: <numeric column name>
    - y: <numeric column name>

4) distribution
  params MUST include:
    - column: <numeric column name>
  params MAY include:
    - quantiles: [0.05, 0.25, 0.5, 0.75, 0.95] (default)

5) topk
  params MUST include:
    - by: <column name>              (category column)
    - metric: <numeric column name>  (what to rank by)
  params MAY include:
    - agg: "sum" | "mean" | "count"  (default "sum")
    - k: integer (default 10)

6) timeseries_agg
  params MUST include:
    - date_col: <date-like column name>
    - metric: <numeric column name>
  params MAY include:
    - freq: "D" | "W" | "M" | "Q" (default "M")
    - agg: "sum" | "mean" | "count" (default "sum")

PLANNING HEURISTICS (IMPORTANT)
- ALWAYS include tasks that directly answer the question.
- If the question asks for a TOTAL (e.g., "total revenue in India"), include:
  - filter_agg on the relevant filter (country=India) AND metric=revenue
  - (optional) groupby_agg over the filter dimension for context (revenue by country)
- If the question asks "why", "too low", "drop", "vary", "difference", or "drivers", include:
  - distribution on the target metric (e.g., revenue)
  - groupby_agg over the strongest categorical dimensions available among:
      country, product_category, customer_id, order_date
  - correlation between the target metric and likely numeric drivers among:
      quantity, unit_price, discount_pct
  - topk for major categories (country/product_category) by target metric
- Prefer 4-8 tasks maximum. Avoid redundant tasks.
- If required columns do not exist, pick the closest available metric/dimensions.

REMEMBER: Return ONLY the JSON object.
"""

def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Robust JSON extractor for LLM outputs.
    Handles:
      - pure JSON
      - ```json ... ``` fenced blocks
      - extra commentary around a JSON object
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("LLM returned empty response (no JSON to parse).")

    # 1) Try direct JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        raise ValueError("Top-level JSON is not an object/dict.")
    except json.JSONDecodeError:
        pass

    # 2) Try fenced ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        obj = json.loads(m.group(1))
        if isinstance(obj, dict):
            return obj
        raise ValueError("Fenced JSON is not an object/dict.")

    # 3) Try first {...} block anywhere
    m = re.search(r"(\{.*\})", raw, re.DOTALL)
    if m:
        obj = json.loads(m.group(1))
        if isinstance(obj, dict):
            return obj
        raise ValueError("Extracted JSON block is not an object/dict.")

    raise ValueError(f"Could not extract JSON from LLM output.\nRaw output:\n{raw}")

def _fallback_plan(question: str, schema_cols: list[str]) -> Dict[str, Any]:
    # Simple deterministic fallback that helps 80% of questions
    tasks = []

    # If revenue exists, compute by common dimensions
    if "revenue" in schema_cols:
        for dim in ["country", "product_category"]:
            if dim in schema_cols:
                tasks.append({
                    "id": f"T_fallback_{dim}",
                    "type": "groupby_agg",
                    "params": {"group_by": dim, "metric": "revenue", "agg": "sum", "limit": 50}
                })

    # If question includes a country name (very rough), try filter_agg
    q = question.lower()
    if "india" in q and "country" in schema_cols and "revenue" in schema_cols:
        tasks.insert(0, {
            "id": "T_india_revenue",
            "type": "filter_agg",
            "params": {"filter_col": "country", "filter_val": "India", "metric": "revenue", "agg": "sum"}
        })

    if not tasks:
        tasks.append({
            "id": "T_basic_profile",
            "type": "distribution",
            "params": {"columns": schema_cols[:10]}
        })

    return {"tasks": tasks, "notes": "Fallback plan used (LLM JSON invalid or empty)."}


class PlannerAgent(Agent):
    name = "planner"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        cfg = ctx["cfg"]
        store = ctx["store"]
        logger = ctx["logger"]
        question = ctx["business_question"]

        profile = ctx["memory"].get("result.profiling") or {}
        schema_cols = profile.get("columns", []) or []
        schema_dtypes = profile.get("dtypes", {}) or {}

        payload = {
            "business_question": question,
            "schema": {"columns": schema_cols, "dtypes": schema_dtypes},
        }

        client = OpenRouterClient(timeout_s=cfg.llm.timeout_s)

        logger.info("[Planner] Calling OpenRouter for task plan...")
        raw = client.chat(
            model=cfg.llm.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
            temperature=0.1,
            max_tokens=800,
        )

        # Always write raw output for debugging
        store.write_text("planner_raw.txt", raw or "")
        if not raw or not raw.strip():
            logger.warning("[Planner] Empty LLM response. Using fallback plan.")
            plan = _fallback_plan(question, schema_cols)
            store.write_json("analysis_tasks.json", plan)
            return plan

        try:
            plan = _extract_json(raw)
        except Exception as e:
            logger.warning(f"[Planner] Invalid JSON from LLM: {e}. Using fallback plan.")
            plan = _fallback_plan(question, schema_cols)

        store.write_json("analysis_tasks.json", plan)
        logger.info("[Planner] Wrote analysis_tasks.json")
        return plan