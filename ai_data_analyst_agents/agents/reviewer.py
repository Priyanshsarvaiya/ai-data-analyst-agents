from __future__ import annotations
from typing import Any, Dict
import re
from ai_data_analyst_agents.core.agent_base import Agent

EV_PATTERN = re.compile(r"\[\[EV:(EV-[a-f0-9]{10})\]\]")

class ReviewerAgent(Agent):
    name = "reviewer"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        store = ctx["store"]
        logger = ctx["logger"]
        evidence_store = ctx["evidence"]

        report_path = store.path("final_report.md")
        report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

        ev_tags = EV_PATTERN.findall(report)
        missing = [ev for ev in ev_tags if ev not in evidence_store.all()]

        status = "pass"
        notes = []

        if not report.strip():
            status = "fail"
            notes.append("final_report.md is empty or missing.")

        if not ev_tags:
            status = "warn" if status != "fail" else status
            notes.append("No evidence tags found. Include [[EV:...]] for numeric claims.")

        if missing:
            status = "fail"
            notes.append(f"Report references missing evidence IDs: {missing}")

        out = {
            "status": status,
            "evidence_tags_found": len(ev_tags),
            "missing_refs": missing,
            "notes": notes,
        }

        store.write_json("review_log.json", out)
        logger.info(f"Reviewer status: {status}")
        return out