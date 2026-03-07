from __future__ import annotations
from typing import Any, Dict
import re
from ai_data_analyst_agents.core.agent_base import Agent

EV_PATTERN = re.compile(r"\[\[(?:EV:)?(EV-[a-f0-9]{10})\]\]|\[\[EV-([a-f0-9]{10})\]\]")
NUM_PATTERN = re.compile(r"(?<![\w-])\d[\d,]*(?:\.\d+)?")


def _section(text: str, heading: str) -> str:
    m = re.search(rf"## {re.escape(heading)}(.*?)(?:\n## |\Z)", text, re.DOTALL)
    return m.group(1) if m else ""


def _numeric_lines_without_ev(section_text: str) -> list[str]:
    lines = [ln.strip() for ln in section_text.splitlines() if ln.strip()]
    bad = []
    for ln in lines:
        if ln.startswith("|"):  # table rows
            continue
        if NUM_PATTERN.search(ln) and "[[EV:" not in ln and "[[EV-" not in ln:
            bad.append(ln)
    return bad

class ReviewerAgent(Agent):
    name = "reviewer"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        store = ctx["store"]
        logger = ctx["logger"]
        evidence_store = ctx["evidence"]

        report_path = store.path("final_report.md")
        report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

        raw_matches = EV_PATTERN.findall(report)
        ev_tags = []
        for a, b in raw_matches:
            ev_id = a or b
            if ev_id:
                ev_tags.append(ev_id)
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

        # Enforce evidence-linking on key answer sections.
        offenders = []
        for sec in ["1) Executive Summary", "2) Question Answer (Evidence)", "5) Analysis Outputs"]:
            offenders.extend(_numeric_lines_without_ev(_section(report, sec)))
        if offenders:
            status = "fail"
            notes.append(
                "Numeric claims without evidence tag found in key sections: "
                + "; ".join(offenders[:5])
            )

        out = {
            "status": status,
            "evidence_tags_found": len(ev_tags),
            "missing_refs": missing,
            "notes": notes,
        }

        store.write_json("review_log.json", out)
        logger.info(f"Reviewer status: {status}")
        return out
