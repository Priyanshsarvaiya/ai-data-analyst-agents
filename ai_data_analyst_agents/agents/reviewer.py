from __future__ import annotations
from typing import Any, Dict
import re
from ai_data_analyst_agents.core.agent_base import Agent

EV_PATTERN = re.compile(r"\[\[(?:EV:)?(EV-[a-f0-9]{10})\]\]|\[\[EV-([a-f0-9]{10})\]\]")
RAW_EV_TOKEN_PATTERN = re.compile(r"\[\[EV[^\]]*\]\]")
NUM_PATTERN = re.compile(r"(?<![\w-])\d[\d,]*(?:\.\d+)?")
CITE_PATTERN = re.compile(r"\[(\d{1,4})\]")
CITE_MAP_LINE_PATTERN = re.compile(r"^\|\s*\[(\d{1,4})\]\s*\|\s*(EV-[a-f0-9]{10})\s*\|")
CAUSAL_PATTERN = re.compile(r"\b(caused|causes|proved|proves|guarantees|drives growth|causal(?:ly)?)\b", re.IGNORECASE)
SIGNIFICANCE_PATTERN = re.compile(r"\b(statistically significant|significant improvement|significant lift)\b", re.IGNORECASE)


def _section(text: str, heading: str) -> str:
    m = re.search(rf"## {re.escape(heading)}(.*?)(?:\n## |\Z)", text, re.DOTALL)
    return m.group(1) if m else ""


def _citation_map(report: str) -> dict[int, str]:
    sec = _section(report, "8) Evidence References")
    if not sec:
        sec = _section(report, "9) Evidence References")
    out: dict[int, str] = {}
    if not sec:
        return out
    for ln in sec.splitlines():
        m = CITE_MAP_LINE_PATTERN.search(ln.strip())
        if not m:
            continue
        out[int(m.group(1))] = m.group(2)
    return out


def _numeric_lines_without_evidence(section_text: str, citation_map: dict[int, str]) -> list[str]:
    lines = [ln.strip() for ln in section_text.splitlines() if ln.strip()]
    bad = []
    for ln in lines:
        if ln.startswith("|"):  # table rows
            continue
        has_ev_tag = "[[EV:" in ln or "[[EV-" in ln
        cite_nums = [int(n) for n in CITE_PATTERN.findall(ln)]
        has_citation = any(n in citation_map for n in cite_nums)
        if NUM_PATTERN.search(ln) and not has_ev_tag and not has_citation:
            bad.append(ln)
    return bad


def _violation(rule_id: str, message: str) -> dict[str, str]:
    return {"rule": rule_id, "severity": "fail", "message": message}

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
        citation_map = _citation_map(report)
        citation_ev_ids = list(citation_map.values())

        cited_numbers = {int(n) for n in CITE_PATTERN.findall(report)}
        unmapped_cites = sorted(n for n in cited_numbers if n not in citation_map)

        all_ref_ids = [*ev_tags, *citation_ev_ids]
        missing = [ev for ev in all_ref_ids if ev not in evidence_store.all()]
        referenced_evidence = [evidence_store.all().get(ev_id) for ev_id in set(all_ref_ids) if ev_id in evidence_store.all()]
        has_statistical_evidence = any(
            ev is not None and str(ev.artifact_path).startswith("statistics/") for ev in referenced_evidence
        ) or "### Statistical Questions Evaluated" in report

        violations: list[dict[str, str]] = []

        if not report.strip():
            violations.append(_violation("report.non_empty", "final_report.md is empty or missing."))

        unresolved_tokens = []
        for token in RAW_EV_TOKEN_PATTERN.findall(report):
            m = re.match(r"\[\[(?:EV:)?(EV-[a-f0-9]{10})\]\]", token)
            if not m:
                unresolved_tokens.append(token)
                continue
            ev_id = m.group(1)
            if ev_id not in evidence_store.all():
                unresolved_tokens.append(token)
        if unresolved_tokens:
            violations.append(
                _violation("evidence.placeholders", f"Unresolved evidence placeholders/tags: {unresolved_tokens[:8]}")
            )

        if not ev_tags and not citation_map:
            violations.append(_violation("evidence.references", "No evidence references found in report."))

        if unmapped_cites:
            violations.append(_violation("evidence.citation_map", f"Unmapped numeric citations found: {unmapped_cites}"))

        if missing:
            violations.append(_violation("evidence.existence", f"Report references missing evidence IDs: {missing}"))

        offenders = []
        for sec in ["1) Executive Summary", "2) Question Answer (Evidence)", "5) Analysis Outputs"]:
            offenders.extend(_numeric_lines_without_evidence(_section(report, sec), citation_map))
        if offenders:
            violations.append(
                _violation(
                    "claims.numeric_supported",
                    "Numeric claims without evidence support: " + "; ".join(offenders[:5]),
                )
            )

        limitations_text = _section(report, "6) Limitations")
        if not limitations_text.strip():
            violations.append(_violation("limitations.present", "Limitations section is empty."))
        elif "not computed in artifacts" in limitations_text.lower():
            metrics_path = store.path("metrics_outputs.json")
            if metrics_path.exists():
                try:
                    import json as _json

                    metrics_out = _json.loads(metrics_path.read_text(encoding="utf-8"))
                    if not (metrics_out.get("failed") or metrics_out.get("skipped")):
                        violations.append(
                            _violation(
                                "limitations.relevance",
                                "Limitations mention missing computations, but no failed/skipped tasks were recorded.",
                            )
                        )
                except Exception:
                    pass

        if has_statistical_evidence:
            if "### Statistical Limitations" not in report:
                violations.append(
                    _violation(
                        "stats.limitations_subsection",
                        "Statistical evidence exists but report lacks '### Statistical Limitations' subsection.",
                    )
                )

            if SIGNIFICANCE_PATTERN.search(report):
                has_pvalue = bool(re.search(r"\bp-value\b|\bp\s*=\s*\d", report, re.IGNORECASE))
                has_ci = "### Confidence Intervals" in report
                has_effect = "### Effect Sizes" in report
                if not (has_pvalue and has_ci and has_effect):
                    violations.append(
                        _violation(
                            "stats.significance_support",
                            "Significance claims require p-value, confidence intervals, and effect sizes.",
                        )
                    )

            stat_claim_lines = []
            for block in [
                _section(report, "1) Executive Summary"),
                _section(report, "2) Question Answer (Evidence)"),
                _section(report, "5) Analysis Outputs"),
            ]:
                for line in block.splitlines():
                    if "business question" in line.lower():
                        continue
                    stat_claim_lines.append(line)
            stat_claim_text = "\n".join(stat_claim_lines)
            if CAUSAL_PATTERN.search(stat_claim_text):
                violations.append(
                    _violation(
                        "stats.causal_language",
                        "Causal wording is blocked for statistical summaries unless explicitly justified.",
                    )
                )

        status = "fail" if violations else "pass"
        notes = [v["message"] for v in violations]
        out = {
            "status": status,
            "evidence_tags_found": len(ev_tags),
            "missing_refs": missing,
            "notes": notes,
            "violations": violations,
        }

        store.write_json("review_log.json", out)
        logger.info(f"Reviewer status: {status}")
        return out
