from __future__ import annotations
from typing import Any, Dict
import json
import re

from ai_data_analyst_agents.core.agent_base import Agent
from ai_data_analyst_agents.core.contracts import (
    ARTIFACT_SCHEMA_VERSION,
    validate_report_metadata_contract,
)
from ai_data_analyst_agents.core.evidence import EvidenceRef
from ai_data_analyst_agents.core.openrouter_client import OpenRouterClient
from ai_data_analyst_agents.core.security import redact_payload_for_llm

SYSTEM_PROMPT = """You are a strict, evidence-grounded data analyst.

NON-NEGOTIABLE RULES
- Use ONLY the provided artifacts context.
- Do NOT invent metrics, numbers, columns, entities, or trends not present in the context.
- Every numeric value or concrete claim MUST include an evidence tag like [[EV:EV-xxxxxxxxxx]].
- If you cannot support a claim with evidence, write EXACTLY: "Not computed in artifacts."
- For statistical results, never use causal wording unless the study design explicitly justifies it in artifacts.
- Do not call a result significant without including p-value context plus confidence interval/effect-size support.
- Prefer computed metric artifacts (metrics_outputs.json, *_filter_*.json, *_groupby_*.json, *_corr_*.json) over descriptive summaries.
- Use `evidence_payloads` values directly whenever available; do not claim values are missing if payloads contain them.
- Do not include generic boilerplate. Every sentence must either (a) answer the business question, or (b) justify limitations/next steps.
- Limitations must only include truly missing computations required for the asked question. If required computations exist, state that clearly instead of generic caveats.
- Next steps must be question-specific and conditional on actual remaining gaps (do not include generic placeholder steps).

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
  - statistical analyses (if computed)
- Each bullet must cite evidence.
- If metrics tasks failed, state that and reference the failure entries (if present).
- If statistical outputs exist, add compact markdown subsections inside section 5 with these exact headings when relevant:
  - `### Statistical Questions Evaluated`
  - `### Methods Chosen and Why`
  - `### Assumption Checks`
  - `### Results`
  - `### Confidence Intervals`
  - `### Effect Sizes`
  - `### A/B Test Readout`
  - `### Regression Findings`

## 6) Limitations
- Must be specific to missing artifacts/failed computations.
- Include at most 5 bullets.
- If everything needed exists, keep this section minimal.
- If statistical outputs exist, include a `### Statistical Limitations` subsection and keep wording conservative.

## 7) Next Steps
- Must be actionable computations or checks that would strengthen the answer.
- Tie each next step to the business question.
"""

EV_TAG_PATTERN = re.compile(r"\[\[(?:EV:)?(EV-[a-f0-9]{10})\]\]|\[\[EV-([a-f0-9]{10})\]\]")
RAW_EV_TOKEN_PATTERN = re.compile(r"\[\[EV[^\]]*\]\]")
NUMERIC_PATTERN = re.compile(r"(?<![\w-])\d[\d,]*(?:\.\d+)?")
CITATION_PATTERN = re.compile(r"\[(\d{1,4})\]")


def _safe_read_json(path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pointer_value(payload: Any, pointer: str | None) -> Any:
    if pointer is None or payload is None:
        return None
    if isinstance(payload, dict) and pointer in payload:
        return payload[pointer]
    cur = payload
    for part in str(pointer).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _compact_json(value: Any, max_items: int = 40) -> Any:
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= max_items:
                out["__truncated__"] = f"dict truncated at {max_items} items"
                break
            out[k] = _compact_json(v, max_items=max_items)
        return out
    if isinstance(value, list):
        if len(value) <= max_items:
            return [_compact_json(v, max_items=max_items) for v in value]
        return [_compact_json(v, max_items=max_items) for v in value[:max_items]] + [
            f"... truncated {len(value) - max_items} items"
        ]
    return value


def _normalize_evidence_tags(text: str) -> str:
    out = re.sub(r"\[\[EV-([a-f0-9]{10})\]\]", r"[[EV:EV-\1]]", text)
    out = re.sub(r"\[\[(EV-[a-f0-9]{10})\]\]", r"[[EV:\1]]", out)
    return out


def _format_evidence_citations(report: str, evidence_refs: Dict[str, EvidenceRef]) -> str:
    report = (report or "").strip()
    if not report:
        return report

    order: list[str] = []
    ev_to_num: dict[str, int] = {}

    def _replace(match: re.Match[str]) -> str:
        ev_id = match.group(1) or (f"EV-{match.group(2)}" if match.group(2) else "")
        if not ev_id:
            return ""
        if ev_id not in ev_to_num:
            ev_to_num[ev_id] = len(order) + 1
            order.append(ev_id)
        return f"[{ev_to_num[ev_id]}]"

    body = EV_TAG_PATTERN.sub(_replace, report)
    body = re.sub(r"\n## [89]\) Evidence References.*\Z", "", body, flags=re.DOTALL).rstrip()

    if not order:
        return body + "\n"

    lines = [
        "## 8) Evidence References",
        "| Ref | Evidence ID | Artifact | Pointer | Summary |",
        "|---|---|---|---|---|",
    ]
    for ev_id in order:
        num = ev_to_num[ev_id]
        ev = evidence_refs.get(ev_id)
        if ev is None:
            lines.append(f"| [{num}] | {ev_id} | - | - | Missing evidence ID |")
            continue
        pointer = (ev.pointer if ev.pointer is not None else "null").replace("|", "/")
        summary = (ev.summary or "").replace("\n", " ").replace("|", "/").strip()
        artifact = (ev.artifact_path or "").replace("|", "/").strip()
        lines.append(f"| [{num}] | {ev_id} | {artifact} | {pointer} | {summary} |")

    return body + "\n\n" + "\n".join(lines) + "\n"


def _report_section(report: str, heading: str) -> str:
    pattern = rf"## {re.escape(heading)}(.*?)(?:\n## |\Z)"
    m = re.search(pattern, report, re.DOTALL)
    return m.group(1) if m else ""


def _collect_report_consistency_issues(report: str, evidence_ids: set[str]) -> list[str]:
    issues: list[str] = []
    txt = (report or "").strip()
    if not txt:
        return ["Empty report."]

    unresolved = []
    for token in RAW_EV_TOKEN_PATTERN.findall(txt):
        m = re.match(r"\[\[(?:EV:)?(EV-[a-f0-9]{10})\]\]", token)
        if not m:
            unresolved.append(token)
            continue
        ev_id = m.group(1)
        if ev_id not in evidence_ids:
            unresolved.append(token)
    if unresolved:
        issues.append(f"Unresolved evidence placeholders/tags found: {unresolved[:6]}")

    required_sections = [
        "1) Executive Summary",
        "2) Question Answer (Evidence)",
        "3) Dataset Overview",
        "4) Data Quality Findings",
        "5) Analysis Outputs",
        "6) Limitations",
        "7) Next Steps",
    ]
    for sec in required_sections:
        if not _report_section(txt, sec).strip():
            issues.append(f"Missing or empty required section: {sec}")

    citation_map = {}
    sec8 = _report_section(txt, "8) Evidence References")
    if sec8:
        for ln in sec8.splitlines():
            m = re.search(r"\|\s*\[(\d{1,4})\]\s*\|\s*(EV-[a-f0-9]{10})\s*\|", ln.strip())
            if m:
                citation_map[int(m.group(1))] = m.group(2)

    for sec in ["1) Executive Summary", "2) Question Answer (Evidence)", "5) Analysis Outputs"]:
        body = _report_section(txt, sec)
        for line in [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("|")]:
            if NUMERIC_PATTERN.search(line):
                has_ev = bool(EV_TAG_PATTERN.search(line))
                cite_nums = [int(x) for x in CITATION_PATTERN.findall(line)]
                has_num_cite = any(n in citation_map for n in cite_nums)
                if not has_ev and not has_num_cite:
                    issues.append(f"Unsupported numeric claim in {sec}: {line[:160]}")

    high_entities = set()
    low_entities = set()
    for line in txt.splitlines():
        line_l = line.lower()
        m_h = re.search(r"([a-z0-9_ \-/]+?)\s+is\s+(?:the\s+)?highest", line_l)
        m_l = re.search(r"([a-z0-9_ \-/]+?)\s+is\s+(?:the\s+)?lowest", line_l)
        if m_h:
            high_entities.add(m_h.group(1).strip())
        if m_l:
            low_entities.add(m_l.group(1).strip())
    clashes = sorted(x for x in high_entities if x in low_entities and x)
    if clashes:
        issues.append(f"Potential contradiction: same entity marked highest and lowest: {clashes}")

    return issues


def _report_needs_fallback(report: str, metrics_out: Dict[str, Any], evidence_payloads: Dict[str, Any]) -> bool:
    if not report.strip():
        return True

    computed = (metrics_out or {}).get("computed", []) or []
    if not computed:
        return False

    has_payload_values = False
    for ev_data in evidence_payloads.values():
        pointer_val = ev_data.get("pointer_value")
        payload = ev_data.get("payload")
        if isinstance(pointer_val, (int, float)):
            has_payload_values = True
            break
        if isinstance(payload, dict) and payload:
            has_payload_values = True
            break
    if not has_payload_values:
        return False

    s2 = _report_section(report, "2) Question Answer (Evidence)")
    if not s2:
        return True
    if not re.search(r"\d", s2):
        return True

    bad_phrases = [
        "not provided in the context",
        "values not provided",
        "not computed in artifacts",
    ]
    s12 = (_report_section(report, "1) Executive Summary") + "\n" + s2).lower()
    return any(p in s12 for p in bad_phrases)


def _extract_statistical_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in entries:
        payload = entry.get("payload")
        if isinstance(payload, dict) and payload.get("analysis_type") in {"hypothesis_test", "ab_test", "regression"}:
            out.append(entry)
    return out


def _format_statistical_subsections(
    stats_entries: list[dict[str, Any]],
    tag_fn,
) -> str:
    if not stats_entries:
        return ""

    question_lines: list[str] = []
    method_lines: list[str] = []
    assumption_lines: list[str] = []
    result_lines: list[str] = []
    ci_lines: list[str] = []
    effect_lines: list[str] = []
    ab_lines: list[str] = []
    regression_lines: list[str] = []
    limitation_lines: list[str] = []

    for entry in stats_entries:
        payload = entry.get("payload") or {}
        ev_tag = tag_fn(entry.get("ev_id"))
        method = str(payload.get("method", "unknown"))
        analysis_type = str(payload.get("analysis_type", "statistical"))
        plain_language = str(payload.get("plain_language", "")).strip()
        method_reason = str(payload.get("method_reason", "")).strip()
        question_lines.append(f"- {analysis_type}: {plain_language or 'Computed statistical analysis.'} {ev_tag}".rstrip())
        method_lines.append(f"- {method}: {method_reason or 'Method rationale not available.'} {ev_tag}".rstrip())

        for check in payload.get("assumptions", []) or []:
            if not isinstance(check, dict):
                continue
            status = check.get("passed")
            label = "pass" if status is True else ("warn" if status is None else "fail")
            assumption_lines.append(f"- {check.get('name', 'assumption')} ({label}): {check.get('detail', '')} {ev_tag}".rstrip())

        interpretation = str(payload.get("interpretation", "")).strip()
        p_value = payload.get("p_value")
        test_statistic = payload.get("test_statistic")
        results_text = interpretation or plain_language or "Result computed."
        if isinstance(p_value, (int, float)):
            results_text += f" p-value={float(p_value):.4f}."
        if isinstance(test_statistic, (int, float)):
            results_text += f" statistic={float(test_statistic):.4f}."
        result_lines.append(f"- {results_text} {ev_tag}".rstrip())

        for ci in payload.get("confidence_intervals", []) or []:
            if not isinstance(ci, dict):
                continue
            try:
                ci_lines.append(
                    f"- {ci.get('parameter')}: {float(ci.get('point_estimate')):.4f} "
                    f"[{float(ci.get('lower_bound')):.4f}, {float(ci.get('upper_bound')):.4f}] "
                    f"at {float(ci.get('confidence_level', 0.95)):.0%} {ev_tag}".rstrip()
                )
            except Exception:
                continue

        for eff in payload.get("effect_sizes", []) or []:
            if not isinstance(eff, dict):
                continue
            try:
                effect_lines.append(
                    f"- {eff.get('name')}: {float(eff.get('value')):.4f} "
                    f"({eff.get('interpretation', 'context-dependent')}) {ev_tag}".rstrip()
                )
            except Exception:
                continue

        if analysis_type == "ab_test":
            metrics = payload.get("metrics", {}) or {}
            if metrics:
                ab_lines.append(f"- A/B summary: {metrics} {ev_tag}".rstrip())
        if analysis_type == "regression":
            metrics = payload.get("metrics", {}) or {}
            coeffs = (payload.get("extra_outputs", {}) or {}).get("coefficients", []) or []
            coeff_preview = coeffs[:3] if isinstance(coeffs, list) else []
            regression_lines.append(
                f"- Regression fit: metrics={metrics}; coefficient preview={coeff_preview} {ev_tag}".rstrip()
            )

        for limitation in payload.get("limitations", []) or []:
            limitation_lines.append(f"- {limitation} {ev_tag}".rstrip())

    parts = [
        "### Statistical Questions Evaluated",
        "\n".join(question_lines) if question_lines else "- None",
        "",
        "### Methods Chosen and Why",
        "\n".join(method_lines) if method_lines else "- None",
        "",
        "### Assumption Checks",
        "\n".join(assumption_lines[:10]) if assumption_lines else "- None",
        "",
        "### Results",
        "\n".join(result_lines) if result_lines else "- None",
        "",
        "### Confidence Intervals",
        "\n".join(ci_lines[:10]) if ci_lines else "- None",
        "",
        "### Effect Sizes",
        "\n".join(effect_lines[:10]) if effect_lines else "- None",
    ]
    if ab_lines:
        parts.extend(["", "### A/B Test Readout", "\n".join(ab_lines[:5])])
    if regression_lines:
        parts.extend(["", "### Regression Findings", "\n".join(regression_lines[:5])])
    if limitation_lines:
        parts.extend(["", "### Statistical Limitations", "\n".join(limitation_lines[:8])])
    return "\n".join(parts).strip()


def _strip_empty_statistical_subsections(report: str) -> str:
    if not report.strip():
        return report
    patterns = [
        r"\n### Statistical Questions Evaluated\s*\n(?:- )?Not computed in artifacts\.\s*",
        r"\n### Methods Chosen and Why\s*\n(?:- )?Not computed in artifacts\.\s*",
        r"\n### Assumption Checks\s*\n(?:- )?Not computed in artifacts\.\s*",
        r"\n### Results\s*\n(?:- )?Not computed in artifacts\.\s*",
        r"\n### Confidence Intervals\s*\n(?:- )?Not computed in artifacts\.\s*",
        r"\n### Effect Sizes\s*\n(?:- )?Not computed in artifacts\.\s*",
        r"\n### A/B Test Readout\s*\n(?:- )?Not computed in artifacts\.\s*",
        r"\n### Regression Findings\s*\n(?:- )?Not computed in artifacts\.\s*",
        r"\n### Statistical Limitations\s*\n(?:- )?Not computed in artifacts\.\s*",
    ]
    out = report
    for pat in patterns:
        out = re.sub(pat, "\n", out, flags=re.IGNORECASE)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + "\n"


def _build_deterministic_report(
    business_question: str,
    profile: Dict[str, Any],
    qa: Dict[str, Any],
    metrics_out: Dict[str, Any],
    evidence_payloads: Dict[str, Any],
) -> str:
    computed = (metrics_out or {}).get("computed", []) or []
    failed = (metrics_out or {}).get("failed", []) or []
    artifact_to_ev = {
        v.get("artifact_path"): ev_id for ev_id, v in evidence_payloads.items() if v.get("artifact_path")
    }

    entries = []
    for item in computed:
        artifact = item.get("artifact")
        ev_id = item.get("evidence_id") or artifact_to_ev.get(artifact)
        ev = evidence_payloads.get(ev_id or "", {})
        entries.append(
            {
                "task_id": item.get("task_id"),
                "artifact": str(artifact or ""),
                "ev_id": ev_id,
                "summary": ev.get("summary"),
                "payload": ev.get("payload"),
                "pointer_value": ev.get("pointer_value"),
            }
        )
    stats_entries = _extract_statistical_entries(entries)

    referenced_ids: list[str] = []
    question_lc = business_question.lower()
    def _has_kw(words: list[str]) -> bool:
        for w in words:
            w = w.lower().strip()
            if not w:
                continue
            if " " in w or "-" in w:
                if w in question_lc:
                    return True
            elif re.search(rf"\b{re.escape(w)}\b", question_lc):
                return True
        return False

    def _tag(ev_id: str | None) -> str:
        if not ev_id:
            return ""
        referenced_ids.append(ev_id)
        return f"[[EV:{ev_id}]]"

    # Build a generic focused comparison (works for country/customer/category or any grouped dimension).
    filter_entry = next(
        (
            e
            for e in entries
            if isinstance(e.get("payload"), dict)
            and isinstance(e["payload"].get("filter"), dict)
            and "value" in e["payload"]
            and e.get("ev_id")
        ),
        None,
    )

    grouped_candidates = [
        e
        for e in entries
        if isinstance(e.get("payload"), dict)
        and len(e["payload"]) >= 2
        and sum(isinstance(v, (int, float)) for v in e["payload"].values()) >= 2
        and e.get("ev_id")
    ]
    grouped_entry = None
    if grouped_candidates:
        q_words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", question_lc))
        preferred_dim_tokens = set()
        if _has_kw(["country", "region", "market"]):
            preferred_dim_tokens.update({"country", "region", "market"})
        if _has_kw(["product", "category", "sku"]):
            preferred_dim_tokens.update({"product", "category", "sku"})
        if _has_kw(["customer", "customers"]):
            preferred_dim_tokens.update({"customer"})
        preferred_dim_tokens.update({"india", "usa", "uk", "germany", "canada", "france", "japan", "china"})

        def _score_grouped(entry: Dict[str, Any]) -> int:
            score = 0
            payload = entry.get("payload")
            artifact = str(entry.get("artifact", "")).lower()
            summary = str(entry.get("summary", "")).lower()
            if isinstance(payload, dict):
                keys_lc = {str(k).lower() for k in payload.keys()}
                if preferred_dim_tokens & keys_lc:
                    score += 10
                if q_words & keys_lc:
                    score += 8
                if "india" in q_words and "india" in keys_lc:
                    score += 15
            if any(tok in artifact for tok in preferred_dim_tokens):
                score += 5
            if any(tok in summary for tok in preferred_dim_tokens):
                score += 5
            if "_sql_query" in artifact:
                score += 2
            return score

        grouped_candidates = sorted(grouped_candidates, key=_score_grouped, reverse=True)
        grouped_entry = grouped_candidates[0]
    group2_entry = next(
        (
            e
            for e in entries
            if isinstance(e.get("payload"), list)
            and e["payload"]
            and isinstance(e["payload"][0], dict)
            and "value" in e["payload"][0]
            and e.get("ev_id")
        ),
        None,
    )
    group_dist_entry = next(
        (
            e
            for e in entries
            if isinstance(e.get("payload"), dict)
            and "quantiles" in e["payload"]
            and "group_by" in e["payload"]
            and e.get("ev_id")
        ),
        None,
    )

    focus_dim = None
    focus_key = None
    focus_value = None
    if filter_entry:
        f = filter_entry["payload"].get("filter", {})
        if isinstance(f, dict) and len(f) == 1:
            focus_dim = next(iter(f.keys()))
            focus_key = f[focus_dim]
            try:
                focus_value = float(filter_entry["payload"].get("value"))
            except Exception:
                focus_value = None

    ranked_items: list[tuple[str, float]] = []
    if grouped_entry:
        for k, v in grouped_entry["payload"].items():
            if isinstance(v, (int, float)):
                ranked_items.append((str(k), float(v)))
        ranked_items = sorted(ranked_items, key=lambda x: x[1], reverse=True)

    focus_rank = None
    top_item = ranked_items[0] if ranked_items else None
    if focus_key is not None and ranked_items:
        for i, (k, v) in enumerate(ranked_items, start=1):
            if str(k) == str(focus_key):
                focus_rank = i
                if focus_value is None:
                    focus_value = v
                break

    if focus_key is None and ranked_items:
        for i, (k, v) in enumerate(ranked_items, start=1):
            if str(k).lower() in question_lc:
                focus_key = k
                focus_value = v
                focus_rank = i
                break

    exec_bullets = [f"- Business question: {business_question}"]
    if focus_key is not None and focus_value is not None and ranked_items:
        top_name, top_val = top_item
        exec_bullets.append(
            f"- {focus_key} value is {focus_value:,.2f} and rank is #{focus_rank or '?'} "
            f"{_tag(filter_entry.get('ev_id') if filter_entry else None)} {_tag(grouped_entry.get('ev_id') if grouped_entry else None)}".strip()
        )
        exec_bullets.append(
            f"- Top {focus_dim or 'segment'} is {top_name} at {top_val:,.2f}; gap is {top_val - focus_value:,.2f} "
            f"{_tag(grouped_entry.get('ev_id'))}".strip()
        )
    elif grouped_entry and ranked_items:
        preview = ", ".join([f"{k} ({v:,.2f})" for k, v in ranked_items[:5]])
        exec_bullets.append(f"- Main grouped results: {preview} {_tag(grouped_entry.get('ev_id'))}")
    else:
        exec_bullets.append("- Quantitative metrics were computed and are listed below.")

    answer_lines = []
    if focus_key is not None and focus_value is not None and ranked_items:
        answer_lines.append(
            f"Direct answer: {focus_key} = {focus_value:,.2f}, rank {focus_rank}/{len(ranked_items)} in the main comparison "
            f"{_tag(filter_entry.get('ev_id') if filter_entry else None)} {_tag(grouped_entry.get('ev_id') if grouped_entry else None)}."
        )
    elif ranked_items and _has_kw(["highest", "top", "most", "largest"]):
        top_name, top_val = ranked_items[0]
        if len(ranked_items) > 1:
            second_name, second_val = ranked_items[1]
            answer_lines.append(
                f"Direct answer: {top_name} is highest at {top_val:,.2f}, leading {second_name} by {top_val - second_val:,.2f} "
                f"{_tag(grouped_entry.get('ev_id'))}."
            )
        else:
            answer_lines.append(f"Direct answer: {top_name} is highest at {top_val:,.2f} {_tag(grouped_entry.get('ev_id'))}.")
    elif ranked_items and _has_kw(["lowest", "least", "smallest"]):
        low_name, low_val = ranked_items[-1]
        if len(ranked_items) > 1:
            prev_name, prev_val = ranked_items[-2]
            answer_lines.append(
                f"Direct answer: {low_name} is lowest at {low_val:,.2f}, trailing {prev_name} by {prev_val - low_val:,.2f} "
                f"{_tag(grouped_entry.get('ev_id'))}."
            )
        else:
            answer_lines.append(f"Direct answer: {low_name} is lowest at {low_val:,.2f} {_tag(grouped_entry.get('ev_id'))}.")
    else:
        answer_lines.append("Direct answer: see computed evidence list; no single filtered target was explicitly requested.")

    if group_dist_entry and isinstance(group_dist_entry.get("payload"), dict):
        qvals = group_dist_entry["payload"].get("quantiles", {})
        if isinstance(qvals, dict):
            p90 = qvals.get("0.9") or qvals.get("0.90")
            p95 = qvals.get("0.95")
            if isinstance(p90, (int, float)):
                txt = f"High-value threshold (P90) is {float(p90):,.2f}"
                if isinstance(p95, (int, float)):
                    txt += f"; P95 is {float(p95):,.2f}"
                answer_lines.append(txt + f" {_tag(group_dist_entry.get('ev_id'))}.")

    if group2_entry and isinstance(group2_entry.get("payload"), list):
        rows = [r for r in group2_entry["payload"] if isinstance(r, dict) and "value" in r]
        if rows:
            top_row = rows[0]
            dims = [k for k in top_row.keys() if k != "value"]
            if len(dims) >= 2:
                d1, d2 = dims[0], dims[1]
                answer_lines.append(
                    f"Top {d1}-{d2} combination is {top_row.get(d1)} / {top_row.get(d2)} "
                    f"with value {float(top_row.get('value')):,.2f} {_tag(group2_entry.get('ev_id'))}."
                )

    key_evidence = []
    if filter_entry:
        key_evidence.append(
            f"- Filter metric: {filter_entry['payload'].get('filter')} -> {filter_entry['payload'].get('value')} "
            f"{_tag(filter_entry.get('ev_id'))}"
        )
    if grouped_entry and ranked_items:
        preview = " > ".join([f"{k} ({v:,.2f})" for k, v in ranked_items[:5]])
        key_evidence.append(f"- Grouped metric ranking: {preview} {_tag(grouped_entry.get('ev_id'))}")
    if not key_evidence:
        key_evidence.append("- Not computed in artifacts.")

    output_lines = []
    for e in entries[:16]:
        ev_tag = _tag(e.get("ev_id"))
        payload = e.get("payload")
        if isinstance(payload, dict) and "filter" in payload and "value" in payload:
            output_lines.append(f"- {e.get('task_id')}: filter={payload.get('filter')} value={payload.get('value')} {ev_tag}")
        elif isinstance(payload, dict) and payload.get("analysis_type") in {"hypothesis_test", "ab_test", "regression"}:
            output_lines.append(
                f"- {e.get('task_id')}: statistical {payload.get('analysis_type')} via {payload.get('method')} "
                f"(decision={payload.get('decision')}, p-value={payload.get('p_value')}) {ev_tag}"
            )
        elif isinstance(payload, dict):
            top_items = list(payload.items())[:5]
            output_lines.append(f"- {e.get('task_id')}: top values={top_items} {ev_tag}")
        elif isinstance(payload, list):
            output_lines.append(f"- {e.get('task_id')}: rows={len(payload)} {ev_tag}")
        else:
            output_lines.append(f"- {e.get('task_id')}: computed {ev_tag}")

    artifact_names = [e.get("artifact", "") for e in entries]
    limitations = []
    if _has_kw(["customer", "high value", "high-value", "rfm"]):
        if not any("_groupby_customer_id_" in a for a in artifact_names):
            limitations.append("- Customer-level spend metrics were not computed.")
        if not any("_group_dist_customer_id_" in a for a in artifact_names):
            limitations.append("- Customer spend percentiles/quantiles were not computed.")
        if not any("_recency_customer_id_" in a for a in artifact_names):
            limitations.append("- Customer recency metrics were not computed.")
    if _has_kw(["why", "driver", "cause", "reason"]):
        if not any("_groupby2_" in a for a in artifact_names):
            limitations.append("- Cross-segment mix analysis (2D groupby) is missing.")
        if not any("_corr_" in a for a in artifact_names):
            limitations.append("- Correlation-based driver checks are missing.")
    if _has_kw(["trend", "time", "monthly", "weekly", "over time", "season"]) and not any(
        "_timeseries_" in a for a in artifact_names
    ):
        limitations.append("- Time-series aggregation was not computed.")
    if not limitations:
        limitations.append("- Core computations needed for this question were generated.")
    statistical_subsections = _format_statistical_subsections(stats_entries, _tag)
    if stats_entries and not any("statistical" in lim.lower() for lim in limitations):
        for entry in stats_entries:
            payload = entry.get("payload") or {}
            for lim in payload.get("limitations", []) or []:
                limitations.append(f"- {lim}")
        deduped: list[str] = []
        seen_lim: set[str] = set()
        for lim in limitations:
            if lim not in seen_lim:
                seen_lim.add(lim)
                deduped.append(lim)
        limitations = deduped[:8]

    next_steps: list[str] = []
    if any("were not computed" in lim for lim in limitations):
        for lim in limitations:
            if "Customer-level spend metrics" in lim:
                next_steps.append("- Compute customer-level monetary totals and order frequency across all customers.")
            elif "percentiles/quantiles" in lim:
                next_steps.append("- Compute customer spend percentiles (P90/P95/P99) to define high-value segments.")
            elif "recency metrics" in lim:
                next_steps.append("- Compute recency by customer using latest order date.")
            elif "Cross-segment mix analysis" in lim:
                next_steps.append("- Compute 2D mix table for primary segments to identify composition effects.")
            elif "Correlation-based driver checks" in lim:
                next_steps.append("- Compute correlations between key numeric drivers and the target metric.")
            elif "Time-series aggregation" in lim:
                next_steps.append("- Add time-series aggregation to evaluate trend stability and seasonality.")
    else:
        if _has_kw(["count", "how many", "number of", "orders"]):
            next_steps.append("- Check month-over-month order counts by top segments to confirm lead stability.")
            next_steps.append("- Compare unique customers and repeat-rate by segment to explain volume differences.")
        elif _has_kw(["customer", "high-value", "high value", "rfm"]):
            next_steps.append("- Build actionable tiers (e.g., P90/P95) and profile category affinity within each tier.")
            next_steps.append("- Add RFM score bands to prioritize retention and upsell actions.")
        elif _has_kw(["why", "driver", "cause", "reason"]):
            next_steps.append("- Quantify each candidate driver contribution using segment-level decomposition.")
            next_steps.append("- Validate whether the gap persists across time windows and major categories.")
        else:
            next_steps.append("- Drill into top and bottom segments to validate stability across subgroups.")
            next_steps.append("- Convert computed metrics into decision thresholds tied to this question.")
    if not next_steps:
        next_steps.append("- No additional computation is required; proceed with decision-making from current evidence.")

    cols = profile.get("columns", []) or []
    rows = profile.get("n_rows", "—")
    ncols = profile.get("n_cols", "—")
    rows_ev = next(
        (
            ev_id
            for ev_id, ev in evidence_payloads.items()
            if ev.get("artifact_path") == "data_profile.json" and ev.get("pointer") == "n_rows"
        ),
        None,
    )
    cols_ev = next(
        (
            ev_id
            for ev_id, ev in evidence_payloads.items()
            if ev.get("artifact_path") == "data_profile.json" and ev.get("pointer") == "n_cols"
        ),
        None,
    )
    missing_ev = next(
        (
            ev_id
            for ev_id, ev in evidence_payloads.items()
            if ev.get("artifact_path") == "quality_report.json" and ev.get("pointer") == "missingness"
        ),
        None,
    )
    dup_ev = next(
        (
            ev_id
            for ev_id, ev in evidence_payloads.items()
            if ev.get("artifact_path") == "quality_report.json" and ev.get("pointer") == "duplicate_rate"
        ),
        None,
    )
    miss = qa.get("missingness")
    miss_txt = "Not computed in artifacts."
    if isinstance(miss, dict):
        miss_txt = "0% across all columns" if all(float(v) == 0.0 for v in miss.values()) else "Non-zero missingness detected"
    dup = qa.get("duplicate_rate", "Not computed in artifacts.")

    failed_txt = "none" if not failed else "; ".join([f"{f.get('task_id')}: {f.get('reason')}" for f in failed[:5]])

    return f"""# Data Analysis Report

## 1) Executive Summary
{chr(10).join(exec_bullets)}

## 2) Question Answer (Evidence)
{" ".join(answer_lines)}

Key evidence:
{chr(10).join(key_evidence)}

## 3) Dataset Overview
- Rows: {rows} {_tag(rows_ev)}
- Columns: {ncols} {_tag(cols_ev)}
- Key fields: {", ".join(cols[:12]) if cols else "Not computed in artifacts."}

## 4) Data Quality Findings
- Missingness: {miss_txt} {_tag(missing_ev)}
- Duplicate rate: {dup} {_tag(dup_ev)}

## 5) Analysis Outputs
{chr(10).join(output_lines) if output_lines else "- Not computed in artifacts."}
- Failed tasks: {failed_txt}
{chr(10) + chr(10) + statistical_subsections if statistical_subsections else ""}

## 6) Limitations
{chr(10).join(limitations)}

## 7) Next Steps
{chr(10).join(next_steps[:5])}
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
        source = ctx.get("source", {"type": "csv"})
        sql_schema = ctx.get("sql_schema") if isinstance(ctx.get("sql_schema"), dict) else None

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

        # Include actual artifact payloads so the report can state concrete answers.
        evidence_payloads: Dict[str, Any] = {}
        allow_raw_rows = bool(getattr(cfg, "security", None) and cfg.security.allow_raw_rows_to_llm)
        max_rows_to_llm = int(getattr(getattr(cfg, "security", None), "max_rows_to_llm", 25))
        for ev_id, ev in evidence_store.all().items():
            payload = None
            if ev.artifact_path and ev.artifact_path.endswith(".json"):
                payload = _safe_read_json(store.path(ev.artifact_path))
            safe_payload = redact_payload_for_llm(
                payload,
                allow_raw_rows=allow_raw_rows,
                max_rows=max_rows_to_llm,
            ) if payload is not None else None
            pointer_value = _pointer_value(payload, ev.pointer)
            safe_pointer_value = redact_payload_for_llm(
                pointer_value,
                allow_raw_rows=allow_raw_rows,
                max_rows=max_rows_to_llm,
            )
            evidence_payloads[ev_id] = {
                "artifact_path": ev.artifact_path,
                "pointer": ev.pointer,
                "summary": ev.summary,
                "pointer_value": safe_pointer_value,
                "payload": _compact_json(safe_payload) if safe_payload is not None else None,
            }

        computed_facts: list[str] = []
        for item in (metrics_out or {}).get("computed", []):
            artifact = item.get("artifact")
            if not artifact:
                continue
            payload = _safe_read_json(store.path(str(artifact)))
            if payload is None:
                continue
            payload = redact_payload_for_llm(
                payload,
                allow_raw_rows=allow_raw_rows,
                max_rows=max_rows_to_llm,
            )
            tid = str(item.get("task_id", "T?"))
            if isinstance(payload, dict) and "filter" in payload and "value" in payload:
                computed_facts.append(
                    f"{tid}: filter={payload.get('filter')} value={payload.get('value')}"
                )
            elif isinstance(payload, dict):
                top_items = list(payload.items())[:5]
                computed_facts.append(f"{tid}: top_values={top_items}")

        # ✅ Include everything the reporter needs to answer the question
        artifacts_context = {
            "business_question": business_question,
            "source": source,
            "analysis_plan": plan,
            "data_profile": profile,
            "sql_schema": sql_schema,
            "quality_report": qa,
            "eda_summary": eda,
            "planner_output": planner_out,        # includes tasks list
            "metrics_outputs": metrics_out,       # includes computed/failed/skipped
            "evidence_map": evidence_map,
            "evidence_payloads": evidence_payloads,
            "computed_facts": computed_facts,
        }

        # Inject the business question into the system prompt (safer than relying on user msg)
        system_prompt = SYSTEM_PROMPT.format(business_question=business_question)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Business Question:\n{business_question}\n\n"
                    "Generate the report now. Answer the business question first, "
                    "then support with evidence tags. Use evidence_payloads and computed_facts for numeric values. "
                    "If metrics_outputs.computed is non-empty, include at least one concrete numeric answer in sections 1 and 2."
                ),
            },
            {"role": "user", "content": "Artifacts Context (JSON):\n" + json.dumps(artifacts_context, indent=2)},
        ]

        logger.info("[OpenRouter] Calling LLM for report generation...")
        report_md = ""
        used_fallback = False
        try:
            client = OpenRouterClient(timeout_s=cfg.llm.timeout_s)
            report_md = client.chat(
                model=cfg.llm.model,
                messages=messages,
                temperature=cfg.llm.temperature,
                max_tokens=cfg.llm.max_tokens,
            )
        except Exception as e:
            logger.warning(f"[Reporting] OpenRouter call failed: {e}. Falling back to deterministic report.")
        report_md = _normalize_evidence_tags(report_md or "")
        if not any(
            isinstance(ev.get("payload"), dict) and ev.get("payload", {}).get("analysis_type") in {"hypothesis_test", "ab_test", "regression"}
            for ev in evidence_payloads.values()
        ):
            report_md = _strip_empty_statistical_subsections(report_md)
        if _report_needs_fallback(report_md, metrics_out or {}, evidence_payloads):
            logger.warning("[Reporting] LLM report incomplete for available metrics. Using deterministic fallback.")
            used_fallback = True
            report_md = _build_deterministic_report(
                business_question=business_question,
                profile=profile or {},
                qa=qa or {},
                metrics_out=metrics_out or {},
                evidence_payloads=evidence_payloads,
            )
            report_md = _normalize_evidence_tags(report_md)
        if not any(
            isinstance(ev.get("payload"), dict) and ev.get("payload", {}).get("analysis_type") in {"hypothesis_test", "ab_test", "regression"}
            for ev in evidence_payloads.values()
        ):
            report_md = _strip_empty_statistical_subsections(report_md)
        report_md = _format_evidence_citations(report_md, evidence_store.all())

        consistency_issues = _collect_report_consistency_issues(report_md, set(evidence_store.all().keys()))
        if consistency_issues:
            logger.warning("[Reporting] Consistency issues found. Falling back to deterministic report.")
            used_fallback = True
            report_md = _build_deterministic_report(
                business_question=business_question,
                profile=profile or {},
                qa=qa or {},
                metrics_out=metrics_out or {},
                evidence_payloads=evidence_payloads,
            )
            report_md = _normalize_evidence_tags(report_md)
            if not any(
                isinstance(ev.get("payload"), dict) and ev.get("payload", {}).get("analysis_type") in {"hypothesis_test", "ab_test", "regression"}
                for ev in evidence_payloads.values()
            ):
                report_md = _strip_empty_statistical_subsections(report_md)
            report_md = _format_evidence_citations(report_md, evidence_store.all())
            consistency_issues = _collect_report_consistency_issues(report_md, set(evidence_store.all().keys()))

        unresolved_count = len([x for x in RAW_EV_TOKEN_PATTERN.findall(report_md) if not re.match(r"\[\[(?:EV:)?EV-[a-f0-9]{10}\]\]", x)])
        unsupported_numeric_claim_lines = len(
            [x for x in consistency_issues if x.startswith("Unsupported numeric claim")]
        )
        contradiction_count = len([x for x in consistency_issues if x.startswith("Potential contradiction")])
        required_sections = [
            "1) Executive Summary",
            "2) Question Answer (Evidence)",
            "3) Dataset Overview",
            "4) Data Quality Findings",
            "5) Analysis Outputs",
            "6) Limitations",
            "7) Next Steps",
        ]
        section_ok = all(_report_section(report_md, sec).strip() for sec in required_sections)
        report_metadata = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "analysis_type": (
                (planner_out or {}).get("analysis_type")
                or (plan or {}).get("analysis_type")
            ),
            "report_path": "final_report.md",
            "used_fallback": bool(used_fallback),
            "unresolved_ev_placeholders": int(unresolved_count),
            "unsupported_numeric_claim_lines": int(unsupported_numeric_claim_lines),
            "section_completeness_ok": bool(section_ok),
            "contradiction_count": int(contradiction_count),
            "consistency_issues": [str(x) for x in consistency_issues],
        }
        report_metadata = validate_report_metadata_contract(report_metadata).model_dump()
        logger.info("[OpenRouter] Report generated.")

        store.write_text("final_report.md", report_md)
        store.write_json("report_metadata.json", report_metadata)
        memory = ctx.get("memory")
        if memory is not None:
            memory.set("result.reporting_metadata", report_metadata)
        return report_md
