from __future__ import annotations

import re

from ai_data_analyst_agents.agents.reporting import (
    _build_deterministic_report,
    _compact_json,
    _format_evidence_citations,
    _normalize_evidence_tags,
    _pointer_value,
    _report_needs_fallback,
)
from ai_data_analyst_agents.core.evidence import EvidenceRef


def test_normalize_evidence_tags_variants() -> None:
    text = "A [[EV-1234567890]] and B [[EV-abcdef1234]] and C [[EV:EV-0000000001]]"
    out = _normalize_evidence_tags(text)
    assert "[[EV:EV-1234567890]]" in out
    assert "[[EV:EV-abcdef1234]]" in out
    assert "[[EV:EV-0000000001]]" in out


def test_pointer_value_direct_nested_and_missing() -> None:
    payload = {"value": 10, "a": {"b": {"c": 20}}}
    assert _pointer_value(payload, "value") == 10
    assert _pointer_value(payload, "a.b.c") == 20
    assert _pointer_value(payload, "a.b.x") is None
    assert _pointer_value(None, "x") is None


def test_compact_json_truncates_dict_and_list() -> None:
    payload = {"k1": 1, "k2": 2, "k3": 3}
    compact = _compact_json(payload, max_items=2)
    assert compact["k1"] == 1
    assert compact["k2"] == 2
    assert "__truncated__" in compact

    arr = [1, 2, 3, 4]
    compact_arr = _compact_json(arr, max_items=2)
    assert compact_arr[:2] == [1, 2]
    assert isinstance(compact_arr[-1], str) and "truncated" in compact_arr[-1]


def test_report_needs_fallback_when_section_two_has_no_numeric_answer() -> None:
    report = (
        "# Data Analysis Report\n\n"
        "## 1) Executive Summary\n- computed\n\n"
        "## 2) Question Answer (Evidence)\nNo numeric details here.\n"
    )
    metrics_out = {"computed": [{"task_id": "T1"}]}
    evidence_payloads = {
        "EV-1111111111": {"pointer_value": 123.0, "payload": {"x": 1}}
    }
    assert _report_needs_fallback(report, metrics_out, evidence_payloads) is True


def test_build_deterministic_report_uses_grouped_entity_from_question() -> None:
    question = "Why did India have less revenue than other countries?"
    profile = {"n_rows": 100, "n_cols": 5, "columns": ["country", "revenue"]}
    qa = {"missingness": {"country": 0.0, "revenue": 0.0}, "duplicate_rate": 0.0}
    metrics_out = {
        "computed": [{"task_id": "T1", "artifact": "country_rev.json", "evidence_id": "EV-aaaaaaaaaa"}],
        "failed": [],
    }
    evidence_payloads = {
        "EV-aaaaaaaaaa": {
            "artifact_path": "country_rev.json",
            "pointer": None,
            "summary": "sum(revenue) by country",
            "pointer_value": None,
            "payload": {"Germany": 200.0, "India": 150.0, "USA": 180.0},
        },
        "EV-rows000001": {
            "artifact_path": "data_profile.json",
            "pointer": "n_rows",
            "summary": "rows",
            "payload": {"n_rows": 100},
            "pointer_value": 100,
        },
        "EV-cols000001": {
            "artifact_path": "data_profile.json",
            "pointer": "n_cols",
            "summary": "cols",
            "payload": {"n_cols": 5},
            "pointer_value": 5,
        },
        "EV-miss000001": {
            "artifact_path": "quality_report.json",
            "pointer": "missingness",
            "summary": "missingness",
            "payload": {"missingness": {"country": 0.0}},
            "pointer_value": {"country": 0.0},
        },
        "EV-dup0000001": {
            "artifact_path": "quality_report.json",
            "pointer": "duplicate_rate",
            "summary": "duplicate",
            "payload": {"duplicate_rate": 0.0},
            "pointer_value": 0.0,
        },
    }

    report = _build_deterministic_report(
        business_question=question,
        profile=profile,
        qa=qa,
        metrics_out=metrics_out,
        evidence_payloads=evidence_payloads,
    )
    assert re.search(r"India.*rank", report, re.IGNORECASE)
    assert "## 1) Executive Summary" in report
    assert "## 8) Artifacts Index" in report
    assert "[[EV:EV-aaaaaaaaaa]]" in report


def test_format_evidence_citations_replaces_inline_tags_with_numeric_refs() -> None:
    report = (
        "# Data Analysis Report\n\n"
        "## 1) Executive Summary\n- Revenue is 100 [[EV:EV-aaaaaaaaaa]] and 50 [[EV:EV-bbbbbbbbbb]]\n"
    )
    refs = {
        "EV-aaaaaaaaaa": EvidenceRef(
            id="EV-aaaaaaaaaa",
            kind="metric",
            artifact_path="x.json",
            pointer="value",
            summary="x",
        ),
        "EV-bbbbbbbbbb": EvidenceRef(
            id="EV-bbbbbbbbbb",
            kind="metric",
            artifact_path="y.json",
            pointer=None,
            summary="y",
        ),
    }
    out = _format_evidence_citations(report, refs)
    assert "[[EV:" not in out
    assert "[1]" in out and "[2]" in out
    assert "## 9) Evidence References" in out
    assert "| [1] | EV-aaaaaaaaaa | x.json | value | x |" in out
