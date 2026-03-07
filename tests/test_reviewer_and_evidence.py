from __future__ import annotations

from ai_data_analyst_agents.agents.reviewer import ReviewerAgent
from ai_data_analyst_agents.core.artifacts import ArtifactStore
from ai_data_analyst_agents.core.evidence import EvidenceStore
from ai_data_analyst_agents.core.logging import setup_logging


def _base_ctx(tmp_path):
    store = ArtifactStore.create(tmp_path / "artifacts")
    logger = setup_logging("INFO", store.path("logs.txt"))
    evidence = EvidenceStore()
    return {"store": store, "logger": logger, "evidence": evidence}


def test_reviewer_fails_on_missing_evidence_refs(tmp_path) -> None:
    ctx = _base_ctx(tmp_path)
    ctx["store"].write_text(
        "final_report.md",
        "# Data Analysis Report\n\n## 1) Executive Summary\n- Revenue is 100 [[EV:EV-deadbeef00]]\n",
    )

    out = ReviewerAgent().run(ctx)
    assert out["status"] == "fail"
    assert "EV-deadbeef00" in out["missing_refs"]


def test_reviewer_fails_on_numeric_claim_without_ev_tag(tmp_path) -> None:
    ctx = _base_ctx(tmp_path)
    ev = ctx["evidence"].add(kind="metric", artifact_path="x.json", pointer="value", summary="x")
    ctx["store"].write_text(
        "final_report.md",
        (
            "# Data Analysis Report\n\n"
            "## 1) Executive Summary\n- Revenue is 100\n"
            "## 2) Question Answer (Evidence)\n- Supported [[EV:%s]]\n"
            "## 5) Analysis Outputs\n- Value is 50 [[EV:%s]]\n"
        )
        % (ev.id, ev.id),
    )

    out = ReviewerAgent().run(ctx)
    assert out["status"] == "fail"
    assert any("Numeric claims without evidence tag" in n for n in out["notes"])


def test_reviewer_passes_when_claims_have_valid_refs(tmp_path) -> None:
    ctx = _base_ctx(tmp_path)
    ev = ctx["evidence"].add(kind="metric", artifact_path="x.json", pointer="value", summary="x")
    ctx["store"].write_text(
        "final_report.md",
        (
            "# Data Analysis Report\n\n"
            "## 1) Executive Summary\n- Revenue is 100 [[EV:%s]]\n"
            "## 2) Question Answer (Evidence)\n- Supported [[EV:%s]]\n"
            "## 5) Analysis Outputs\n- Value is 50 [[EV:%s]]\n"
        )
        % (ev.id, ev.id, ev.id),
    )

    out = ReviewerAgent().run(ctx)
    assert out["status"] == "pass"
    assert out["missing_refs"] == []


def test_reviewer_passes_with_numeric_citations_and_reference_table(tmp_path) -> None:
    ctx = _base_ctx(tmp_path)
    ev = ctx["evidence"].add(kind="metric", artifact_path="x.json", pointer="value", summary="x")
    ctx["store"].write_text(
        "final_report.md",
        (
            "# Data Analysis Report\n\n"
            "## 1) Executive Summary\n- Revenue is 100 [1]\n"
            "## 2) Question Answer (Evidence)\n- Supported [1]\n"
            "## 5) Analysis Outputs\n- Value is 50 [1]\n"
            "## 9) Evidence References\n"
            "| Ref | Evidence ID | Artifact | Pointer | Summary |\n"
            "|---|---|---|---|---|\n"
            f"| [1] | {ev.id} | x.json | value | x |\n"
        ),
    )

    out = ReviewerAgent().run(ctx)
    assert out["status"] == "pass"
    assert out["missing_refs"] == []
