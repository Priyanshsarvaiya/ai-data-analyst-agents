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
            "## 6) Limitations\n- Limited to provided artifacts.\n"
        )
        % (ev.id, ev.id),
    )

    out = ReviewerAgent().run(ctx)
    assert out["status"] == "fail"
    assert any("Numeric claims without evidence support" in n for n in out["notes"])


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
            "## 6) Limitations\n- Limited to provided artifacts.\n"
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
            "## 6) Limitations\n- Limited to provided artifacts.\n"
            "## 8) Evidence References\n"
            "| Ref | Evidence ID | Artifact | Pointer | Summary |\n"
            "|---|---|---|---|---|\n"
            f"| [1] | {ev.id} | x.json | value | x |\n"
        ),
    )

    out = ReviewerAgent().run(ctx)
    assert out["status"] == "pass"
    assert out["missing_refs"] == []


def test_reviewer_fails_on_statistical_significance_without_ci_and_effects(tmp_path) -> None:
    ctx = _base_ctx(tmp_path)
    ev = ctx["evidence"].add(
        kind="json",
        artifact_path="statistics/T1_two_proportion_z_test/summary.json",
        pointer=None,
        summary="ab test",
    )
    ctx["store"].write_text(
        "final_report.md",
        (
            "# Data Analysis Report\n\n"
            "## 1) Executive Summary\n- The treatment produced a statistically significant improvement [1]\n"
            "## 2) Question Answer (Evidence)\n- Supported [1]\n"
            "## 5) Analysis Outputs\n- Statistical result [1]\n"
            "## 6) Limitations\n- Minimal.\n"
            "## 8) Evidence References\n"
            "| Ref | Evidence ID | Artifact | Pointer | Summary |\n"
            "|---|---|---|---|---|\n"
            f"| [1] | {ev.id} | statistics/T1_two_proportion_z_test/summary.json | null | ab test |\n"
        ),
    )

    out = ReviewerAgent().run(ctx)
    assert out["status"] == "fail"
    assert any("p-value, confidence intervals, and effect sizes" in note for note in out["notes"])


def test_reviewer_blocks_causal_language_for_statistical_summary(tmp_path) -> None:
    ctx = _base_ctx(tmp_path)
    ev = ctx["evidence"].add(
        kind="json",
        artifact_path="statistics/T3_ols/summary.json",
        pointer=None,
        summary="regression",
    )
    ctx["store"].write_text(
        "final_report.md",
        (
            "# Data Analysis Report\n\n"
            "## 1) Executive Summary\n- Revenue was caused by marketing spend [1]\n"
            "## 2) Question Answer (Evidence)\n- Supported [1]\n"
            "## 5) Analysis Outputs\n"
            "### Confidence Intervals\n- coefficient:marketing_spend [1]\n"
            "### Effect Sizes\n- r_squared [1]\n"
            "### Statistical Limitations\n- Observational data only [1]\n"
            "## 6) Limitations\n- Observational data only.\n"
            "## 8) Evidence References\n"
            "| Ref | Evidence ID | Artifact | Pointer | Summary |\n"
            "|---|---|---|---|---|\n"
            f"| [1] | {ev.id} | statistics/T3_ols/summary.json | null | regression |\n"
        ),
    )

    out = ReviewerAgent().run(ctx)
    assert out["status"] == "fail"
    assert any("Causal wording is blocked" in note for note in out["notes"])
