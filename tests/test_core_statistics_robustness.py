from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ai_data_analyst_agents.core.artifacts import ArtifactStore
from ai_data_analyst_agents.core.evidence import EvidenceStore
from ai_data_analyst_agents.core.security import validate_read_only_sql
from ai_data_analyst_agents.core.settings import load_app_cfg
from ai_data_analyst_agents.statistics.hypothesis_tests import paired_t_test
from ai_data_analyst_agents.statistics.models import HypothesisTestRequest
from ai_data_analyst_agents.statistics.selector import run_hypothesis_test, select_hypothesis_method


def test_artifact_store_blocks_escape_and_serializes_nonfinite_as_null(tmp_path) -> None:
    store = ArtifactStore.create(tmp_path / "artifacts")

    with pytest.raises(ValueError, match="escapes"):
        store.write_text("../outside.txt", "unsafe")

    path = store.write_json("safe.json", {"nan": float("nan"), "inf": np.float64("inf")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"nan": None, "inf": None}

    with pytest.raises(ValueError, match="safe run-relative"):
        EvidenceStore().add(kind="json", artifact_path="", summary="missing path")


def test_sql_guard_ignores_keywords_and_semicolons_inside_literals() -> None:
    query = "SELECT 'update; still text' AS note, \"delete\" FROM events;"
    assert validate_read_only_sql(query) == query[:-1]


def test_llm_output_budgets_can_be_overridden_independently(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_PLANNER_MAX_TOKENS", "3072")
    monkeypatch.setenv("LLM_REPORT_MAX_TOKENS", "6144")
    cfg = load_app_cfg(tmp_path / "missing-settings.yaml")
    assert cfg.llm.planner_max_tokens == 3072
    assert cfg.llm.report_max_tokens == 6144


def test_sparse_binary_groups_select_fisher_exact() -> None:
    df = pd.DataFrame(
        {
            "group": ["A"] * 8 + ["B"] * 8,
            "converted": [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        }
    )
    request = HypothesisTestRequest(
        group_col="group",
        metric="converted",
        group_a="A",
        group_b="B",
        metric_type="binary",
    )
    selection, result = run_hypothesis_test(df, request, analysis_id="sparse")
    assert selection.method == "fisher_exact_test"
    assert result.method == "fisher_exact_test"
    assert 0.0 <= (result.p_value or 0.0) <= 1.0


def test_binary_success_value_zero_is_counted_without_treating_true_as_success() -> None:
    df = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "flag": ([0] * 15 + [1] * 5) + ([0] * 8 + [1] * 12),
        }
    )
    _, result = run_hypothesis_test(
        df,
        HypothesisTestRequest(
            group_col="group",
            metric="flag",
            group_a="A",
            group_b="B",
            metric_type="binary",
            success_value=0,
        ),
        analysis_id="zero-success",
    )
    assert result.metrics["rate_A"] == pytest.approx(0.75)
    assert result.metrics["rate_B"] == pytest.approx(0.40)


def test_paired_test_preserves_original_row_alignment_with_missing_values() -> None:
    result = paired_t_test(
        analysis_id="paired",
        paired=pd.DataFrame({"before": [1.0, None, 4.0], "after": [0.0, 100.0, 2.0]}),
        left_col="before",
        right_col="after",
        label_left="before",
        label_right="after",
        metric="score",
        alpha=0.05,
    )
    assert result.sample_sizes == {"pairs": 2}
    assert result.metrics["mean_difference"] == pytest.approx(1.5)
    assert result.confidence_intervals[0].point_estimate == pytest.approx(1.5)
    assert result.effect_sizes[0].name == "cohens_dz"


def test_explicit_continuous_metric_does_not_auto_select_binary_test() -> None:
    df = pd.DataFrame({"group": ["A"] * 20 + ["B"] * 20, "value": [0, 1] * 20})
    selection = select_hypothesis_method(
        df,
        HypothesisTestRequest(
            group_col="group",
            metric="value",
            group_a="A",
            group_b="B",
            metric_type="continuous",
        ),
    )
    assert selection.method == "welch_t_test"


def test_explicit_continuous_metric_rejects_mostly_non_numeric_values() -> None:
    df = pd.DataFrame({"group": ["A"] * 5 + ["B"] * 5, "value": ["bad"] * 10})
    with pytest.raises(ValueError, match="declared continuous"):
        select_hypothesis_method(
            df,
            HypothesisTestRequest(
                group_col="group",
                metric="value",
                group_a="A",
                group_b="B",
                metric_type="continuous",
            ),
        )
