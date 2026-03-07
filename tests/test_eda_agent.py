from __future__ import annotations

import pandas as pd

from ai_data_analyst_agents.agents.eda import EDAAgent
from ai_data_analyst_agents.core.artifacts import ArtifactStore
from ai_data_analyst_agents.core.logging import setup_logging
from ai_data_analyst_agents.core.memory import SharedMemory
from ai_data_analyst_agents.core.settings import load_app_cfg


def test_eda_generates_question_aware_charts_from_metrics(tmp_path, sample_df: pd.DataFrame) -> None:
    cfg = load_app_cfg()
    store = ArtifactStore.create(tmp_path / "artifacts")
    logger = setup_logging("INFO", store.path("logs.txt"))
    memory = SharedMemory()

    store.write_json("T1_group.json", {"India": 100.0, "Germany": 200.0, "USA": 150.0})
    store.write_json(
        "T2_mix.json",
        [
            {"country": "India", "product_category": "Books", "value": 10.0},
            {"country": "Germany", "product_category": "Books", "value": 20.0},
        ],
    )
    memory.set(
        "result.metrics",
        {"computed": [{"task_id": "T1", "artifact": "T1_group.json"}, {"task_id": "T2", "artifact": "T2_mix.json"}]},
    )

    ctx = {"cfg": cfg, "store": store, "logger": logger, "df": sample_df, "memory": memory}
    out = EDAAgent().run(ctx)

    assert out["question_aware_charts"], "Expected at least one question-aware chart."
    for chart in out["question_aware_charts"]:
        assert (store.run_dir / "charts" / chart).exists()


def test_eda_handles_empty_dataset_without_crashing(tmp_path) -> None:
    cfg = load_app_cfg()
    store = ArtifactStore.create(tmp_path / "artifacts")
    logger = setup_logging("INFO", store.path("logs.txt"))
    memory = SharedMemory()
    memory.set("result.metrics", {"computed": []})

    empty_df = pd.DataFrame(columns=["a", "b"])
    ctx = {"cfg": cfg, "store": store, "logger": logger, "df": empty_df, "memory": memory}
    out = EDAAgent().run(ctx)

    assert out["charts"] == []
    assert out["question_aware_charts"] == []
