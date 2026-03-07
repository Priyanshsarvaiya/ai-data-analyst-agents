from __future__ import annotations

import pandas as pd

from ai_data_analyst_agents.agents.wrangling import WranglingAgent
from ai_data_analyst_agents.core.artifacts import ArtifactStore
from ai_data_analyst_agents.core.logging import setup_logging
from ai_data_analyst_agents.core.memory import SharedMemory


def test_wrangling_deduplicates_and_writes_artifacts(tmp_path, sample_df_with_duplicate: pd.DataFrame) -> None:
    store = ArtifactStore.create(tmp_path / "artifacts")
    logger = setup_logging("INFO", store.path("logs.txt"))
    memory = SharedMemory()
    ctx = {"store": store, "logger": logger, "df": sample_df_with_duplicate, "memory": memory}

    out = WranglingAgent().run(ctx)

    assert out["rows_before"] == len(sample_df_with_duplicate)
    assert out["rows_after"] == len(sample_df_with_duplicate.drop_duplicates())
    assert store.path("cleaned.csv").exists()
    assert store.path("feature_log.json").exists()

    cleaned = pd.read_csv(store.path("cleaned.csv"))
    assert len(cleaned) == out["rows_after"]
    assert memory.get("df.cleaned") is not None


def test_shared_memory_is_isolated() -> None:
    m1 = SharedMemory()
    m2 = SharedMemory()

    m1.set("x", 1)
    m1.log({"message": "a"})

    assert m1.get("x") == 1
    assert m2.get("x") is None
    assert len(m1.messages) == 1
    assert len(m2.messages) == 0
