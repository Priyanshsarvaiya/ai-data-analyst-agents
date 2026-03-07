from __future__ import annotations

from ai_data_analyst_agents.agents.metrics import MetricsAgent
from ai_data_analyst_agents.core.artifacts import ArtifactStore
from ai_data_analyst_agents.core.evidence import EvidenceStore
from ai_data_analyst_agents.core.logging import setup_logging
from ai_data_analyst_agents.core.memory import SharedMemory
from ai_data_analyst_agents.core.settings import load_app_cfg
from ai_data_analyst_agents.core.sql_source import SQLDataSource


def test_metrics_agent_executes_phase3_task_types(sqlite_star_db, tmp_path) -> None:
    source = SQLDataSource(db_url=f"sqlite:///{sqlite_star_db}")
    schema = source.inspect_schema(include_row_counts=True)
    df = source.load_table("orders")

    cfg = load_app_cfg()
    store = ArtifactStore.create(tmp_path / "artifacts")
    logger = setup_logging("INFO", store.path("logs.txt"))
    memory = SharedMemory()
    evidence = EvidenceStore()

    tasks = [
        {"id": "T1", "type": "groupby_agg", "params": {"group_by": "product_category", "metric": "revenue", "agg": "sum"}},
        {"id": "T2", "type": "sql_join_profile", "params": {"fact_table": "orders", "dimension_table": "customers"}},
        {
            "id": "T3",
            "type": "sql_query",
            "params": {
                "query": (
                    "SELECT c.country, SUM(o.revenue) AS value "
                    "FROM orders o LEFT JOIN customers c ON o.customer_id = c.customer_id "
                    "GROUP BY c.country ORDER BY value DESC"
                ),
                "output": "mapping",
                "limit": 1000,
            },
        },
        {"id": "T4", "type": "kpi_template_apply", "params": {"domain": "ecommerce"}},
        {"id": "T5", "type": "segment_analysis", "params": {"segment_by": "product_category", "metric": "revenue"}},
        {"id": "T6", "type": "cohort_analysis", "params": {"entity_col": "customer_id", "date_col": "order_date", "freq": "M"}},
        {"id": "T7", "type": "metric_definition", "params": {"name": "rev_per_unit", "expression": "revenue/quantity"}},
    ]
    memory.set("result.planner", {"tasks": tasks})

    ctx = {
        "cfg": cfg,
        "store": store,
        "logger": logger,
        "df": df,
        "memory": memory,
        "evidence": evidence,
        "sql_source": source,
        "sql_schema": schema,
        "business_question": "Why is revenue different by country?",
    }

    out = MetricsAgent().run(ctx)
    assert out["failed"] == []
    assert len(out["computed"]) >= 7

    artifacts = [x["artifact"] for x in out["computed"]]
    assert any(a.endswith("_sql_query.json") for a in artifacts)
    assert any(a.endswith("_sql_join_profile.json") for a in artifacts)
    assert any("_kpi_template_ecommerce.json" in a for a in artifacts)

    for artifact in artifacts:
        assert store.path(artifact).exists()

    assert len(evidence.all()) >= len(out["computed"])
