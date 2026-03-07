from __future__ import annotations

import argparse
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from ai_data_analyst_agents.agents.eda import EDAAgent
from ai_data_analyst_agents.agents.intake import IntakeAgent
from ai_data_analyst_agents.agents.metrics import MetricsAgent
from ai_data_analyst_agents.agents.planner import PlannerAgent
from ai_data_analyst_agents.agents.profiling import ProfilingAgent
from ai_data_analyst_agents.agents.quality import QualityAgent
from ai_data_analyst_agents.agents.reporting import ReportingAgent
from ai_data_analyst_agents.agents.reviewer import ReviewerAgent
from ai_data_analyst_agents.agents.wrangling import WranglingAgent
from ai_data_analyst_agents.core.artifacts import ArtifactStore
from ai_data_analyst_agents.core.evidence import EvidenceStore
from ai_data_analyst_agents.core.logging import setup_logging
from ai_data_analyst_agents.core.memory import SharedMemory
from ai_data_analyst_agents.core.orchestrator import Orchestrator
from ai_data_analyst_agents.core.settings import load_app_cfg
from ai_data_analyst_agents.core.sql_source import SQLDataSource, choose_primary_table
from ai_data_analyst_agents.core.task_planner import default_tasks_phase2


def _redact_db_url(db_url: str) -> str:
    return re.sub(r"://([^:/?#]+):([^@/]+)@", r"://\1:***@", db_url)


def run_pipeline(
    db_url: str,
    business_question: str,
    base_table: str | None = None,
    preview_rows: int | None = None,
    artifact_callback: Callable[[Path, Path], None] | None = None,
) -> Path:
    cfg = load_app_cfg()
    store = ArtifactStore.create(cfg.runtime.artifacts_dir, on_artifact_written=artifact_callback)
    logger = setup_logging(cfg.runtime.log_level, store.path("logs.txt"))

    logger.info(f"Run dir: {store.run_dir}")
    logger.info("Loading SQL source.")

    sql_source = SQLDataSource(
        db_url=db_url,
        timeout_s=cfg.llm.timeout_s,
        max_query_rows=cfg.sql.default_query_row_limit,
        enforce_read_only_sql=cfg.security.enforce_read_only_sql,
    )
    schema = sql_source.inspect_schema(
        include_row_counts=cfg.sql.include_row_counts,
        max_tables=cfg.sql.introspection_max_tables,
    )
    store.write_json("db_schema.json", schema)
    logger.info("Wrote db_schema.json")

    table_names = {str(t.get("name", "")) for t in schema.get("tables", [])}
    analysis_table = base_table if base_table and base_table in table_names else choose_primary_table(schema)
    if not analysis_table:
        raise ValueError("No tables found in SQL source. Cannot run pipeline.")

    limit = int(preview_rows or cfg.sql.preview_rows)
    df = sql_source.load_table(analysis_table, limit=limit)
    logger.info(f"Loaded analysis table '{analysis_table}' with {len(df)} rows")

    memory = SharedMemory()
    evidence = EvidenceStore()

    ctx = {
        "cfg": cfg,
        "store": store,
        "logger": logger,
        "df": df,
        "business_question": business_question,
        "memory": memory,
        "evidence": evidence,
        "sql_source": sql_source,
        "sql_schema": schema,
        "source": {
            "type": "sql",
            "db_url": _redact_db_url(db_url),
            "analysis_table": analysis_table,
            "preview_rows": limit,
        },
    }

    agents = {
        "intake": IntakeAgent(),
        "profiling": ProfilingAgent(),
        "quality": QualityAgent(),
        "wrangling": WranglingAgent(),
        "planner": PlannerAgent(),
        "metrics": MetricsAgent(),
        "eda": EDAAgent(),
        "reporting": ReportingAgent(),
        "reviewer": ReviewerAgent(),
    }

    tasks = default_tasks_phase2()
    orch = Orchestrator(agents=agents, logger=logger)

    try:
        orch.run(tasks, ctx)
    except Exception:
        store.write_json(
            "agent_messages.json",
            [asdict(m) if hasattr(m, "__dataclass_fields__") else {"value": str(m)} for m in memory.messages],
        )
        logger.exception("Pipeline failed.")
        store.write_json(
            "run_manifest.json",
            {
                "status": "failed",
                "inputs": {
                    "db_url": _redact_db_url(db_url),
                    "business_question": business_question,
                    "analysis_table": analysis_table,
                },
                "tasks": [{"name": t.name, "reason": t.reason} for t in tasks],
                "evidence_ids": list(evidence.all().keys()),
            },
        )
        raise

    store.write_json(
        "run_manifest.json",
        {
            "status": "success",
            "inputs": {
                "db_url": _redact_db_url(db_url),
                "business_question": business_question,
                "analysis_table": analysis_table,
            },
            "tasks": [{"name": t.name, "reason": t.reason} for t in tasks],
            "evidence_ids": list(evidence.all().keys()),
        },
    )
    store.write_json(
        "agent_messages.json",
        [asdict(m) if hasattr(m, "__dataclass_fields__") else {"value": str(m)} for m in memory.messages],
    )
    logger.info("Done.")
    return store.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True, help="Database URL, e.g. sqlite:///data/my.db")
    parser.add_argument("--question", required=True, help="Business question to answer")
    parser.add_argument("--table", required=False, help="Optional primary analysis table")
    parser.add_argument("--preview-rows", type=int, required=False, help="Rows loaded to working dataframe")
    args = parser.parse_args()

    run_dir = run_pipeline(
        db_url=args.db_url,
        business_question=args.question,
        base_table=args.table,
        preview_rows=args.preview_rows,
    )
    print(str(run_dir))


if __name__ == "__main__":
    main()
