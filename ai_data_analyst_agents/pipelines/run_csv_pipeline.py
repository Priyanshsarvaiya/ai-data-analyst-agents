from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import pandas as pd

from ai_data_analyst_agents.core.artifacts import ArtifactStore
from ai_data_analyst_agents.core.logging import setup_logging
from ai_data_analyst_agents.core.settings import load_app_cfg

from ai_data_analyst_agents.core.memory import SharedMemory
from ai_data_analyst_agents.core.evidence import EvidenceStore
from ai_data_analyst_agents.core.orchestrator import Orchestrator
from ai_data_analyst_agents.core.task_planner import default_tasks_phase2

from ai_data_analyst_agents.agents.intake import IntakeAgent
from ai_data_analyst_agents.agents.profiling import ProfilingAgent
from ai_data_analyst_agents.agents.quality import QualityAgent
from ai_data_analyst_agents.agents.wrangling import WranglingAgent
from ai_data_analyst_agents.agents.eda import EDAAgent
from ai_data_analyst_agents.agents.planner import PlannerAgent
from ai_data_analyst_agents.agents.metrics import MetricsAgent
from ai_data_analyst_agents.agents.reporting import ReportingAgent
from ai_data_analyst_agents.agents.reviewer import ReviewerAgent


def run_pipeline(
    file_path: str,
    business_question: str,
    artifact_callback: Callable[[Path, Path], None] | None = None,
) -> Path:
    cfg = load_app_cfg()
    store = ArtifactStore.create(cfg.runtime.artifacts_dir, on_artifact_written=artifact_callback)
    logger = setup_logging(cfg.runtime.log_level, store.path("logs.txt"))

    logger.info(f"Run dir: {store.run_dir}")
    logger.info(f"Loading CSV: {file_path}")

    df = pd.read_csv(file_path)

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
        "source": {"type": "csv", "file_path": file_path},
    }

    # ✅ Agent registry (planner + metrics included)
    agents = {
        "intake": IntakeAgent(),
        "profiling": ProfilingAgent(),
        "quality": QualityAgent(),
        "wrangling": WranglingAgent(),
        "eda": EDAAgent(),
        "planner": PlannerAgent(),
        "metrics": MetricsAgent(),
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
        # Ensure a failing run still leaves a trace in artifacts/logs
        logger.exception("Pipeline failed.")
        store.write_json(
            "run_manifest.json",
            {
                "status": "failed",
                "inputs": {"file_path": file_path, "business_question": business_question},
                "tasks": [{"name": t.name, "reason": t.reason} for t in tasks],
                "evidence_ids": list(evidence.all().keys()),
            },
        )
        raise

    store.write_json(
        "run_manifest.json",
        {
            "status": "success",
            "inputs": {"file_path": file_path, "business_question": business_question},
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
    parser.add_argument("--file", required=True, help="Path to CSV file (e.g., data/sample.csv)")
    parser.add_argument("--question", required=True, help="Business question to answer")
    args = parser.parse_args()

    run_dir = run_pipeline(args.file, args.question)
    print(str(run_dir))


if __name__ == "__main__":
    main()
