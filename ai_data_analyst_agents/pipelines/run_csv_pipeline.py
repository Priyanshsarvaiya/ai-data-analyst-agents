from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ai_data_analyst_agents.core.artifacts import ArtifactStore
from ai_data_analyst_agents.core.logging import setup_logging
from ai_data_analyst_agents.core.settings import load_app_cfg

from ai_data_analyst_agents.agents.intake import run_intake
from ai_data_analyst_agents.agents.profiling import run_profiling
from ai_data_analyst_agents.agents.quality import run_quality
from ai_data_analyst_agents.agents.wrangling import run_wrangling
from ai_data_analyst_agents.agents.eda import run_eda
from ai_data_analyst_agents.agents.reporting import run_reporting
from ai_data_analyst_agents.agents.reviewer import run_reviewer


def run_pipeline(file_path: str, business_question: str) -> Path:
    cfg = load_app_cfg()
    store = ArtifactStore.create(cfg.runtime.artifacts_dir)
    logger = setup_logging(cfg.runtime.log_level, store.path("logs.txt"))

    logger.info(f"Run dir: {store.run_dir}")
    logger.info(f"Loading CSV: {file_path}")

    df = pd.read_csv(file_path)

    plan = run_intake(cfg, df, business_question, store, logger)
    profile = run_profiling(cfg, df, store, logger)
    qa = run_quality(cfg, df, store, logger)
    clean_df, feature_log = run_wrangling(cfg, df, store, logger)
    eda = run_eda(cfg, clean_df, store, logger)
    report_md = run_reporting(cfg, plan, profile, qa, eda, store, logger)
    review = run_reviewer(cfg, report_md, store, logger)

    store.write_json(
        "run_manifest.json",
        {
            "inputs": {"file_path": file_path, "business_question": business_question},
            "artifacts_dir": str(store.run_dir),
        },
    )

    logger.info("Done.")
    return store.run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    run_dir = run_pipeline(args.file, args.question)
    print(str(run_dir))


if __name__ == "__main__":
    main()