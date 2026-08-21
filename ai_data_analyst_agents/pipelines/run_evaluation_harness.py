from __future__ import annotations

import argparse
from pathlib import Path

from ai_data_analyst_agents.evaluation.harness import run_benchmark_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        required=False,
        default="benchmarks/core_quality_suite.yaml",
        help="Path to benchmark suite YAML.",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Optional output directory for evaluation summary artifacts.",
    )
    args = parser.parse_args()

    summary = run_benchmark_suite(
        suite_path=Path(args.suite),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(str(summary))


if __name__ == "__main__":
    main()
