from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ai_data_analyst_agents.pipelines.run_csv_pipeline import run_pipeline as run_csv_pipeline
from ai_data_analyst_agents.pipelines.run_sql_pipeline import run_pipeline as run_sql_pipeline


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    source_type: str
    question: str
    file_path: str | None = None
    db_url: str | None = None
    base_table: str | None = None
    expected_route: str | None = None
    tags: List[str] = field(default_factory=list)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_path(base_dir: Path, p: str | None) -> str | None:
    if not p:
        return None
    candidate = Path(p)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def _resolve_sqlite_db_url(base_dir: Path, db_url: str) -> str:
    prefix = "sqlite:///"
    if not db_url.startswith(prefix):
        return db_url
    raw_path = db_url[len(prefix):]
    db_path = Path(raw_path)
    if db_path.is_absolute():
        return db_url
    return f"{prefix}{(base_dir / db_path).resolve()}"


def load_benchmark_suite(path: str | Path) -> List[BenchmarkCase]:
    p = Path(path)
    payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cases_raw = list(payload.get("cases", []) or [])
    out: List[BenchmarkCase] = []
    for i, row in enumerate(cases_raw, start=1):
        row = dict(row or {})
        out.append(
            BenchmarkCase(
                case_id=str(row.get("id") or f"case_{i}"),
                source_type=str(row.get("source_type", "csv")).strip().lower(),
                question=str(row.get("question", "")).strip(),
                file_path=row.get("file_path"),
                db_url=row.get("db_url"),
                base_table=row.get("base_table"),
                expected_route=row.get("expected_route"),
                tags=[str(t) for t in list(row.get("tags", []) or [])],
            )
        )
    return out


def _case_observation(case: BenchmarkCase, run_dir: Path, error: str | None = None) -> Dict[str, Any]:
    analysis_plan = _read_json(run_dir / "analysis_plan.json")
    analysis_tasks = _read_json(run_dir / "analysis_tasks.json")
    metrics_out = _read_json(run_dir / "metrics_outputs.json")
    review_log = _read_json(run_dir / "review_log.json")
    report_meta = _read_json(run_dir / "report_metadata.json")
    scorecard = _read_json(run_dir / "run_scorecard.json")

    planned_tasks = list(analysis_tasks.get("tasks", []) or [])
    planned_stat_ids = {
        str(t.get("id", "")).strip()
        for t in planned_tasks
        if str(t.get("type", "")).strip() in {"statistical_test", "ab_test", "ols_regression"}
    }
    computed_stat = [
        x for x in list(metrics_out.get("computed", []) or [])
        if str(x.get("task_id", "")).strip() in planned_stat_ids
    ]
    artifacts = [str(x.get("artifact", "")).strip() for x in list(metrics_out.get("computed", []) or []) if str(x.get("artifact", "")).strip()]
    duplicate_artifacts = max(0, len(artifacts) - len(set(artifacts)))

    route = str(analysis_plan.get("analysis_type") or "")
    route_correct = None
    if case.expected_route:
        route_correct = (route == case.expected_route)

    reviewer_status = str(review_log.get("status", "")).strip().lower()
    section_ok = bool(report_meta.get("section_completeness_ok", False))
    unsupported_claim = int(report_meta.get("unsupported_numeric_claim_lines", 0)) > 0
    if reviewer_status == "pass":
        relevance = 1.0
    elif section_ok:
        relevance = 0.6
    elif len(metrics_out.get("computed", []) or []) > 0:
        relevance = 0.4
    else:
        relevance = 0.2

    return {
        "id": case.case_id,
        "source_type": case.source_type,
        "question": case.question,
        "tags": case.tags,
        "run_dir": str(run_dir),
        "error": error,
        "expected_route": case.expected_route,
        "actual_route": route or None,
        "route_correct": route_correct,
        "unsupported_claim": unsupported_claim,
        "planned_stat_task_count": len(planned_stat_ids),
        "computed_stat_task_count": len(computed_stat),
        "computed_artifact_count": len(artifacts),
        "duplicate_artifact_count": duplicate_artifacts,
        "answer_relevance_score": relevance,
        "reviewer_status": reviewer_status or None,
        "final_quality_status": scorecard.get("final_quality_status"),
    }


def compute_quality_kpis(case_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    rows = [dict(r or {}) for r in case_rows]
    n = len(rows)
    if n == 0:
        return {
            "unsupported_claim_rate": 0.0,
            "stat_task_success_rate": 0.0,
            "duplicate_artifact_rate": 0.0,
            "answer_relevance_score": 0.0,
            "route_accuracy": 0.0,
            "case_count": 0.0,
        }

    unsupported_claim_rate = sum(1 for r in rows if bool(r.get("unsupported_claim"))) / float(n)

    total_stat_planned = sum(int(r.get("planned_stat_task_count", 0) or 0) for r in rows)
    total_stat_computed = sum(int(r.get("computed_stat_task_count", 0) or 0) for r in rows)
    stat_task_success_rate = (total_stat_computed / float(total_stat_planned)) if total_stat_planned else 1.0

    total_computed_artifacts = sum(int(r.get("computed_artifact_count", 0) or 0) for r in rows)
    total_duplicate_artifacts = sum(int(r.get("duplicate_artifact_count", 0) or 0) for r in rows)
    duplicate_artifact_rate = (
        total_duplicate_artifacts / float(total_computed_artifacts)
        if total_computed_artifacts
        else 0.0
    )

    answer_relevance_score = sum(float(r.get("answer_relevance_score", 0.0) or 0.0) for r in rows) / float(n)

    route_rows = [r for r in rows if r.get("route_correct") is not None]
    route_accuracy = (
        sum(1 for r in route_rows if bool(r.get("route_correct"))) / float(len(route_rows))
        if route_rows
        else 0.0
    )

    return {
        "unsupported_claim_rate": round(unsupported_claim_rate, 4),
        "stat_task_success_rate": round(stat_task_success_rate, 4),
        "duplicate_artifact_rate": round(duplicate_artifact_rate, 4),
        "answer_relevance_score": round(answer_relevance_score, 4),
        "route_accuracy": round(route_accuracy, 4),
        "case_count": float(n),
    }


def run_benchmark_suite(suite_path: str | Path, output_dir: str | Path | None = None) -> Path:
    suite_file = Path(suite_path).resolve()
    suite_dir = suite_file.parent
    cases = load_benchmark_suite(suite_file)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        out_dir = Path("artifacts") / "evals" / f"eval_{ts}"
    else:
        out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for case in cases:
        run_dir: Path | None = None
        error: str | None = None
        try:
            if case.source_type == "csv":
                fp = _resolve_path(suite_dir, case.file_path)
                if not fp:
                    raise ValueError(f"CSV case '{case.case_id}' missing file_path.")
                run_dir = Path(run_csv_pipeline(fp, case.question))
            elif case.source_type == "sql":
                if not case.db_url:
                    raise ValueError(f"SQL case '{case.case_id}' missing db_url.")
                resolved_db_url = _resolve_sqlite_db_url(suite_dir, case.db_url)
                run_dir = Path(
                    run_sql_pipeline(
                        db_url=resolved_db_url,
                        business_question=case.question,
                        base_table=case.base_table,
                    )
                )
            else:
                raise ValueError(f"Unsupported source_type '{case.source_type}' in case '{case.case_id}'.")
        except Exception as exc:
            error = str(exc)

        if run_dir is None:
            rows.append(
                {
                    "id": case.case_id,
                    "source_type": case.source_type,
                    "question": case.question,
                    "tags": case.tags,
                    "run_dir": None,
                    "error": error,
                    "expected_route": case.expected_route,
                    "actual_route": None,
                    "route_correct": False if case.expected_route else None,
                    "unsupported_claim": True,
                    "planned_stat_task_count": 0,
                    "computed_stat_task_count": 0,
                    "computed_artifact_count": 0,
                    "duplicate_artifact_count": 0,
                    "answer_relevance_score": 0.0,
                    "reviewer_status": None,
                    "final_quality_status": "fail",
                }
            )
            continue
        rows.append(_case_observation(case, run_dir, error=error))

    summary = {
        "suite_file": str(suite_file),
        "run_at": datetime.now().isoformat(),
        "cases": rows,
        "kpis": compute_quality_kpis(rows),
    }
    summary_path = out_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path
