from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect

from app.postgres_auth import PostgresAuthStore
from app.run_tracking import (
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_RUNNING,
    RunTrackingStore,
    execute_tracked_run,
)


def _email_has_unique_constraint_or_index(engine) -> bool:
    inspector = inspect(engine)
    for c in inspector.get_unique_constraints("users"):
        cols = [str(x).lower() for x in (c.get("column_names") or [])]
        if "email" in cols:
            return True
    for idx in inspector.get_indexes("users"):
        cols = [str(x).lower() for x in (idx.get("column_names") or [])]
        if idx.get("unique") and "email" in cols:
            return True
    return False


def _create_user(auth: PostgresAuthStore, email: str, name: str) -> int:
    ok, msg = auth.create_user(email=email, full_name=name, password="StrongPass1!")
    assert ok, msg
    user, msg = auth.authenticate(email=email, password="StrongPass1!")
    assert user is not None, msg
    return int(user.id)


def test_users_identity_constraints_present(tmp_path: Path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'auth.db').as_posix()}"
    auth = PostgresAuthStore(database_url=db_url, iterations=120_000)
    _create_user(auth, "analyst@example.com", "Analyst")

    inspector = inspect(auth.engine)
    pk = inspector.get_pk_constraint("users") or {}
    pk_cols = [str(c).lower() for c in (pk.get("constrained_columns") or [])]
    assert "id" in pk_cols
    assert _email_has_unique_constraint_or_index(auth.engine)


def test_user_to_run_and_run_to_artifact_relationship(tmp_path: Path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    auth = PostgresAuthStore(database_url=db_url, iterations=120_000)
    user_id = _create_user(auth, "owner@example.com", "Run Owner")
    store = RunTrackingStore(database_url=db_url)

    run = store.create_run(
        user_id=user_id,
        source_type="csv",
        source_name="orders.csv",
        run_metadata={"business_question": "Why did India have less revenue?"},
    )
    run = store.mark_running(run_id=run.id)
    assert run.status == RUN_STATUS_RUNNING

    run_dir = tmp_path / "artifacts" / "run_test"
    (run_dir / "charts").mkdir(parents=True, exist_ok=True)
    (run_dir / "final_report.md").write_text("# report", encoding="utf-8")
    (run_dir / "data_profile.json").write_text("{}", encoding="utf-8")
    (run_dir / "charts" / "qa_chart.png").write_bytes(b"png-bytes")

    synced = store.sync_artifacts_from_run_dir(run_id=run.id, run_dir=run_dir)
    assert len(synced) >= 3

    run = store.mark_completed(
        run_id=run.id,
        report_path=store.relative_artifact_path(run_dir=run_dir, artifact_path=run_dir / "final_report.md"),
    )
    assert run.status == RUN_STATUS_COMPLETED

    detail = store.get_run_details_for_user(user_id=user_id, run_uuid=run.run_uuid)
    assert detail is not None
    assert detail.run.user_id == user_id
    assert any(a.artifact_type == "markdown_report" for a in detail.artifacts)
    assert any(a.artifact_type == "chart" for a in detail.artifacts)


def test_run_status_transitions_and_failure_path(tmp_path: Path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'status.db').as_posix()}"
    auth = PostgresAuthStore(database_url=db_url, iterations=120_000)
    user_id = _create_user(auth, "status@example.com", "Status User")
    store = RunTrackingStore(database_url=db_url)

    run = store.create_run(user_id=user_id, source_type="csv", source_name="input.csv", run_metadata={})
    run = store.mark_running(run_id=run.id)
    run = store.mark_failed(run_id=run.id, error_message="forced failure")
    assert run.status == RUN_STATUS_FAILED
    assert run.completed_at is not None
    assert "forced failure" in (run.error_message or "")

    with pytest.raises(ValueError, match="Invalid run status transition"):
        store.mark_completed(run_id=run.id)


def test_list_runs_scoped_by_user(tmp_path: Path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'scope.db').as_posix()}"
    auth = PostgresAuthStore(database_url=db_url, iterations=120_000)
    user1 = _create_user(auth, "user1@example.com", "User One")
    user2 = _create_user(auth, "user2@example.com", "User Two")
    store = RunTrackingStore(database_url=db_url)

    store.create_run(user_id=user1, source_type="csv", source_name="u1.csv", run_metadata={})
    store.create_run(user_id=user2, source_type="sql", source_name="u2", run_metadata={})
    store.create_run(user_id=user1, source_type="csv", source_name="u1b.csv", run_metadata={})

    rows = store.list_runs_for_user(user_id=user1, limit=20)
    assert len(rows) == 2
    assert all(r.user_id == user1 for r in rows)


def test_run_uuid_uniqueness_constraint(tmp_path: Path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'uuid.db').as_posix()}"
    auth = PostgresAuthStore(database_url=db_url, iterations=120_000)
    user_id = _create_user(auth, "uuid@example.com", "UUID User")
    store = RunTrackingStore(database_url=db_url)

    store.create_run(
        user_id=user_id,
        source_type="csv",
        source_name="first.csv",
        run_metadata={},
        run_uuid="run-fixed-uuid",
    )
    with pytest.raises(ValueError, match="unique/foreign-key constraints"):
        store.create_run(
            user_id=user_id,
            source_type="csv",
            source_name="second.csv",
            run_metadata={},
            run_uuid="run-fixed-uuid",
        )


def test_execute_tracked_run_success_and_failure(tmp_path: Path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'workflow.db').as_posix()}"
    auth = PostgresAuthStore(database_url=db_url, iterations=120_000)
    user_id = _create_user(auth, "workflow@example.com", "Workflow User")
    store = RunTrackingStore(database_url=db_url)

    ok_run_dir = tmp_path / "artifacts" / "run_ok"
    ok_run_dir.mkdir(parents=True, exist_ok=True)
    (ok_run_dir / "final_report.md").write_text("# ok", encoding="utf-8")

    def _ok_runner(cb):  # noqa: ANN001
        cb(ok_run_dir, ok_run_dir / "final_report.md")
        return ok_run_dir

    run, run_dir = execute_tracked_run(
        run_store=store,
        user_id=user_id,
        source_type="csv",
        source_name="ok.csv",
        run_metadata={"business_question": "ok?"},
        runner=_ok_runner,
    )
    assert run.status == RUN_STATUS_COMPLETED
    assert run_dir == ok_run_dir
    detail = store.get_run_details_for_user(user_id=user_id, run_uuid=run.run_uuid)
    assert detail is not None
    assert detail.run.run_metadata.get("run_dir") == str(ok_run_dir)
    assert any(a.artifact_path == "final_report.md" for a in detail.artifacts)

    fail_run_dir = tmp_path / "artifacts" / "run_fail"
    fail_run_dir.mkdir(parents=True, exist_ok=True)
    (fail_run_dir / "partial.json").write_text("{}", encoding="utf-8")

    def _fail_runner(cb):  # noqa: ANN001
        cb(fail_run_dir, fail_run_dir / "partial.json")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        execute_tracked_run(
            run_store=store,
            user_id=user_id,
            source_type="sql",
            source_name="orders",
            run_metadata={"business_question": "fail?"},
            runner=_fail_runner,
        )

    runs = store.list_runs_for_user(user_id=user_id, limit=10)
    failed = next(r for r in runs if r.status == RUN_STATUS_FAILED)
    failed_detail = store.get_run_details_for_user(user_id=user_id, run_uuid=failed.run_uuid)
    assert failed_detail is not None
    assert any(a.artifact_path == "partial.json" for a in failed_detail.artifacts)
