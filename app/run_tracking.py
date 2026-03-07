from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import mimetypes
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    inspect,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from ai_data_analyst_agents.core.security import sanitize_user_error_message
try:
    from app.postgres_auth import Base, UserModel, ensure_users_identity_constraints
except ModuleNotFoundError:
    from postgres_auth import Base, UserModel, ensure_users_identity_constraints


ArtifactObserver = Callable[[Path, Path], None]
RunExecutor = Callable[[ArtifactObserver], Path]

RUN_STATUS_QUEUED = "queued"
RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"
RUN_STATUSES = {
    RUN_STATUS_QUEUED,
    RUN_STATUS_RUNNING,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
}
RUN_TRANSITIONS: dict[str, set[str]] = {
    RUN_STATUS_QUEUED: {RUN_STATUS_RUNNING, RUN_STATUS_COMPLETED, RUN_STATUS_FAILED},
    RUN_STATUS_RUNNING: {RUN_STATUS_COMPLETED, RUN_STATUS_FAILED},
    RUN_STATUS_COMPLETED: set(),
    RUN_STATUS_FAILED: set(),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    v = _ensure_utc(dt)
    return v.isoformat() if v else None


class AnalysisRunModel(Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed')",
            name="ck_analysis_runs_status",
        ),
        UniqueConstraint("run_uuid", name="uq_analysis_runs_run_uuid"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    run_uuid: Mapped[str] = mapped_column(String(64), nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=RUN_STATUS_QUEUED)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    run_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now)


class RunArtifactModel(Base):
    __tablename__ = "run_artifacts"
    __table_args__ = (
        UniqueConstraint("run_id", "artifact_path", name="uq_run_artifacts_run_path"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_name: Mapped[str] = mapped_column(String(255), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    artifact_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now)


@dataclass(frozen=True)
class RunArtifact:
    id: int
    run_id: int
    artifact_type: str
    artifact_name: str
    artifact_path: str
    mime_type: str | None
    file_size_bytes: int | None
    artifact_metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class AnalysisRun:
    id: int
    user_id: int
    run_uuid: str
    source_type: str
    source_name: str | None
    status: str
    started_at: str | None
    completed_at: str | None
    error_message: str | None
    report_path: str | None
    run_metadata: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AnalysisRunDetail:
    run: AnalysisRun
    artifacts: list[RunArtifact]


class RunTrackingStore:
    def __init__(self, *, database_url: str) -> None:
        if not database_url:
            raise RuntimeError("AUTH_DATABASE_URL or DATABASE_URL is required for run tracking.")
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.session_local = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False)
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        Base.metadata.create_all(
            bind=self.engine,
            tables=[UserModel.__table__, AnalysisRunModel.__table__, RunArtifactModel.__table__],
        )
        ensure_users_identity_constraints(self.engine)

        with self.engine.begin() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_analysis_runs_user_id ON analysis_runs (user_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_analysis_runs_status ON analysis_runs (status)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_analysis_runs_created_at ON analysis_runs (created_at)"))
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_analysis_runs_run_uuid ON analysis_runs (run_uuid)"))

            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_run_artifacts_run_id ON run_artifacts (run_id)"))
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS ix_run_artifacts_artifact_type ON run_artifacts (artifact_type)")
            )
            conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_run_artifacts_run_path "
                    "ON run_artifacts (run_id, artifact_path)"
                )
            )

            if self.engine.dialect.name == "postgresql":
                conn.execute(text("ALTER TABLE analysis_runs ALTER COLUMN user_id SET NOT NULL"))
                conn.execute(text("ALTER TABLE run_artifacts ALTER COLUMN run_id SET NOT NULL"))
                conn.execute(text("ALTER TABLE analysis_runs ALTER COLUMN run_uuid SET NOT NULL"))

        inspector = inspect(self.engine)
        tables = set(inspector.get_table_names())
        missing = {"analysis_runs", "run_artifacts"} - tables
        if missing:
            raise RuntimeError(f"Missing required run-tracking tables: {sorted(missing)}")

    def _to_run(self, row: AnalysisRunModel) -> AnalysisRun:
        return AnalysisRun(
            id=int(row.id),
            user_id=int(row.user_id),
            run_uuid=str(row.run_uuid),
            source_type=str(row.source_type),
            source_name=str(row.source_name) if row.source_name is not None else None,
            status=str(row.status),
            started_at=_iso(row.started_at),
            completed_at=_iso(row.completed_at),
            error_message=str(row.error_message) if row.error_message else None,
            report_path=str(row.report_path) if row.report_path else None,
            run_metadata=dict(row.run_metadata or {}),
            created_at=_iso(row.created_at) or "",
            updated_at=_iso(row.updated_at) or "",
        )

    def _to_artifact(self, row: RunArtifactModel) -> RunArtifact:
        return RunArtifact(
            id=int(row.id),
            run_id=int(row.run_id),
            artifact_type=str(row.artifact_type),
            artifact_name=str(row.artifact_name),
            artifact_path=str(row.artifact_path),
            mime_type=str(row.mime_type) if row.mime_type else None,
            file_size_bytes=int(row.file_size_bytes) if row.file_size_bytes is not None else None,
            artifact_metadata=dict(row.artifact_metadata or {}),
            created_at=_iso(row.created_at) or "",
        )

    def _assert_user_exists(self, user_id: int) -> None:
        with self.session_local() as db:
            user = db.execute(select(UserModel.id).where(UserModel.id == int(user_id))).first()
            if user is None:
                raise ValueError(f"User id {user_id} does not exist.")

    def _assert_valid_transition(self, current: str, nxt: str) -> None:
        if nxt not in RUN_STATUSES:
            raise ValueError(f"Invalid run status '{nxt}'.")
        if current == nxt:
            return
        if nxt not in RUN_TRANSITIONS.get(current, set()):
            raise ValueError(f"Invalid run status transition: {current} -> {nxt}")

    def _merge_metadata(self, current: dict[str, Any] | None, updates: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(current or {})
        if updates:
            merged.update(updates)
        return merged

    def create_run(
        self,
        *,
        user_id: int,
        source_type: str,
        source_name: str | None,
        run_metadata: dict[str, Any] | None = None,
        status: str = RUN_STATUS_QUEUED,
        run_uuid: str | None = None,
    ) -> AnalysisRun:
        self._assert_user_exists(user_id)
        if status not in RUN_STATUSES:
            raise ValueError(f"Invalid run status '{status}'.")
        now = _utc_now()
        row = AnalysisRunModel(
            user_id=int(user_id),
            run_uuid=str(run_uuid or uuid4()),
            source_type=str(source_type).strip().lower() or "unknown",
            source_name=(source_name or "").strip() or None,
            status=status,
            started_at=now if status == RUN_STATUS_RUNNING else None,
            completed_at=now if status in {RUN_STATUS_COMPLETED, RUN_STATUS_FAILED} else None,
            error_message=None,
            report_path=None,
            run_metadata=dict(run_metadata or {}),
            created_at=now,
            updated_at=now,
        )
        try:
            with self.session_local() as db:
                db.add(row)
                db.commit()
                db.refresh(row)
                return self._to_run(row)
        except IntegrityError as exc:
            raise ValueError("Failed to create run record due to unique/foreign-key constraints.") from exc

    def get_run_by_id(self, run_id: int) -> AnalysisRun | None:
        with self.session_local() as db:
            row = db.execute(select(AnalysisRunModel).where(AnalysisRunModel.id == int(run_id))).scalar_one_or_none()
            if row is None:
                return None
            return self._to_run(row)

    def update_run_status(
        self,
        *,
        run_id: int,
        status: str,
        error_message: str | None = None,
        report_path: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> AnalysisRun:
        with self.session_local() as db:
            row = db.execute(select(AnalysisRunModel).where(AnalysisRunModel.id == int(run_id))).scalar_one_or_none()
            if row is None:
                raise ValueError(f"Run id {run_id} does not exist.")
            self._assert_valid_transition(str(row.status), status)

            now = _utc_now()
            row.status = status
            row.updated_at = now
            if status == RUN_STATUS_RUNNING and row.started_at is None:
                row.started_at = now
            if status in {RUN_STATUS_COMPLETED, RUN_STATUS_FAILED} and row.completed_at is None:
                row.completed_at = now
            if error_message is not None:
                row.error_message = str(error_message)[:4000] if error_message else None
            if report_path is not None:
                row.report_path = str(report_path) if report_path else None
            if metadata_updates:
                row.run_metadata = self._merge_metadata(row.run_metadata, metadata_updates)

            db.add(row)
            db.commit()
            db.refresh(row)
            return self._to_run(row)

    def mark_running(self, *, run_id: int) -> AnalysisRun:
        return self.update_run_status(run_id=run_id, status=RUN_STATUS_RUNNING)

    def mark_completed(
        self,
        *,
        run_id: int,
        report_path: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> AnalysisRun:
        return self.update_run_status(
            run_id=run_id,
            status=RUN_STATUS_COMPLETED,
            report_path=report_path,
            metadata_updates=metadata_updates,
        )

    def mark_failed(
        self,
        *,
        run_id: int,
        error_message: str,
        metadata_updates: dict[str, Any] | None = None,
    ) -> AnalysisRun:
        return self.update_run_status(
            run_id=run_id,
            status=RUN_STATUS_FAILED,
            error_message=error_message,
            metadata_updates=metadata_updates,
        )

    def relative_artifact_path(self, *, run_dir: Path, artifact_path: Path) -> str:
        run_dir_r = Path(run_dir).resolve()
        artifact_r = Path(artifact_path).resolve()
        try:
            return artifact_r.relative_to(run_dir_r).as_posix()
        except Exception:
            return artifact_r.as_posix()

    def infer_artifact_type(self, artifact_path: str) -> str:
        p = artifact_path.lower()
        name = Path(p).name
        if p.startswith("charts/") or name.startswith("qa_") and name.endswith(".png"):
            return "chart"
        if name == "final_report.md":
            return "markdown_report"
        if name == "cleaned.csv":
            return "cleaned_csv"
        if name == "data_profile.json":
            return "profile_json"
        if name == "quality_report.json":
            return "validation_json"
        if name.endswith(".sql"):
            return "sql_query"
        if name.endswith(".json"):
            return "json"
        if name.endswith(".csv"):
            return "csv"
        if name.endswith(".md"):
            return "markdown"
        if name.endswith(".txt"):
            return "text"
        return "artifact"

    def attach_artifact(
        self,
        *,
        run_id: int,
        artifact_type: str,
        artifact_name: str,
        artifact_path: str,
        mime_type: str | None = None,
        file_size_bytes: int | None = None,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> RunArtifact:
        normalized_path = str(artifact_path).replace("\\", "/")
        with self.session_local() as db:
            run = db.execute(select(AnalysisRunModel).where(AnalysisRunModel.id == int(run_id))).scalar_one_or_none()
            if run is None:
                raise ValueError(f"Run id {run_id} does not exist.")

            row = db.execute(
                select(RunArtifactModel).where(
                    RunArtifactModel.run_id == int(run_id),
                    RunArtifactModel.artifact_path == normalized_path,
                )
            ).scalar_one_or_none()

            if row is None:
                row = RunArtifactModel(
                    run_id=int(run_id),
                    artifact_type=str(artifact_type),
                    artifact_name=str(artifact_name),
                    artifact_path=normalized_path,
                    mime_type=str(mime_type) if mime_type else None,
                    file_size_bytes=int(file_size_bytes) if file_size_bytes is not None else None,
                    artifact_metadata=dict(artifact_metadata or {}),
                    created_at=_utc_now(),
                )
            else:
                row.artifact_type = str(artifact_type)
                row.artifact_name = str(artifact_name)
                row.mime_type = str(mime_type) if mime_type else None
                row.file_size_bytes = int(file_size_bytes) if file_size_bytes is not None else None
                row.artifact_metadata = dict(artifact_metadata or {})

            db.add(row)
            db.commit()
            db.refresh(row)
            return self._to_artifact(row)

    def attach_artifact_from_file(self, *, run_id: int, run_dir: Path, artifact_file: Path) -> RunArtifact | None:
        p = Path(artifact_file)
        if not p.exists() or not p.is_file():
            return None

        rel_path = self.relative_artifact_path(run_dir=Path(run_dir), artifact_path=p)
        guessed_mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        metadata = {"ext": p.suffix.lower()}
        return self.attach_artifact(
            run_id=run_id,
            artifact_type=self.infer_artifact_type(rel_path),
            artifact_name=p.name,
            artifact_path=rel_path,
            mime_type=guessed_mime,
            file_size_bytes=p.stat().st_size,
            artifact_metadata=metadata,
        )

    def sync_artifacts_from_run_dir(self, *, run_id: int, run_dir: Path) -> list[RunArtifact]:
        root = Path(run_dir)
        if not root.exists():
            return []
        out: list[RunArtifact] = []
        for p in sorted(root.rglob("*")):
            if p.is_file():
                row = self.attach_artifact_from_file(run_id=run_id, run_dir=root, artifact_file=p)
                if row is not None:
                    out.append(row)
        return out

    def list_runs_for_user(self, *, user_id: int, limit: int = 50, offset: int = 0) -> list[AnalysisRun]:
        with self.session_local() as db:
            rows = db.execute(
                select(AnalysisRunModel)
                .where(AnalysisRunModel.user_id == int(user_id))
                .order_by(AnalysisRunModel.created_at.desc())
                .offset(int(offset))
                .limit(int(limit))
            ).scalars()
            return [self._to_run(x) for x in rows]

    def get_run_details_for_user(self, *, user_id: int, run_uuid: str) -> AnalysisRunDetail | None:
        with self.session_local() as db:
            run = db.execute(
                select(AnalysisRunModel).where(
                    AnalysisRunModel.user_id == int(user_id),
                    AnalysisRunModel.run_uuid == str(run_uuid),
                )
            ).scalar_one_or_none()
            if run is None:
                return None
            artifacts = db.execute(
                select(RunArtifactModel)
                .where(RunArtifactModel.run_id == int(run.id))
                .order_by(RunArtifactModel.created_at.asc(), RunArtifactModel.id.asc())
            ).scalars()
            return AnalysisRunDetail(
                run=self._to_run(run),
                artifacts=[self._to_artifact(x) for x in artifacts],
            )


def execute_tracked_run(
    *,
    run_store: RunTrackingStore,
    user_id: int,
    source_type: str,
    source_name: str | None,
    run_metadata: dict[str, Any] | None,
    runner: RunExecutor,
) -> tuple[AnalysisRun, Path]:
    run = run_store.create_run(
        user_id=int(user_id),
        source_type=source_type,
        source_name=source_name,
        run_metadata=run_metadata,
        status=RUN_STATUS_QUEUED,
    )
    run_store.mark_running(run_id=run.id)

    observed_run_dir: Path | None = None

    def _artifact_observer(run_dir: Path, artifact_path: Path) -> None:
        nonlocal observed_run_dir
        observed_run_dir = Path(run_dir)
        run_store.attach_artifact_from_file(
            run_id=run.id,
            run_dir=Path(run_dir),
            artifact_file=Path(artifact_path),
        )

    try:
        run_dir = Path(runner(_artifact_observer))
        observed_run_dir = run_dir
        run_store.sync_artifacts_from_run_dir(run_id=run.id, run_dir=run_dir)
        report_path = None
        final_report = run_dir / "final_report.md"
        if final_report.exists():
            report_path = run_store.relative_artifact_path(run_dir=run_dir, artifact_path=final_report)
        completed = run_store.mark_completed(
            run_id=run.id,
            report_path=report_path,
            metadata_updates={"run_dir": str(run_dir)},
        )
        return completed, run_dir
    except Exception as exc:
        updates = {}
        if observed_run_dir is not None:
            updates["run_dir"] = str(observed_run_dir)
            try:
                run_store.sync_artifacts_from_run_dir(run_id=run.id, run_dir=observed_run_dir)
            except Exception:
                pass
        run_store.mark_failed(
            run_id=run.id,
            error_message=sanitize_user_error_message(exc),
            metadata_updates=updates or None,
        )
        raise
