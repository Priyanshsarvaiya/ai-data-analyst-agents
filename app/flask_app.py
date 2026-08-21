from __future__ import annotations

import html
import json
import os
import secrets
import tempfile
from collections.abc import Callable
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

import markdown
import pandas as pd
from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from markupsafe import Markup
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from ai_data_analyst_agents.core.security import sanitize_user_error_message
from ai_data_analyst_agents.pipelines.run_csv_pipeline import run_pipeline as run_csv_pipeline
from ai_data_analyst_agents.pipelines.run_sql_pipeline import run_pipeline as run_sql_pipeline
from app.postgres_auth import (
    AuthUser,
    PostgresAuthStore,
    load_auth_settings,
    validate_password_strength,
)
from app.run_tracking import RunTrackingStore, execute_tracked_run

ROOT = Path(__file__).resolve().parents[1]
F = TypeVar("F", bound=Callable[..., Any])


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _render_markdown(path: Path) -> Markup:
    if not path.exists():
        return Markup("<p class='empty-state'>No report was generated.</p>")
    safe_source = html.escape(path.read_text(encoding="utf-8", errors="replace"))
    rendered = markdown.markdown(safe_source, extensions=["tables", "fenced_code", "sane_lists"])
    return Markup(rendered)


def _csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return str(token)


def _validate_csrf() -> None:
    supplied = request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    if not supplied or not expected or not secrets.compare_digest(str(supplied), str(expected)):
        abort(400, "Invalid form token. Refresh the page and try again.")


def _save_csv_uploads(uploads: list[FileStorage], upload_dir: Path) -> tuple[Path, list[str]]:
    """Save CSV uploads and combine compatible files into one pipeline input."""
    valid_uploads = [upload for upload in uploads if upload.filename]
    if not valid_uploads:
        raise ValueError("Choose at least one CSV file to analyze.")

    upload_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    filenames: list[str] = []
    saved_paths: list[Path] = []
    expected_columns: list[str] | None = None

    for index, upload in enumerate(valid_uploads, start=1):
        safe_name = secure_filename(str(upload.filename)) or f"dataset_{index}.csv"
        if not safe_name.lower().endswith(".csv"):
            raise ValueError(f"{safe_name} is not a CSV file.")
        destination = upload_dir / f"{index:02d}_{safe_name}"
        upload.save(destination)
        saved_paths.append(destination)
        try:
            frame = pd.read_csv(destination)
        except Exception as exc:
            raise ValueError(f"Could not read {safe_name} as CSV: {exc}") from exc
        columns = [str(column) for column in frame.columns]
        if expected_columns is None:
            expected_columns = columns
        elif set(columns) != set(expected_columns):
            missing = sorted(set(expected_columns) - set(columns))
            extra = sorted(set(columns) - set(expected_columns))
            details = []
            if missing:
                details.append(f"missing columns: {', '.join(missing)}")
            if extra:
                details.append(f"extra columns: {', '.join(extra)}")
            raise ValueError(
                f"{safe_name} does not match the first CSV schema ({'; '.join(details)})."
            )
        assert expected_columns is not None
        frames.append(frame.reindex(columns=expected_columns))
        filenames.append(safe_name)

    if len(frames) == 1:
        return saved_paths[0], filenames

    combined_path = upload_dir / "combined_upload.csv"
    pd.concat(frames, ignore_index=True).to_csv(combined_path, index=False)
    return combined_path, filenames


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    auth_cfg = load_auth_settings()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_mapping(
        SECRET_KEY=(
            os.getenv("FLASK_SECRET_KEY") or auth_cfg.AUTH_PASSWORD_PEPPER or secrets.token_hex(32)
        ),
        AUTH_DATABASE_URL=auth_cfg.resolved_database_url,
        AUTH_PASSWORD_PEPPER=auth_cfg.AUTH_PASSWORD_PEPPER,
        AUTH_PASSWORD_HASH_ITERATIONS=int(auth_cfg.AUTH_PASSWORD_HASH_ITERATIONS),
        AUTH_LOCKOUT_ATTEMPTS=int(auth_cfg.AUTH_LOCKOUT_ATTEMPTS),
        AUTH_LOCKOUT_MINUTES=int(auth_cfg.AUTH_LOCKOUT_MINUTES),
        PERMANENT_SESSION_LIFETIME=timedelta(minutes=int(auth_cfg.AUTH_SESSION_TTL_MIN)),
        MAX_CONTENT_LENGTH=int(auth_cfg.MAX_UPLOAD_MB) * 1024 * 1024,
        MAX_UPLOAD_MB=int(auth_cfg.MAX_UPLOAD_MB),
    )
    if test_config:
        app.config.update(test_config)

    database_url = str(app.config.get("AUTH_DATABASE_URL", "")).strip()
    if database_url:
        app.extensions["auth_store"] = PostgresAuthStore(
            database_url=database_url,
            pepper=str(app.config.get("AUTH_PASSWORD_PEPPER", "")),
            iterations=int(app.config.get("AUTH_PASSWORD_HASH_ITERATIONS", 390_000)),
            lock_after_failures=int(app.config.get("AUTH_LOCKOUT_ATTEMPTS", 5)),
            lock_minutes=int(app.config.get("AUTH_LOCKOUT_MINUTES", 15)),
        )
        app.extensions["run_store"] = RunTrackingStore(database_url=database_url)

    def auth_store() -> PostgresAuthStore:
        store = app.extensions.get("auth_store")
        if store is None:
            raise RuntimeError("AUTH_DATABASE_URL or DATABASE_URL is not configured.")
        return cast(PostgresAuthStore, store)

    def run_store() -> RunTrackingStore:
        store = app.extensions.get("run_store")
        if store is None:
            raise RuntimeError("AUTH_DATABASE_URL or DATABASE_URL is not configured.")
        return cast(RunTrackingStore, store)

    def current_user() -> AuthUser | None:
        user_id = session.get("user_id")
        if not user_id:
            return None
        user = auth_store().get_user_by_id(int(user_id))
        if user is None:
            session.clear()
        return user

    def require_current_user() -> AuthUser:
        """Return the authenticated user with an explicit non-optional type."""
        user = current_user()
        if user is None:
            abort(401)
        return user

    def login_required(view: F) -> F:
        @wraps(view)
        def wrapped(*args: Any, **kwargs: Any):
            if current_user() is None:
                flash("Please sign in to continue.", "info")
                return redirect(url_for("auth", next=request.path))
            return view(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    @app.context_processor
    def inject_globals() -> dict[str, Any]:
        return {"csrf_token": _csrf_token, "current_user": current_user()}

    @app.template_filter("filesize")
    def filesize(value: int | None) -> str:
        size = int(value or 0)
        if size < 1024:
            return f"{size} B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / (1024 * 1024):.1f} MB"

    @app.get("/")
    def index():
        return redirect(url_for("dashboard" if current_user() else "auth"))

    @app.route("/auth", methods=["GET", "POST"])
    def auth():
        if current_user() is not None:
            return redirect(url_for("dashboard"))
        mode = request.args.get("mode", "login")
        if request.method == "POST":
            _validate_csrf()
            mode = request.form.get("mode", "login")
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "")
            if mode == "signup":
                full_name = request.form.get("full_name", "").strip()
                confirm = request.form.get("confirm_password", "")
                if password != confirm:
                    flash("Passwords do not match.", "error")
                elif issues := validate_password_strength(password):
                    flash(" ".join(issues), "error")
                else:
                    ok, message = auth_store().create_user(
                        email=email,
                        full_name=full_name,
                        password=password,
                    )
                    flash(message, "success" if ok else "error")
                    if ok:
                        return redirect(url_for("auth", mode="login"))
            else:
                user, message = auth_store().authenticate(email=email, password=password)
                if user is None:
                    flash(message, "error")
                else:
                    session.clear()
                    session["user_id"] = int(user.id)
                    session.permanent = True
                    flash(f"Welcome back, {user.full_name.split()[0]}.", "success")
                    return redirect(url_for("dashboard"))
        return render_template("auth.html", mode=mode)

    @app.post("/logout")
    def logout():
        _validate_csrf()
        session.clear()
        return redirect(url_for("auth"))

    @app.get("/dashboard")
    @login_required
    def dashboard():
        user = require_current_user()
        runs = run_store().list_runs_for_user(user_id=int(user.id), limit=12)
        completed = sum(run.status == "completed" for run in runs)
        return render_template(
            "dashboard.html",
            runs=runs,
            completed=completed,
            max_upload_mb=int(app.config["MAX_UPLOAD_MB"]),
        )

    @app.post("/analyze")
    @login_required
    def analyze():
        _validate_csrf()
        user = require_current_user()
        source_type = request.form.get("source_type", "csv").strip().lower()
        question = request.form.get("question", "").strip()
        if not question:
            flash("Enter a business question before running the analysis.", "error")
            return redirect(url_for("dashboard"))

        metadata: dict[str, Any] = {
            "business_question": question,
            "source_type": source_type,
        }
        try:
            if source_type == "csv":
                upload_dir = Path(tempfile.mkdtemp(prefix="ai_analyst_"))
                csv_path, filenames = _save_csv_uploads(
                    request.files.getlist("csv_files"), upload_dir
                )
                source_name = filenames[0] if len(filenames) == 1 else f"{len(filenames)} CSV files"
                metadata.update(
                    {
                        "source_name": source_name,
                        "source_files": filenames,
                        "file_path": str(csv_path),
                    }
                )
                run, _ = execute_tracked_run(
                    run_store=run_store(),
                    user_id=int(user.id),
                    source_type="csv",
                    source_name=source_name,
                    run_metadata=metadata,
                    runner=lambda callback: run_csv_pipeline(
                        str(csv_path), question, artifact_callback=callback
                    ),
                )
            elif source_type == "sql":
                db_url = request.form.get("db_url", "").strip()
                table = request.form.get("base_table", "").strip()
                if not db_url:
                    raise ValueError("Enter a database URL to analyze.")
                metadata.update(
                    {"source_name": table or "sql_source", "analysis_table": table or None}
                )
                run, _ = execute_tracked_run(
                    run_store=run_store(),
                    user_id=int(user.id),
                    source_type="sql",
                    source_name=table or "sql_source",
                    run_metadata=metadata,
                    runner=lambda callback: run_sql_pipeline(
                        db_url=db_url,
                        business_question=question,
                        base_table=table or None,
                        artifact_callback=callback,
                    ),
                )
            else:
                raise ValueError("Unsupported data source type.")
        except Exception as exc:  # noqa: BLE001 - route boundary sanitizes pipeline failures
            flash(sanitize_user_error_message(exc), "error")
            return redirect(url_for("dashboard"))
        return redirect(url_for("run_detail", run_uuid=run.run_uuid))

    @app.get("/runs/<run_uuid>")
    @login_required
    def run_detail(run_uuid: str):
        user = require_current_user()
        detail = run_store().get_run_details_for_user(user_id=int(user.id), run_uuid=run_uuid)
        if detail is None:
            abort(404)
        run_dir = Path(str(detail.run.run_metadata.get("run_dir", "")))
        plan = _read_json(run_dir / "analysis_plan.json")
        profile = _read_json(run_dir / "data_profile.json")
        quality = _read_json(run_dir / "quality_report.json")
        eda = _read_json(run_dir / "eda_summary.json")
        review = _read_json(run_dir / "review_log.json")
        charts = [artifact for artifact in detail.artifacts if artifact.artifact_type == "chart"]
        return render_template(
            "run_detail.html",
            detail=detail,
            run=detail.run,
            plan=plan,
            profile=profile,
            quality=quality,
            eda=eda,
            review=review,
            charts=charts,
            report_html=_render_markdown(run_dir / "final_report.md"),
        )

    @app.get("/runs/<run_uuid>/artifacts/<path:artifact_path>")
    @login_required
    def artifact(run_uuid: str, artifact_path: str):
        user = require_current_user()
        detail = run_store().get_run_details_for_user(user_id=int(user.id), run_uuid=run_uuid)
        if detail is None:
            abort(404)
        allowed = {item.artifact_path for item in detail.artifacts}
        if artifact_path not in allowed:
            abort(404)
        run_dir = Path(str(detail.run.run_metadata.get("run_dir", ""))).resolve()
        target = (run_dir / artifact_path).resolve()
        if run_dir not in target.parents or not target.is_file():
            abort(404)
        return send_file(target, as_attachment=request.args.get("download") == "1")

    @app.errorhandler(413)
    def upload_too_large(_error):  # type: ignore[no-untyped-def]
        flash(f"Upload exceeds the {app.config['MAX_UPLOAD_MB']} MB limit.", "error")
        return redirect(url_for("dashboard"))

    return app


if __name__ == "__main__":
    create_app().run(debug=os.getenv("FLASK_DEBUG", "0") == "1")
