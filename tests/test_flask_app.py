from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from werkzeug.datastructures import FileStorage

from app.flask_app import _save_csv_uploads, create_app
from app.run_tracking import RunTrackingStore


def _app(tmp_path: Path):
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "AUTH_DATABASE_URL": f"sqlite:///{(tmp_path / 'web.db').as_posix()}",
            "AUTH_PASSWORD_HASH_ITERATIONS": 120_000,
        }
    )


def _csrf(client) -> str:
    with client.session_transaction() as session:
        return str(session["csrf_token"])


def _signup_and_login(client) -> None:
    client.get("/auth?mode=signup")
    response = client.post(
        "/auth?mode=signup",
        data={
            "csrf_token": _csrf(client),
            "mode": "signup",
            "full_name": "Data Analyst",
            "email": "analyst@example.com",
            "password": "StrongPass1!",
            "confirm_password": "StrongPass1!",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Account created" in response.data
    response = client.post(
        "/auth",
        data={
            "csrf_token": _csrf(client),
            "mode": "login",
            "email": "analyst@example.com",
            "password": "StrongPass1!",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Start a new analysis" in response.data


def test_auth_and_dashboard_flow(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client = app.test_client()

    response = client.get("/")
    assert response.status_code == 302
    assert "/auth" in response.headers["Location"]

    _signup_and_login(client)
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Good data. Clear answers." in response.data
    assert b"Add more CSV" in response.data
    assert b"Data preview" in response.data


def test_multiple_csv_uploads_are_combined_by_rows(tmp_path: Path) -> None:
    first = FileStorage(stream=BytesIO(b"id,value\n1,10\n2,20\n"), filename="first.csv")
    second = FileStorage(stream=BytesIO(b"value,id\n30,3\n"), filename="second.csv")

    combined_path, filenames = _save_csv_uploads([first, second], tmp_path / "uploads")

    combined = pd.read_csv(combined_path)
    assert filenames == ["first.csv", "second.csv"]
    assert combined.columns.tolist() == ["id", "value"]
    assert combined.to_dict("records") == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30},
    ]


def test_multiple_csv_uploads_require_compatible_columns(tmp_path: Path) -> None:
    first = FileStorage(stream=BytesIO(b"id,value\n1,10\n"), filename="first.csv")
    incompatible = FileStorage(
        stream=BytesIO(b"id,revenue\n2,20\n"), filename="other.csv"
    )

    with pytest.raises(ValueError, match="does not match the first CSV schema"):
        _save_csv_uploads([first, incompatible], tmp_path / "uploads")


def test_run_detail_and_artifact_are_user_scoped(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client = app.test_client()
    _signup_and_login(client)

    with client.session_transaction() as session:
        user_id = int(session["user_id"])
    store = app.extensions["run_store"]
    assert isinstance(store, RunTrackingStore)
    run_dir = tmp_path / "run_demo"
    run_dir.mkdir()
    report = run_dir / "final_report.md"
    report.write_text("# Reliable result\n\nEvidence-backed finding.", encoding="utf-8")
    run = store.create_run(
        user_id=user_id,
        source_type="csv",
        source_name="demo.csv",
        run_metadata={"business_question": "What changed?", "run_dir": str(run_dir)},
    )
    store.attach_artifact_from_file(run_id=run.id, run_dir=run_dir, artifact_file=report)
    run = store.mark_completed(run_id=run.id, report_path="final_report.md")

    response = client.get(f"/runs/{run.run_uuid}")
    assert response.status_code == 200
    assert b"Reliable result" in response.data
    response = client.get(f"/runs/{run.run_uuid}/artifacts/final_report.md")
    assert response.status_code == 200
