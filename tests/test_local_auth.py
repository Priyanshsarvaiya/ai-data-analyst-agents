from __future__ import annotations

import pytest

from app.postgres_auth import (
    PostgresAuthStore,
    normalize_email,
    validate_email,
    validate_password_strength,
)


def test_email_normalization_and_validation() -> None:
    assert normalize_email("  USER@Example.COM ") == "user@example.com"
    assert validate_email("person@example.com")
    assert not validate_email("bad-email")


def test_password_strength_rules() -> None:
    issues = validate_password_strength("weak")
    assert len(issues) >= 3
    assert validate_password_strength("StrongPass1!") == []


def test_postgres_auth_create_and_authenticate(tmp_path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'auth.db').as_posix()}"
    auth = PostgresAuthStore(database_url=db_url, iterations=120_000)
    ok, _ = auth.create_user(
        email="analyst@example.com",
        full_name="Data Analyst",
        password="StrongPass1!",
    )
    assert ok

    user, msg = auth.authenticate(email="analyst@example.com", password="StrongPass1!")
    assert user is not None
    assert user.email == "analyst@example.com"
    assert "Authenticated" in msg


def test_postgres_auth_rejects_duplicate_email(tmp_path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'auth.db').as_posix()}"
    auth = PostgresAuthStore(database_url=db_url, iterations=120_000)
    ok1, _ = auth.create_user(
        email="analyst@example.com",
        full_name="Analyst One",
        password="StrongPass1!",
    )
    ok2, msg2 = auth.create_user(
        email="ANALYST@example.com",
        full_name="Analyst Two",
        password="StrongPass1!",
    )
    assert ok1
    assert not ok2
    assert "already exists" in msg2


def test_postgres_auth_rejects_weak_password(tmp_path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'auth.db').as_posix()}"
    auth = PostgresAuthStore(database_url=db_url, iterations=120_000)
    ok, msg = auth.create_user(
        email="analyst@example.com",
        full_name="Analyst One",
        password="weak",
    )
    assert not ok
    assert "Password must" in msg


def test_postgres_auth_lockout_after_failed_attempts(tmp_path) -> None:
    db_url = f"sqlite:///{(tmp_path / 'auth.db').as_posix()}"
    auth = PostgresAuthStore(
        database_url=db_url,
        iterations=120_000,
        lock_after_failures=3,
        lock_minutes=1,
    )
    ok, _ = auth.create_user(
        email="analyst@example.com",
        full_name="Data Analyst",
        password="StrongPass1!",
    )
    assert ok

    for _ in range(3):
        user, _ = auth.authenticate(email="analyst@example.com", password="wrong-pass")
        assert user is None

    user, msg = auth.authenticate(email="analyst@example.com", password="StrongPass1!")
    assert user is None
    assert "Too many failed attempts" in msg


def test_postgres_auth_requires_database_url() -> None:
    with pytest.raises(RuntimeError, match="AUTH_DATABASE_URL or DATABASE_URL"):
        PostgresAuthStore(database_url="")
