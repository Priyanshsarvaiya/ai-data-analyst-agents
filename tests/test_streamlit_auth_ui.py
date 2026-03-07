from __future__ import annotations

from pathlib import Path

import pytest


def _streamlit_source() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "app" / "streamlit_app.py").read_text(encoding="utf-8")


def test_auth_ui_removes_verbose_info_blocks() -> None:
    src = _streamlit_source()
    assert "Before you start" not in src
    assert "Security defaults" not in src
    assert "stat-chip" not in src
    assert "I understand this uses my configured PostgreSQL auth store." not in src


def test_auth_ui_keeps_minimal_form_copy() -> None:
    src = _streamlit_source()
    assert "Secure Workspace Access" in src
    assert "Sign in or create an account to continue." in src
    assert "Use 12+ chars with upper, lower, number, and special character." in src


def test_streamlit_auth_store_errors_when_database_url_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.streamlit_app as streamlit_app

    streamlit_app._get_auth_store.clear()
    streamlit_app._get_auth_cfg.clear()

    class _AuthCfg:
        resolved_database_url = ""
        AUTH_PASSWORD_PEPPER = ""
        AUTH_PASSWORD_HASH_ITERATIONS = 390000
        AUTH_LOCKOUT_ATTEMPTS = 5
        AUTH_LOCKOUT_MINUTES = 15

    monkeypatch.setattr(streamlit_app, "_get_auth_cfg", lambda: _AuthCfg())

    with pytest.raises(RuntimeError, match="AUTH_DATABASE_URL/DATABASE_URL is not configured"):
        streamlit_app._get_auth_store()

