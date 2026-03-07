from __future__ import annotations

import pytest

from ai_data_analyst_agents.core.security import (
    clamp_positive_limit,
    redact_payload_for_llm,
    sanitize_user_error_message,
    validate_read_only_sql,
)
from ai_data_analyst_agents.core.sql_source import SQLDataSource


def test_validate_read_only_sql_accepts_select_and_cte() -> None:
    assert validate_read_only_sql("SELECT 1 AS x;") == "SELECT 1 AS x"
    q = validate_read_only_sql("WITH t AS (SELECT 1 AS x) SELECT x FROM t")
    assert q.lower().startswith("with ")


def test_validate_read_only_sql_rejects_writes_and_multi_statement() -> None:
    with pytest.raises(ValueError):
        validate_read_only_sql("DELETE FROM orders")
    with pytest.raises(ValueError):
        validate_read_only_sql("SELECT 1; DROP TABLE orders")
    with pytest.raises(ValueError):
        validate_read_only_sql("SELECT * FROM orders FOR UPDATE")


def test_sql_datasource_rejects_non_read_only_query(sqlite_star_db) -> None:
    src = SQLDataSource(db_url=f"sqlite:///{sqlite_star_db}", enforce_read_only_sql=True)
    with pytest.raises(ValueError):
        src.execute_query("DROP TABLE customers")


def test_sql_datasource_clamps_query_limit(sqlite_star_db) -> None:
    src = SQLDataSource(db_url=f"sqlite:///{sqlite_star_db}", max_query_rows=7)
    out = src.execute_query("SELECT * FROM orders", limit=9999)
    assert len(out) == 7
    out2 = src.execute_query("SELECT * FROM orders", limit=0)
    assert len(out2) == 7


def test_clamp_positive_limit_defaults_to_max() -> None:
    assert clamp_positive_limit(None, max_limit=100) == 100
    assert clamp_positive_limit(-5, max_limit=100) == 100
    assert clamp_positive_limit(7, max_limit=100) == 7
    assert clamp_positive_limit(700, max_limit=100) == 100


def test_redact_payload_for_llm_hides_raw_rows_by_default() -> None:
    payload = {
        "columns": ["customer_email"],
        "rows": [{"customer_email": "alice@example.com"}, {"customer_email": "bob@example.com"}],
        "n_rows": 2,
    }
    redacted = redact_payload_for_llm(payload, allow_raw_rows=False, max_rows=5)
    assert redacted["rows"] == "REDACTED(2 rows)"
    assert redacted["row_count"] == 2
    assert "alice@example.com" not in str(redacted)


def test_redact_payload_for_llm_allows_truncated_rows_when_enabled() -> None:
    payload = {"rows": [{"x": 1}, {"x": 2}, {"x": 3}]}
    out = redact_payload_for_llm(payload, allow_raw_rows=True, max_rows=2)
    assert isinstance(out["rows"], list)
    assert len(out["rows"]) == 2
    assert out["rows_truncated"] == "1 rows omitted"


def test_sanitize_user_error_message_redacts_credentials_and_tokens() -> None:
    msg = (
        "DB failed postgresql+psycopg://user:supersecret@localhost:5432/db "
        "OPENROUTER_API_KEY=sk-abcdef123456"
    )
    out = sanitize_user_error_message(msg, max_chars=200)
    assert "supersecret" not in out
    assert "sk-abcdef123456" not in out
    assert "***" in out
