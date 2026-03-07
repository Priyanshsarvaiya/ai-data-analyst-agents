from __future__ import annotations

from typing import Any
import re


_SQL_DENY_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "grant",
    "revoke",
    "merge",
    "call",
    "exec",
    "execute",
    "copy",
    "vacuum",
    "analyze",
    "attach",
    "detach",
    "pragma",
    "set",
    "begin",
    "commit",
    "rollback",
}


def _strip_sql_comments(sql: str) -> str:
    s = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    s = re.sub(r"--[^\n\r]*", " ", s)
    return s


def validate_read_only_sql(query: str) -> str:
    q = _strip_sql_comments((query or "")).strip()
    if not q:
        raise ValueError("Empty SQL query.")
    q = q.rstrip(";").strip()
    if not q:
        raise ValueError("Empty SQL query.")

    # Keep execution single-statement.
    if ";" in q:
        raise ValueError("Only single-statement read-only SQL is allowed.")

    q_lc = q.lower()
    if not (q_lc.startswith("select") or q_lc.startswith("with ")):
        raise ValueError("Only SELECT/CTE read-only SQL is allowed.")

    for kw in _SQL_DENY_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", q_lc):
            raise ValueError(f"Blocked SQL keyword in read-only mode: {kw}")

    if re.search(r"\bfor\s+update\b", q_lc):
        raise ValueError("FOR UPDATE is blocked in read-only mode.")
    if re.search(r"\bselect\b[\s\S]*\binto\b", q_lc):
        raise ValueError("SELECT INTO is blocked in read-only mode.")

    return q


def clamp_positive_limit(limit: int | None, *, max_limit: int) -> int:
    safe_max = max(1, int(max_limit))
    if limit is None:
        return safe_max
    try:
        lim = int(limit)
    except Exception:
        return safe_max
    if lim <= 0:
        return safe_max
    return min(lim, safe_max)


def redact_payload_for_llm(
    payload: Any,
    *,
    allow_raw_rows: bool = False,
    max_rows: int = 25,
) -> Any:
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key, val in payload.items():
            if key == "rows" and isinstance(val, list):
                if allow_raw_rows:
                    limit = max(1, int(max_rows))
                    out["rows"] = [redact_payload_for_llm(v, allow_raw_rows=allow_raw_rows, max_rows=max_rows) for v in val[:limit]]
                    if len(val) > limit:
                        out["rows_truncated"] = f"{len(val) - limit} rows omitted"
                else:
                    out["rows"] = f"REDACTED({len(val)} rows)"
                    out["row_count"] = len(val)
                continue
            out[str(key)] = redact_payload_for_llm(val, allow_raw_rows=allow_raw_rows, max_rows=max_rows)
        return out
    if isinstance(payload, list):
        limit = max(1, int(max_rows))
        out_list = [redact_payload_for_llm(v, allow_raw_rows=allow_raw_rows, max_rows=max_rows) for v in payload[:limit]]
        if len(payload) > limit:
            out_list.append(f"... truncated {len(payload) - limit} items")
        return out_list
    return payload


def redact_secrets(text: str) -> str:
    out = text or ""
    patterns = [
        (r"://([^:/?#\s]+):([^@/\s]+)@", r"://\1:***@"),
        (r"(authorization\s*:\s*bearer\s+)[^\s]+", r"\1***"),
        (r"(openrouter_api_key\s*[=:]\s*)[^\s]+", r"\1***"),
        (r"(api[_-]?key\s*[=:]\s*)[^\s]+", r"\1***"),
    ]
    for pat, rep in patterns:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


def sanitize_user_error_message(err: Exception | str, *, max_chars: int = 400) -> str:
    msg = redact_secrets(str(err or "Unexpected error"))
    msg = " ".join(msg.split())
    if not msg:
        msg = "Unexpected error"
    limit = max(80, int(max_chars))
    if len(msg) > limit:
        msg = msg[:limit].rstrip() + "..."
    return msg
