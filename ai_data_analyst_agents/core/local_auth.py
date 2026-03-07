from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import base64
import hashlib
import hmac
import os
import re
import sqlite3


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_email(email: str) -> bool:
    if not email:
        return False
    # Practical email sanity check; avoids overfitting to strict RFC edge cases.
    return bool(re.match(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", email.strip()))


def validate_password_strength(password: str) -> list[str]:
    issues: list[str] = []
    if len(password or "") < 12:
        issues.append("Password must be at least 12 characters long.")
    if not re.search(r"[A-Z]", password or ""):
        issues.append("Password must include at least one uppercase letter.")
    if not re.search(r"[a-z]", password or ""):
        issues.append("Password must include at least one lowercase letter.")
    if not re.search(r"[0-9]", password or ""):
        issues.append("Password must include at least one number.")
    if not re.search(r"[^A-Za-z0-9]", password or ""):
        issues.append("Password must include at least one special character.")
    return issues


def _hash_password(password: str, *, salt: bytes, iterations: int, pepper: str) -> bytes:
    raw = (pepper + password).encode("utf-8")
    return hashlib.pbkdf2_hmac("sha256", raw, salt, iterations)


@dataclass(frozen=True)
class AuthUser:
    id: int
    email: str
    full_name: str
    created_at: str
    last_login_at: str | None


class LocalAuthStore:
    def __init__(
        self,
        db_path: str | Path,
        *,
        pepper: str | None = None,
        iterations: int = 390_000,
        lock_after_failures: int = 5,
        lock_minutes: int = 15,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pepper = pepper if pepper is not None else os.getenv("AUTH_PASSWORD_PEPPER", "")
        self.iterations = max(100_000, int(iterations))
        self.lock_after_failures = max(3, int(lock_after_failures))
        self.lock_minutes = max(1, int(lock_minutes))
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=3000;")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    full_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    password_iterations INTEGER NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until TEXT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            conn.commit()

    def create_user(self, *, email: str, full_name: str, password: str) -> tuple[bool, str]:
        email_n = normalize_email(email)
        full_name_n = (full_name or "").strip()
        if not validate_email(email_n):
            return False, "Enter a valid email address."
        if len(full_name_n) < 2:
            return False, "Full name must be at least 2 characters."
        issues = validate_password_strength(password)
        if issues:
            return False, " ".join(issues)

        salt = os.urandom(16)
        digest = _hash_password(
            password=password,
            salt=salt,
            iterations=self.iterations,
            pepper=self.pepper,
        )
        salt_b64 = base64.b64encode(salt).decode("ascii")
        hash_b64 = base64.b64encode(digest).decode("ascii")
        now = _iso(_utc_now())
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO users(
                        email, full_name, password_hash, password_salt, password_iterations,
                        is_active, failed_attempts, locked_until, created_at, updated_at, last_login_at
                    )
                    VALUES (?, ?, ?, ?, ?, 1, 0, NULL, ?, ?, NULL)
                    """,
                    (email_n, full_name_n, hash_b64, salt_b64, self.iterations, now, now),
                )
                conn.commit()
                return True, "Account created."
        except sqlite3.IntegrityError:
            return False, "An account with this email already exists."

    def _dummy_verify_cost(self, password: str) -> None:
        # Keep timing closer for non-existing emails.
        salt = b"\x00" * 16
        _hash_password(password=password or "", salt=salt, iterations=self.iterations, pepper=self.pepper)

    def authenticate(self, *, email: str, password: str) -> tuple[Optional[AuthUser], str]:
        email_n = normalize_email(email)
        if not validate_email(email_n):
            self._dummy_verify_cost(password)
            return None, "Invalid credentials."

        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ? LIMIT 1",
                (email_n,),
            ).fetchone()
            if row is None:
                self._dummy_verify_cost(password)
                return None, "Invalid credentials."

            if int(row["is_active"]) != 1:
                return None, "Account is disabled."

            now = _utc_now()
            locked_until = _parse_iso(row["locked_until"])
            if locked_until and locked_until > now:
                remaining = int((locked_until - now).total_seconds() // 60) + 1
                return None, f"Too many failed attempts. Try again in {remaining} minute(s)."

            try:
                salt = base64.b64decode(str(row["password_salt"]).encode("ascii"))
                expected = base64.b64decode(str(row["password_hash"]).encode("ascii"))
                iters = int(row["password_iterations"])
            except Exception:
                return None, "Account configuration error. Reset this account."

            candidate = _hash_password(password=password, salt=salt, iterations=iters, pepper=self.pepper)
            if not hmac.compare_digest(expected, candidate):
                failures = int(row["failed_attempts"]) + 1
                lock_until = None
                if failures >= self.lock_after_failures:
                    lock_until = _iso(now + timedelta(minutes=self.lock_minutes))
                    failures = 0
                conn.execute(
                    """
                    UPDATE users
                    SET failed_attempts = ?, locked_until = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (failures, lock_until, _iso(now), int(row["id"])),
                )
                conn.commit()
                return None, "Invalid credentials."

            conn.execute(
                """
                UPDATE users
                SET failed_attempts = 0, locked_until = NULL, last_login_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (_iso(now), _iso(now), int(row["id"])),
            )
            conn.commit()

            return (
                AuthUser(
                    id=int(row["id"]),
                    email=str(row["email"]),
                    full_name=str(row["full_name"]),
                    created_at=str(row["created_at"]),
                    last_login_at=str(row["last_login_at"]) if row["last_login_at"] else None,
                ),
                "Authenticated.",
            )

    def get_user_by_id(self, user_id: int) -> Optional[AuthUser]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, full_name, created_at, last_login_at FROM users WHERE id = ? LIMIT 1",
                (int(user_id),),
            ).fetchone()
            if row is None:
                return None
            return AuthUser(
                id=int(row["id"]),
                email=str(row["email"]),
                full_name=str(row["full_name"]),
                created_at=str(row["created_at"]),
                last_login_at=str(row["last_login_at"]) if row["last_login_at"] else None,
            )
