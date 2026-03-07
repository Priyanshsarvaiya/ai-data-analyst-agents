from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import hmac
import os
import re

from sqlalchemy import Boolean, DateTime, Integer, String, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from pydantic_settings import BaseSettings, SettingsConfigDict

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class AuthSettings(BaseSettings):
    AUTH_DATABASE_URL: str = ""
    DATABASE_URL: str = ""
    AUTH_PASSWORD_PEPPER: str = ""
    AUTH_PASSWORD_HASH_ITERATIONS: int = 390_000
    AUTH_LOCKOUT_ATTEMPTS: int = 5
    AUTH_LOCKOUT_MINUTES: int = 15
    AUTH_SESSION_TTL_MIN: int = 240
    MAX_UPLOAD_MB: int = 50

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def resolved_database_url(self) -> str:
        return (self.AUTH_DATABASE_URL or self.DATABASE_URL or "").strip()


def load_auth_settings() -> AuthSettings:
    return AuthSettings()


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


def _is_email_uniquely_constrained(engine) -> bool:
    inspector = inspect(engine)
    try:
        for c in inspector.get_unique_constraints("users"):
            cols = [str(x).lower() for x in (c.get("column_names") or [])]
            if cols == ["email"] or "email" in cols:
                return True
    except Exception:
        pass
    try:
        for idx in inspector.get_indexes("users"):
            cols = [str(x).lower() for x in (idx.get("column_names") or [])]
            if idx.get("unique") and (cols == ["email"] or "email" in cols):
                return True
    except Exception:
        pass
    return False


def ensure_users_identity_constraints(engine) -> None:
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return

    with engine.begin() as conn:
        # Safe migration path for existing databases where constraints may be missing.
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_users_email ON users (email)"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ux_users_id ON users (id)"))

        if engine.dialect.name == "postgresql":
            conn.execute(text("ALTER TABLE users ALTER COLUMN id SET NOT NULL"))
            conn.execute(text("ALTER TABLE users ALTER COLUMN email SET NOT NULL"))
            conn.execute(
                text(
                    """
                    DO $$
                    BEGIN
                      IF NOT EXISTS (
                        SELECT 1
                        FROM pg_constraint
                        WHERE conrelid = 'users'::regclass
                          AND contype = 'p'
                      ) THEN
                        ALTER TABLE users ADD CONSTRAINT users_pkey PRIMARY KEY (id);
                      END IF;
                    END $$;
                    """
                )
            )

    inspector = inspect(engine)
    pk = inspector.get_pk_constraint("users") or {}
    pk_cols = [str(c).lower() for c in (pk.get("constrained_columns") or [])]
    if "id" not in pk_cols:
        raise RuntimeError(
            "users.id is not configured as the primary key. "
            "Please clean data (null/duplicate ids) and apply the users primary key migration."
        )
    if not _is_email_uniquely_constrained(engine):
        raise RuntimeError(
            "users.email is not uniquely constrained. "
            "Please resolve duplicate emails and apply a unique email constraint."
        )


def _pbkdf2_hash(password: str, *, salt: bytes, iterations: int, pepper: str) -> bytes:
    raw = (pepper + password).encode("utf-8")
    return hashlib.pbkdf2_hmac("sha256", raw, salt, iterations)


def _encode_hash(password: str, *, pepper: str, iterations: int) -> str:
    salt = os.urandom(16)
    digest = _pbkdf2_hash(password, salt=salt, iterations=iterations, pepper=pepper)
    return "pbkdf2_sha256${}${}${}".format(
        iterations,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def _verify_hash(password: str, encoded: str, *, pepper: str) -> bool:
    try:
        algo, i_str, salt_b64, hash_b64 = encoded.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(i_str)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(hash_b64.encode("ascii"))
    except Exception:
        return False
    candidate = _pbkdf2_hash(password, salt=salt, iterations=iterations, pepper=pepper)
    return hmac.compare_digest(expected, candidate)


class Base(DeclarativeBase):
    pass


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(1024), nullable=False)

    failed_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


@dataclass(frozen=True)
class AuthUser:
    id: int
    email: str
    full_name: str
    created_at: str
    last_login_at: str | None


class PostgresAuthStore:
    def __init__(
        self,
        *,
        database_url: str,
        pepper: str = "",
        iterations: int = 390_000,
        lock_after_failures: int = 5,
        lock_minutes: int = 15,
    ) -> None:
        if not database_url:
            raise RuntimeError("AUTH_DATABASE_URL or DATABASE_URL is required for Postgres auth store.")
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.session_local = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False)
        self.pepper = pepper
        self.iterations = max(120_000, int(iterations))
        self.lock_after_failures = max(3, int(lock_after_failures))
        self.lock_minutes = max(1, int(lock_minutes))
        Base.metadata.create_all(bind=self.engine)
        ensure_users_identity_constraints(self.engine)

    def _to_user(self, u: UserModel) -> AuthUser:
        created_at = _ensure_utc(u.created_at)
        last_login_at = _ensure_utc(u.last_login_at)
        return AuthUser(
            id=int(u.id),
            email=str(u.email),
            full_name=str(u.full_name),
            created_at=created_at.isoformat() if created_at else "",
            last_login_at=last_login_at.isoformat() if last_login_at else None,
        )

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

        with self.session_local() as db:
            exists = db.execute(select(UserModel).where(UserModel.email == email_n)).scalar_one_or_none()
            if exists is not None:
                return False, "An account with this email already exists."

            now = _utc_now()
            user = UserModel(
                email=email_n,
                full_name=full_name_n,
                password_hash=_encode_hash(password, pepper=self.pepper, iterations=self.iterations),
                failed_attempts=0,
                locked_until=None,
                is_active=True,
                created_at=now,
                updated_at=now,
                last_login_at=None,
            )
            db.add(user)
            db.commit()
            return True, "Account created."

    def authenticate(self, *, email: str, password: str) -> tuple[AuthUser | None, str]:
        email_n = normalize_email(email)
        if not validate_email(email_n):
            return None, "Invalid credentials."

        with self.session_local() as db:
            user = db.execute(select(UserModel).where(UserModel.email == email_n)).scalar_one_or_none()
            if user is None:
                return None, "Invalid credentials."
            if not user.is_active:
                return None, "Account is disabled."

            now = _utc_now()
            locked_until = _ensure_utc(user.locked_until)
            if locked_until and locked_until > now:
                remaining = int((locked_until - now).total_seconds() // 60) + 1
                return None, f"Too many failed attempts. Try again in {remaining} minute(s)."

            if not _verify_hash(password, user.password_hash, pepper=self.pepper):
                user.failed_attempts = int(user.failed_attempts or 0) + 1
                if user.failed_attempts >= self.lock_after_failures:
                    user.failed_attempts = 0
                    user.locked_until = now + timedelta(minutes=self.lock_minutes)
                user.updated_at = now
                db.add(user)
                db.commit()
                return None, "Invalid credentials."

            user.failed_attempts = 0
            user.locked_until = None
            user.last_login_at = now
            user.updated_at = now
            db.add(user)
            db.commit()
            db.refresh(user)
            return self._to_user(user), "Authenticated."

    def get_user_by_id(self, user_id: int) -> AuthUser | None:
        with self.session_local() as db:
            user = db.execute(select(UserModel).where(UserModel.id == int(user_id))).scalar_one_or_none()
            if user is None:
                return None
            return self._to_user(user)
