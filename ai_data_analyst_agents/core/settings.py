from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class RuntimeCfg(BaseModel):
    artifacts_dir: str = "artifacts"
    log_level: str = "INFO"
    max_rows_profile: int = 50000


class LlmCfg(BaseModel):
    provider: str = "openrouter"
    model: str = "z-ai/glm-5"
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout_s: int = 60


class QACfg(BaseModel):
    missingness_warn_threshold: float = 0.2
    duplicate_warn_threshold: float = 0.01
    outlier_z_threshold: float = 4.0


class EDACfg(BaseModel):
    max_unique_for_category: int = 30
    max_plots: int = 12


class SQLCfg(BaseModel):
    default_query_row_limit: int = 50000
    introspection_max_tables: int = 200
    preview_rows: int = 10000
    include_row_counts: bool = True


class SecurityCfg(BaseModel):
    enforce_read_only_sql: bool = True
    allow_raw_rows_to_llm: bool = False
    max_rows_to_llm: int = 25
    user_error_max_chars: int = 400


class AppCfg(BaseModel):
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)
    llm: LlmCfg = Field(default_factory=LlmCfg)
    qa: QACfg = Field(default_factory=QACfg)
    eda: EDACfg = Field(default_factory=EDACfg)
    sql: SQLCfg = Field(default_factory=SQLCfg)
    security: SecurityCfg = Field(default_factory=SecurityCfg)


class EnvSettings(BaseSettings):
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "z-ai/glm-5"
    OPENROUTER_SITE_URL: Optional[str] = None
    OPENROUTER_APP_NAME: Optional[str] = None

    ENV: str = "local"
    ARTIFACTS_DIR: str = "artifacts"
    LOG_LEVEL: str = "INFO"
    SQL_DEFAULT_QUERY_ROW_LIMIT: Optional[int] = None
    SQL_INTROSPECTION_MAX_TABLES: Optional[int] = None
    SQL_PREVIEW_ROWS: Optional[int] = None
    SECURITY_ENFORCE_READ_ONLY_SQL: Optional[bool] = None
    SECURITY_ALLOW_RAW_ROWS_TO_LLM: Optional[bool] = None
    SECURITY_MAX_ROWS_TO_LLM: Optional[int] = None
    SECURITY_USER_ERROR_MAX_CHARS: Optional[int] = None

    class Config:
        env_file = ".env"
        extra = "ignore"


def load_app_cfg(path: str | Path = "configs/settings.yaml") -> AppCfg:
    p = Path(path)
    data: dict[str, Any] = {}
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}

    cfg = AppCfg.model_validate(data)

    env = EnvSettings()
    cfg.runtime.artifacts_dir = env.ARTIFACTS_DIR or cfg.runtime.artifacts_dir
    cfg.runtime.log_level = env.LOG_LEVEL or cfg.runtime.log_level
    cfg.llm.model = env.OPENROUTER_MODEL or cfg.llm.model
    if env.SQL_DEFAULT_QUERY_ROW_LIMIT is not None:
        cfg.sql.default_query_row_limit = int(env.SQL_DEFAULT_QUERY_ROW_LIMIT)
    if env.SQL_INTROSPECTION_MAX_TABLES is not None:
        cfg.sql.introspection_max_tables = int(env.SQL_INTROSPECTION_MAX_TABLES)
    if env.SQL_PREVIEW_ROWS is not None:
        cfg.sql.preview_rows = int(env.SQL_PREVIEW_ROWS)
    if env.SECURITY_ENFORCE_READ_ONLY_SQL is not None:
        cfg.security.enforce_read_only_sql = bool(env.SECURITY_ENFORCE_READ_ONLY_SQL)
    if env.SECURITY_ALLOW_RAW_ROWS_TO_LLM is not None:
        cfg.security.allow_raw_rows_to_llm = bool(env.SECURITY_ALLOW_RAW_ROWS_TO_LLM)
    if env.SECURITY_MAX_ROWS_TO_LLM is not None:
        cfg.security.max_rows_to_llm = int(env.SECURITY_MAX_ROWS_TO_LLM)
    if env.SECURITY_USER_ERROR_MAX_CHARS is not None:
        cfg.security.user_error_max_chars = int(env.SECURITY_USER_ERROR_MAX_CHARS)
    return cfg
