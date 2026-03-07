from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sqlite3
from typing import Iterator

import pandas as pd
import pytest

from ai_data_analyst_agents.core.settings import load_app_cfg


class DummyOpenRouterClient:
    def __init__(self, timeout_s: int = 60) -> None:  # noqa: ARG002
        pass

    def chat(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
        return ""


@pytest.fixture(autouse=True)
def _matplotlib_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    mpl_dir = tmp_path / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_dir))
    yield


@pytest.fixture
def sample_df() -> pd.DataFrame:
    countries = ["India", "Germany", "USA", "Canada", "UK"]
    categories = ["Electronics", "Books", "Home"]
    rows = []
    for i in range(30):
        qty = (i % 4) + 1
        unit_price = 100 + (i % 7) * 10
        discount_pct = [0.0, 0.05, 0.1][i % 3]
        rows.append(
            {
                "order_id": f"O{i+1:04d}",
                "order_date": (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                "customer_id": f"CUST{(i % 10) + 1:03d}",
                "country": countries[i % len(countries)],
                "product_category": categories[i % len(categories)],
                "quantity": qty,
                "unit_price": float(unit_price),
                "discount_pct": float(discount_pct),
                "revenue": float(qty * unit_price * (1.0 - discount_pct)),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_df_with_duplicate(sample_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)


@pytest.fixture
def sample_csv_path(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    p = tmp_path / "sample.csv"
    sample_df.to_csv(p, index=False)
    return p


@pytest.fixture
def sqlite_orders_db(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    db_path = tmp_path / "orders.db"
    con = sqlite3.connect(db_path)
    try:
        sample_df.to_sql("orders", con, index=False, if_exists="replace")
    finally:
        con.close()
    return db_path


@pytest.fixture
def sqlite_star_db(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    db_path = tmp_path / "star.db"
    customers = (
        sample_df.sort_values(["customer_id", "order_date"])
        .groupby("customer_id", as_index=False)
        .agg(country=("country", "first"))
    )
    orders = sample_df[
        ["order_id", "order_date", "customer_id", "product_category", "quantity", "unit_price", "discount_pct", "revenue"]
    ].copy()

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        cur.execute("DROP TABLE IF EXISTS orders;")
        cur.execute("DROP TABLE IF EXISTS customers;")
        cur.execute(
            """
            CREATE TABLE customers (
                customer_id TEXT PRIMARY KEY,
                country TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                order_date TEXT,
                customer_id TEXT,
                product_category TEXT,
                quantity INTEGER,
                unit_price REAL,
                discount_pct REAL,
                revenue REAL,
                FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
            );
            """
        )
        cur.executemany(
            "INSERT INTO customers(customer_id, country) VALUES (?, ?);",
            list(customers.itertuples(index=False, name=None)),
        )
        cur.executemany(
            """
            INSERT INTO orders(order_id, order_date, customer_id, product_category, quantity, unit_price, discount_pct, revenue)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            list(orders.itertuples(index=False, name=None)),
        )
        con.commit()
    finally:
        con.close()
    return db_path


@pytest.fixture
def patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    import ai_data_analyst_agents.agents.planner as planner_mod
    import ai_data_analyst_agents.agents.reporting as reporting_mod

    monkeypatch.setattr(planner_mod, "OpenRouterClient", DummyOpenRouterClient)
    monkeypatch.setattr(reporting_mod, "OpenRouterClient", DummyOpenRouterClient)


@pytest.fixture
def patch_pipeline_cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    import ai_data_analyst_agents.pipelines.run_csv_pipeline as csv_pipe
    import ai_data_analyst_agents.pipelines.run_sql_pipeline as sql_pipe

    base_cfg = load_app_cfg()
    artifacts_root = tmp_path / "artifacts"

    def _cfg():
        cfg = deepcopy(base_cfg)
        cfg.runtime.artifacts_dir = str(artifacts_root)
        cfg.runtime.log_level = "INFO"
        return cfg

    monkeypatch.setattr(csv_pipe, "load_app_cfg", _cfg)
    monkeypatch.setattr(sql_pipe, "load_app_cfg", _cfg)
    return artifacts_root


def latest_run_dir(artifacts_root: Path) -> Path:
    run_dirs = sorted([p for p in artifacts_root.glob("run_*") if p.is_dir()])
    if not run_dirs:
        raise AssertionError(f"No run directory found under {artifacts_root}")
    return run_dirs[-1]
