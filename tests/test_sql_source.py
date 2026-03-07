from __future__ import annotations

import pytest

from ai_data_analyst_agents.core.sql_source import (
    SQLDataSource,
    build_groupby_query,
    choose_primary_table,
    column_table_index,
    compute_join_profile,
    find_join_path,
)


def test_sql_schema_discovery_and_primary_table(sqlite_star_db) -> None:
    src = SQLDataSource(db_url=f"sqlite:///{sqlite_star_db}")
    schema = src.inspect_schema(include_row_counts=True)

    table_names = {t["name"] for t in schema["tables"]}
    assert {"orders", "customers"}.issubset(table_names)
    assert choose_primary_table(schema) == "orders"

    col_idx = column_table_index(schema)
    assert "revenue" in col_idx and "orders" in col_idx["revenue"]
    assert "country" in col_idx and "customers" in col_idx["country"]


def test_join_path_groupby_query_and_profile(sqlite_star_db) -> None:
    src = SQLDataSource(db_url=f"sqlite:///{sqlite_star_db}")
    schema = src.inspect_schema(include_row_counts=True)

    path = find_join_path(schema, "orders", "customers")
    assert path == ["orders", "customers"]

    built = build_groupby_query(
        engine=src.engine,
        schema=schema,
        metric_col="revenue",
        group_cols=["country"],
        agg="sum",
        preferred_fact_table="orders",
        limit=1000,
    )
    assert "LEFT JOIN" in built["query"]
    assert "customers" in built["query"]

    out_df = src.execute_query(built["query"])
    assert "value" in out_df.columns
    assert len(out_df) > 0

    profile = compute_join_profile(
        engine=src.engine,
        schema=schema,
        fact_table="orders",
        dimension_table="customers",
    )
    assert profile["status"] == "ok"
    assert profile["join_path"] == ["orders", "customers"]
    assert profile["joined_rows"] == profile["fact_rows"]


def test_build_groupby_query_rejects_unknown_columns(sqlite_star_db) -> None:
    src = SQLDataSource(db_url=f"sqlite:///{sqlite_star_db}")
    schema = src.inspect_schema(include_row_counts=False)

    with pytest.raises(ValueError, match="Metric column"):
        build_groupby_query(
            engine=src.engine,
            schema=schema,
            metric_col="unknown_metric",
            group_cols=["country"],
        )

    with pytest.raises(ValueError, match="Group column"):
        build_groupby_query(
            engine=src.engine,
            schema=schema,
            metric_col="revenue",
            group_cols=["not_a_col"],
        )


def test_sql_datasource_accepts_postgres_url_without_connecting() -> None:
    src = SQLDataSource(db_url="postgresql+psycopg://user:pass@localhost:5432/analytics")
    assert src.engine is not None
    assert src.engine.url.get_backend_name().startswith("postgresql")
