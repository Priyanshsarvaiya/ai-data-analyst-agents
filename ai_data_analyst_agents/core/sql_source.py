from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from collections import deque
import re

import pandas as pd

try:
    from sqlalchemy import create_engine, inspect, text
    from sqlalchemy.engine import Engine
except Exception:  # pragma: no cover - optional dependency path
    create_engine = None
    inspect = None
    text = None
    Engine = Any  # type: ignore[misc,assignment]


def _quote_ident(engine: Engine, name: str) -> str:
    return engine.dialect.identifier_preparer.quote(name)


def _quote_table(engine: Engine, table_name: str, schema_name: str | None = None) -> str:
    q_table = _quote_ident(engine, table_name)
    if not schema_name:
        return q_table
    return f"{_quote_ident(engine, schema_name)}.{q_table}"


def _sanitize_alias(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    if not safe:
        return "col"
    if safe[0].isdigit():
        safe = f"c_{safe}"
    return safe


@dataclass
class SQLDataSource:
    db_url: str
    timeout_s: int = 30

    def __post_init__(self) -> None:
        if create_engine is None:
            raise ImportError(
                "sqlalchemy is required for SQL data sources. Install with `pip install sqlalchemy psycopg[binary]`."
            )
        self.engine = create_engine(self.db_url, pool_pre_ping=True, pool_recycle=3600)

    def inspect_schema(
        self,
        include_row_counts: bool = True,
        max_tables: int = 200,
    ) -> Dict[str, Any]:
        insp = inspect(self.engine)
        tables = insp.get_table_names()
        if max_tables > 0:
            tables = tables[:max_tables]

        schema_tables: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        default_schema = insp.default_schema_name

        with self.engine.connect() as conn:
            for table in tables:
                columns = insp.get_columns(table)
                pk = insp.get_pk_constraint(table) or {}
                fk_list = insp.get_foreign_keys(table) or []

                row_count = None
                if include_row_counts:
                    try:
                        q_table = _quote_table(self.engine, table, default_schema)
                        row_count = int(conn.execute(text(f"SELECT COUNT(*) FROM {q_table}")).scalar_one())
                    except Exception:
                        row_count = None

                for fk in fk_list:
                    referred = fk.get("referred_table")
                    if not referred:
                        continue
                    relationships.append(
                        {
                            "from_table": table,
                            "from_columns": list(fk.get("constrained_columns") or []),
                            "to_table": referred,
                            "to_columns": list(fk.get("referred_columns") or []),
                        }
                    )

                schema_tables.append(
                    {
                        "name": table,
                        "n_rows": row_count,
                        "columns": [
                            {
                                "name": str(c.get("name", "")),
                                "dtype": str(c.get("type", "")),
                                "nullable": bool(c.get("nullable", True)),
                                "default": str(c.get("default")) if c.get("default") is not None else None,
                            }
                            for c in columns
                        ],
                        "primary_key": list(pk.get("constrained_columns") or []),
                        "foreign_keys": [
                            {
                                "constrained_columns": list(fk.get("constrained_columns") or []),
                                "referred_table": fk.get("referred_table"),
                                "referred_columns": list(fk.get("referred_columns") or []),
                            }
                            for fk in fk_list
                        ],
                    }
                )

        return {
            "dialect": str(self.engine.dialect.name),
            "tables": schema_tables,
            "relationships": relationships,
            "default_schema": default_schema,
        }

    def execute_query(self, query: str, limit: Optional[int] = None) -> pd.DataFrame:
        q = (query or "").strip().rstrip(";")
        if not q:
            raise ValueError("Empty SQL query.")

        final_query = q
        if limit is not None and limit > 0:
            final_query = f"SELECT * FROM ({q}) AS _q LIMIT {int(limit)}"

        with self.engine.connect() as conn:
            return pd.read_sql_query(text(final_query), conn)

    def load_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        q_table = _quote_table(self.engine, table_name)
        if limit is None or limit <= 0:
            query = f"SELECT * FROM {q_table}"
        else:
            query = f"SELECT * FROM {q_table} LIMIT {int(limit)}"
        with self.engine.connect() as conn:
            return pd.read_sql_query(text(query), conn)


def choose_primary_table(schema: Dict[str, Any]) -> str | None:
    tables = list((schema or {}).get("tables", []) or [])
    if not tables:
        return None
    with_counts = [t for t in tables if isinstance(t.get("n_rows"), int)]
    if with_counts:
        with_counts.sort(key=lambda t: int(t.get("n_rows") or 0), reverse=True)
        return str(with_counts[0].get("name"))
    return str(tables[0].get("name"))


def column_table_index(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for t in (schema or {}).get("tables", []) or []:
        table_name = str(t.get("name", ""))
        for c in t.get("columns", []) or []:
            col = str(c.get("name", "")).strip()
            if not col:
                continue
            idx.setdefault(col, [])
            if table_name not in idx[col]:
                idx[col].append(table_name)
    return idx


def find_join_path(schema: Dict[str, Any], start_table: str, end_table: str) -> List[str]:
    if start_table == end_table:
        return [start_table]

    rels = (schema or {}).get("relationships", []) or []
    graph: Dict[str, List[str]] = {}
    for r in rels:
        a = str(r.get("from_table", ""))
        b = str(r.get("to_table", ""))
        if not a or not b:
            continue
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)

    q: deque[tuple[str, List[str]]] = deque([(start_table, [start_table])])
    visited = {start_table}
    while q:
        node, path = q.popleft()
        for nxt in graph.get(node, []):
            if nxt in visited:
                continue
            new_path = path + [nxt]
            if nxt == end_table:
                return new_path
            visited.add(nxt)
            q.append((nxt, new_path))
    return []


def _find_relationship(schema: Dict[str, Any], left: str, right: str) -> Dict[str, Any] | None:
    for r in (schema or {}).get("relationships", []) or []:
        from_table = str(r.get("from_table", ""))
        to_table = str(r.get("to_table", ""))
        if (from_table == left and to_table == right) or (from_table == right and to_table == left):
            return r
    return None


def _join_condition(
    rel: Dict[str, Any],
    left_table: str,
    left_alias: str,
    right_table: str,
    right_alias: str,
    engine: Engine,
) -> str:
    from_table = str(rel.get("from_table", ""))
    to_table = str(rel.get("to_table", ""))
    from_cols = list(rel.get("from_columns") or [])
    to_cols = list(rel.get("to_columns") or [])
    if not from_cols or not to_cols or len(from_cols) != len(to_cols):
        raise ValueError(f"Invalid relationship metadata between {left_table} and {right_table}: {rel}")

    parts: List[str] = []
    if from_table == left_table and to_table == right_table:
        for lcol, rcol in zip(from_cols, to_cols):
            parts.append(
                f"{left_alias}.{_quote_ident(engine, str(lcol))} = {right_alias}.{_quote_ident(engine, str(rcol))}"
            )
    elif from_table == right_table and to_table == left_table:
        for rcol, lcol in zip(from_cols, to_cols):
            parts.append(
                f"{left_alias}.{_quote_ident(engine, str(lcol))} = {right_alias}.{_quote_ident(engine, str(rcol))}"
            )
    else:
        raise ValueError(f"Relationship does not connect requested tables: {left_table} <-> {right_table}")
    return " AND ".join(parts)


def build_groupby_query(
    *,
    engine: Engine,
    schema: Dict[str, Any],
    metric_col: str,
    group_cols: Sequence[str],
    agg: str = "sum",
    limit: int = 1000,
    preferred_fact_table: str | None = None,
) -> Dict[str, Any]:
    group_cols = [str(c) for c in group_cols if str(c).strip()]
    if not group_cols:
        raise ValueError("group_cols must contain at least one column.")

    idx = column_table_index(schema)
    all_metric_tables = idx.get(metric_col, [])
    if not all_metric_tables:
        raise ValueError(f"Metric column '{metric_col}' not found in SQL schema.")

    fact_table = preferred_fact_table if preferred_fact_table in all_metric_tables else all_metric_tables[0]
    selected_groups: List[Dict[str, str]] = []
    join_tables = {fact_table}

    for col in group_cols:
        tables = idx.get(col, [])
        if not tables:
            raise ValueError(f"Group column '{col}' not found in SQL schema.")

        best_table = fact_table if fact_table in tables else tables[0]
        if best_table != fact_table:
            path = find_join_path(schema, fact_table, best_table)
            if not path:
                raise ValueError(f"No join path between '{fact_table}' and '{best_table}' for column '{col}'.")
            join_tables.update(path)
        selected_groups.append({"column": col, "table": best_table})

    # Build deterministic path order from fact -> each group table.
    ordered_tables: List[str] = [fact_table]
    for g in selected_groups:
        tgt = g["table"]
        if tgt in ordered_tables:
            continue
        path = find_join_path(schema, fact_table, tgt)
        for t in path[1:]:
            if t not in ordered_tables:
                ordered_tables.append(t)

    aliases: Dict[str, str] = {table: f"t{i}" for i, table in enumerate(ordered_tables)}
    q_fact = _quote_table(engine, fact_table)
    from_sql = f"{q_fact} AS {aliases[fact_table]}"

    join_clauses: List[str] = []
    for i in range(1, len(ordered_tables)):
        left_table = ordered_tables[i - 1]
        right_table = ordered_tables[i]
        rel = _find_relationship(schema, left_table, right_table)
        if rel is None:
            # if direct relation not found, search an already-added table with a relation
            found = False
            for existing in ordered_tables[:i]:
                rel2 = _find_relationship(schema, existing, right_table)
                if rel2:
                    cond = _join_condition(
                        rel2,
                        existing,
                        aliases[existing],
                        right_table,
                        aliases[right_table],
                        engine,
                    )
                    join_clauses.append(
                        f"LEFT JOIN {_quote_table(engine, right_table)} AS {aliases[right_table]} ON {cond}"
                    )
                    found = True
                    break
            if found:
                continue
            raise ValueError(f"Unable to build join clause for table '{right_table}'.")

        cond = _join_condition(
            rel,
            left_table,
            aliases[left_table],
            right_table,
            aliases[right_table],
            engine,
        )
        join_clauses.append(
            f"LEFT JOIN {_quote_table(engine, right_table)} AS {aliases[right_table]} ON {cond}"
        )

    agg_fn = (agg or "sum").lower().strip()
    if agg_fn not in {"sum", "mean", "avg", "count", "min", "max", "median"}:
        agg_fn = "sum"
    if agg_fn == "mean":
        agg_fn = "avg"

    q_metric = f"{aliases[fact_table]}.{_quote_ident(engine, metric_col)}"
    value_expr = f"{agg_fn.upper()}({q_metric}) AS value"

    dim_selects: List[str] = []
    dim_groups: List[str] = []
    dim_aliases: List[str] = []
    for i, g in enumerate(selected_groups, start=1):
        alias = aliases[g["table"]]
        col = g["column"]
        out_alias = _sanitize_alias(f"g{i}_{col}")
        expr = f"{alias}.{_quote_ident(engine, col)}"
        dim_selects.append(f"{expr} AS {_quote_ident(engine, out_alias)}")
        dim_groups.append(expr)
        dim_aliases.append(out_alias)

    select_parts = [*dim_selects, value_expr]
    sql = (
        "SELECT "
        + ", ".join(select_parts)
        + f" FROM {from_sql} "
        + (" ".join(join_clauses) if join_clauses else "")
        + (" GROUP BY " + ", ".join(dim_groups) if dim_groups else "")
        + " ORDER BY value DESC"
    )
    if limit and limit > 0:
        sql += f" LIMIT {int(limit)}"

    return {
        "query": sql,
        "fact_table": fact_table,
        "group_columns": dim_aliases,
        "metric": metric_col,
        "agg": agg_fn,
        "joined_tables": ordered_tables,
    }


def compute_join_profile(
    *,
    engine: Engine,
    schema: Dict[str, Any],
    fact_table: str,
    dimension_table: str | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "fact_table": fact_table,
        "dimension_table": dimension_table,
        "join_path": [],
        "fact_rows": None,
        "joined_rows": None,
        "status": "ok",
    }
    q_fact = _quote_table(engine, fact_table)

    with engine.connect() as conn:
        out["fact_rows"] = int(conn.execute(text(f"SELECT COUNT(*) FROM {q_fact}")).scalar_one())

        if not dimension_table or dimension_table == fact_table:
            out["joined_rows"] = out["fact_rows"]
            return out

        path = find_join_path(schema, fact_table, dimension_table)
        if not path:
            out["status"] = "no_path"
            out["join_path"] = []
            return out
        out["join_path"] = path

        aliases: Dict[str, str] = {table: f"t{i}" for i, table in enumerate(path)}
        from_sql = f"{_quote_table(engine, fact_table)} AS {aliases[fact_table]}"
        joins: List[str] = []
        for i in range(1, len(path)):
            left = path[i - 1]
            right = path[i]
            rel = _find_relationship(schema, left, right)
            if rel is None:
                out["status"] = "invalid_path"
                return out
            cond = _join_condition(rel, left, aliases[left], right, aliases[right], engine)
            joins.append(f"LEFT JOIN {_quote_table(engine, right)} AS {aliases[right]} ON {cond}")

        query = f"SELECT COUNT(*) AS joined_rows FROM {from_sql} {' '.join(joins)}"
        out["joined_rows"] = int(conn.execute(text(query)).scalar_one())
        return out
