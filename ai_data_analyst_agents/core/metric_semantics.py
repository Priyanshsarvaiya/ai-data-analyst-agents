from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from ai_data_analyst_agents.core.kpi_templates import (
    allowed_aggs_for_metric,
    default_agg_for_metric,
    infer_metric_kind,
)


GrainKind = str


@dataclass(frozen=True)
class MetricSemantics:
    kind: str
    allowed_aggs: Tuple[str, ...]
    default_agg: str
    compatible_grains: Tuple[GrainKind, ...]
    required_columns: Tuple[str, ...]


_KIND_GRAIN_COMPATIBILITY: Dict[str, Tuple[GrainKind, ...]] = {
    "additive": ("overall", "segment", "time", "group", "cohort"),
    "count": ("overall", "segment", "time", "group", "cohort"),
    "duration": ("overall", "segment", "time", "group", "cohort"),
    "ratio": ("overall", "segment", "time", "group"),
    "rate": ("overall", "segment", "time", "group"),
    "unknown": ("overall", "segment", "time", "group", "cohort"),
}

_REQUIRED_COLUMNS_BY_METRIC: Dict[str, Tuple[str, ...]] = {
    "avg_order_value": ("revenue", "order_id"),
    "conversion_rate": ("conversion",),
    "ctr": ("clicks", "impressions"),
    "cpa": ("spend", "conversions"),
    "roas": ("revenue", "spend"),
}


def parse_grain(aggregation_level: str | None) -> GrainKind:
    lvl = (aggregation_level or "").strip().lower()
    if not lvl:
        return "overall"
    if ":" in lvl:
        prefix = lvl.split(":", 1)[0].strip()
        return prefix or "overall"
    return lvl


def metric_semantics(metric_name: str) -> MetricSemantics:
    name = (metric_name or "").strip()
    key = name.lower()
    kind = infer_metric_kind(name)
    allowed = tuple(allowed_aggs_for_metric(name, metric_kind=kind))
    default = default_agg_for_metric(name)
    compatible_grains = _KIND_GRAIN_COMPATIBILITY.get(kind, _KIND_GRAIN_COMPATIBILITY["unknown"])
    required_columns = _REQUIRED_COLUMNS_BY_METRIC.get(key, tuple())
    return MetricSemantics(
        kind=kind,
        allowed_aggs=allowed,
        default_agg=default,
        compatible_grains=compatible_grains,
        required_columns=required_columns,
    )


def validate_metric_request(
    *,
    metric_name: str,
    agg: str,
    aggregation_level: str | None,
    available_columns: Iterable[str] | None = None,
) -> tuple[bool, str | None, MetricSemantics]:
    semantics = metric_semantics(metric_name)
    candidate_agg = (agg or "").strip().lower()
    grain = parse_grain(aggregation_level)

    if candidate_agg not in set(semantics.allowed_aggs):
        return (
            False,
            (
                f"Metric '{metric_name}' (kind={semantics.kind}) does not allow agg '{candidate_agg}'. "
                f"Allowed aggs: {list(semantics.allowed_aggs)}; safe default: '{semantics.default_agg}'."
            ),
            semantics,
        )

    if grain not in set(semantics.compatible_grains):
        return (
            False,
            (
                f"Metric '{metric_name}' (kind={semantics.kind}) is incompatible with grain '{grain}'. "
                f"Compatible grains: {list(semantics.compatible_grains)}."
            ),
            semantics,
        )

    if semantics.required_columns and available_columns is not None:
        cols = {str(c) for c in available_columns}
        missing = [c for c in semantics.required_columns if c not in cols]
        if missing:
            return (
                False,
                f"Metric '{metric_name}' requires columns {list(semantics.required_columns)}; missing {missing}.",
                semantics,
            )

    return True, None, semantics


def build_metric_registry_snapshot(metrics: List[str]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for metric in metrics:
        sem = metric_semantics(metric)
        out[metric] = {
            "kind": sem.kind,
            "allowed_aggs": list(sem.allowed_aggs),
            "default_agg": sem.default_agg,
            "compatible_grains": list(sem.compatible_grains),
            "required_columns": list(sem.required_columns),
        }
    return out
