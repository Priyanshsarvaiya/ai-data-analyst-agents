from __future__ import annotations

import re
from typing import Any

KPI_TEMPLATE_LIBRARY: dict[str, dict[str, Any]] = {
    "general": {
        "keywords": [],
        "metric_defs": [
            {"name": "record_count", "agg": "count", "candidates": ["id", "record_id", "event_id", "__rows__"]},
        ],
        "derived_defs": [],
        "segment_candidates": ["segment", "category", "type", "status", "region", "country", "channel"],
        "cohort_entity_candidates": ["customer_id", "user_id", "account_id", "entity_id"],
        "cohort_date_candidates": ["date", "created_at", "event_date", "timestamp"],
    },
    "ecommerce": {
        "keywords": ["order", "revenue", "gmv", "basket", "cart", "product", "aov", "checkout", "refund"],
        "metric_defs": [
            {"name": "total_revenue", "agg": "sum", "candidates": ["revenue", "net_revenue", "total_amount", "sales", "gmv", "amount"]},
            {"name": "order_count", "agg": "nunique", "candidates": ["order_id", "transaction_id", "invoice_id"]},
            {"name": "customer_count", "agg": "nunique", "candidates": ["customer_id", "buyer_id", "user_id"]},
            {"name": "units_sold", "agg": "sum", "candidates": ["quantity", "units"]},
            {"name": "avg_discount_pct", "agg": "mean", "candidates": ["discount_pct", "discount"]},
            {"name": "refund_amount", "agg": "sum", "candidates": ["refund_amount", "refunded_amount", "returns_value"]},
        ],
        "derived_defs": [
            {"name": "avg_order_value", "expr": "total_revenue / order_count"},
            {"name": "revenue_per_customer", "expr": "total_revenue / customer_count"},
            {"name": "units_per_order", "expr": "units_sold / order_count"},
            {"name": "refund_rate_value", "expr": "refund_amount / total_revenue"},
        ],
        "segment_candidates": ["country", "region", "market", "product_category", "category", "channel", "customer_id"],
        "cohort_entity_candidates": ["customer_id"],
        "cohort_date_candidates": ["order_date", "purchase_date", "date", "created_at"],
    },
    "saas": {
        "keywords": ["mrr", "arr", "subscription", "plan", "seat", "churn", "retention", "expansion"],
        "metric_defs": [
            {"name": "total_mrr", "agg": "sum", "candidates": ["mrr", "monthly_recurring_revenue", "revenue", "amount"]},
            {"name": "customer_count", "agg": "nunique", "candidates": ["customer_id", "account_id", "subscription_id"]},
            {"name": "churn_rate_signal", "agg": "mean", "candidates": ["churned_flag", "is_churned", "churn_rate"]},
            {"name": "expansion_mrr", "agg": "sum", "candidates": ["expansion_mrr", "expansion_revenue", "upgrade_mrr"]},
            {"name": "contraction_mrr", "agg": "sum", "candidates": ["contraction_mrr", "downgrade_mrr"]},
        ],
        "derived_defs": [
            {"name": "arpu", "expr": "total_mrr / customer_count"},
            {"name": "net_expansion", "expr": "expansion_mrr - contraction_mrr"},
        ],
        "segment_candidates": ["plan", "plan_tier", "country", "region", "industry", "segment"],
        "cohort_entity_candidates": ["customer_id", "account_id", "subscription_id"],
        "cohort_date_candidates": ["signup_date", "start_date", "created_at", "date"],
    },
    "marketing": {
        "keywords": ["campaign", "channel", "cac", "roas", "ctr", "cpc", "impression", "lead", "conversion"],
        "metric_defs": [
            {"name": "spend", "agg": "sum", "candidates": ["spend", "cost", "ad_spend"]},
            {"name": "conversions", "agg": "sum", "candidates": ["conversions", "orders", "purchases"]},
            {"name": "clicks", "agg": "sum", "candidates": ["clicks"]},
            {"name": "impressions", "agg": "sum", "candidates": ["impressions"]},
            {"name": "attributed_revenue", "agg": "sum", "candidates": ["attributed_revenue", "conversion_value", "revenue"]},
            {"name": "leads", "agg": "sum", "candidates": ["leads", "lead_count"]},
        ],
        "derived_defs": [
            {"name": "cpa", "expr": "spend / conversions"},
            {"name": "ctr", "expr": "clicks / impressions"},
            {"name": "cpc", "expr": "spend / clicks"},
            {"name": "cpm", "expr": "spend / impressions * 1000"},
            {"name": "roas", "expr": "attributed_revenue / spend"},
            {"name": "lead_to_conversion_rate", "expr": "conversions / leads"},
        ],
        "segment_candidates": ["channel", "campaign", "country", "region", "device", "audience"],
        "cohort_entity_candidates": ["customer_id", "lead_id", "user_id"],
        "cohort_date_candidates": ["date", "event_date", "created_at"],
    },
    "ops": {
        "keywords": ["sla", "latency", "throughput", "defect", "ticket", "resolution", "downtime", "incident"],
        "metric_defs": [
            {"name": "ticket_count", "agg": "count", "candidates": ["ticket_id", "incident_id", "case_id"]},
            {"name": "avg_resolution_time", "agg": "mean", "candidates": ["resolution_time_hours", "resolution_time", "cycle_time"]},
            {"name": "throughput", "agg": "sum", "candidates": ["units_processed", "items_processed", "quantity"]},
            {"name": "defect_count", "agg": "sum", "candidates": ["defects", "defect_count", "failed_units"]},
            {"name": "downtime_hours", "agg": "sum", "candidates": ["downtime_hours", "downtime", "outage_hours"]},
        ],
        "derived_defs": [{"name": "defects_per_unit", "expr": "defect_count / throughput"}],
        "segment_candidates": ["team", "region", "queue", "service", "priority"],
        "cohort_entity_candidates": ["ticket_id", "incident_id", "customer_id"],
        "cohort_date_candidates": ["created_at", "opened_at", "date"],
    },
    "finance": {
        "keywords": ["profit", "margin", "expense", "budget", "cash flow", "ebitda", "invoice", "cost"],
        "metric_defs": [
            {"name": "revenue", "agg": "sum", "candidates": ["revenue", "net_sales", "sales", "income"]},
            {"name": "cost", "agg": "sum", "candidates": ["cost", "cogs", "expense", "expenses"]},
            {"name": "operating_expense", "agg": "sum", "candidates": ["operating_expense", "opex"]},
            {"name": "invoice_count", "agg": "nunique", "candidates": ["invoice_id", "transaction_id"]},
        ],
        "derived_defs": [
            {"name": "gross_profit", "expr": "revenue - cost"},
            {"name": "gross_margin", "expr": "(revenue - cost) / revenue"},
            {"name": "operating_profit", "expr": "revenue - cost - operating_expense"},
        ],
        "segment_candidates": ["business_unit", "department", "cost_center", "region", "product", "account"],
        "cohort_entity_candidates": ["customer_id", "account_id", "invoice_id"],
        "cohort_date_candidates": ["invoice_date", "posting_date", "date", "created_at"],
    },
    "product": {
        "keywords": ["active user", "engagement", "feature", "session", "activation", "retention", "event"],
        "metric_defs": [
            {"name": "active_users", "agg": "nunique", "candidates": ["user_id", "account_id", "visitor_id"]},
            {"name": "sessions", "agg": "nunique", "candidates": ["session_id", "visit_id"]},
            {"name": "events", "agg": "count", "candidates": ["event_id", "event_name", "event_type"]},
            {"name": "session_duration", "agg": "mean", "candidates": ["session_duration", "duration_seconds", "engagement_time"]},
            {"name": "activation_rate", "agg": "mean", "candidates": ["activated", "activation_flag", "is_activated"]},
        ],
        "derived_defs": [
            {"name": "events_per_user", "expr": "events / active_users"},
            {"name": "sessions_per_user", "expr": "sessions / active_users"},
        ],
        "segment_candidates": ["platform", "device", "feature", "plan", "country", "acquisition_channel"],
        "cohort_entity_candidates": ["user_id", "account_id", "visitor_id"],
        "cohort_date_candidates": ["signup_date", "event_date", "timestamp", "created_at"],
    },
    "support": {
        "keywords": ["support", "ticket", "case", "csat", "resolution", "response time", "backlog"],
        "metric_defs": [
            {"name": "ticket_count", "agg": "nunique", "candidates": ["ticket_id", "case_id", "conversation_id"]},
            {"name": "avg_first_response_time", "agg": "mean", "candidates": ["first_response_time", "first_response_hours"]},
            {"name": "avg_resolution_time", "agg": "mean", "candidates": ["resolution_time", "resolution_hours"]},
            {"name": "csat", "agg": "mean", "candidates": ["csat", "satisfaction_score", "rating"]},
            {"name": "reopened_tickets", "agg": "sum", "candidates": ["reopened", "reopen_count", "is_reopened"]},
        ],
        "derived_defs": [{"name": "reopen_rate", "expr": "reopened_tickets / ticket_count"}],
        "segment_candidates": ["team", "agent", "queue", "priority", "channel", "issue_type", "region"],
        "cohort_entity_candidates": ["customer_id", "ticket_id", "case_id"],
        "cohort_date_candidates": ["created_at", "opened_at", "ticket_date", "date"],
    },
    "people": {
        "keywords": ["employee", "headcount", "attrition", "turnover", "hiring", "salary", "workforce"],
        "metric_defs": [
            {"name": "headcount", "agg": "nunique", "candidates": ["employee_id", "worker_id", "person_id"]},
            {"name": "hires", "agg": "sum", "candidates": ["hire_flag", "hires", "new_hires"]},
            {"name": "departures", "agg": "sum", "candidates": ["termination_flag", "departures", "attrition_flag"]},
            {"name": "avg_salary", "agg": "mean", "candidates": ["salary", "base_salary", "compensation"]},
        ],
        "derived_defs": [{"name": "turnover_signal", "expr": "departures / headcount"}],
        "segment_candidates": ["department", "team", "location", "job_level", "manager", "employment_type"],
        "cohort_entity_candidates": ["employee_id", "worker_id"],
        "cohort_date_candidates": ["hire_date", "start_date", "termination_date", "date"],
    },
}


DEFAULT_DOMAIN = "general"

_SEMANTIC_AGGS: dict[str, list[str]] = {
    "additive": ["sum", "mean", "min", "max", "median"],
    "count": ["sum", "mean", "min", "max", "median", "count"],
    "identifier": ["nunique", "count"],
    "ratio": ["mean", "median", "min", "max"],
    "rate": ["mean", "median", "min", "max"],
    "duration": ["mean", "median", "min", "max", "sum"],
    "unknown": ["sum", "mean", "count", "min", "max", "median"],
}

_RATE_TOKENS = {
    "rate",
    "ratio",
    "pct",
    "percent",
    "share",
    "ctr",
    "cvr",
    "aov",
    "arpu",
    "cpa",
    "roas",
    "avg_",
}
_COUNT_TOKENS = {
    "count",
    "orders",
    "transactions",
    "impressions",
    "clicks",
    "sessions",
    "visits",
    "users",
    "customers",
    "n_",
}
_DURATION_TOKENS = {
    "latency",
    "duration",
    "time",
    "cycle",
    "days",
    "hours",
    "minutes",
}
_ADDITIVE_TOKENS = {
    "revenue",
    "sales",
    "amount",
    "gmv",
    "spend",
    "cost",
    "profit",
    "price",
    "quantity",
    "units",
    "mrr",
    "arr",
    "value",
    "total",
}


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def score_business_domains(question: str, schema_cols: list[str]) -> list[dict[str, Any]]:
    """Return transparent, deterministic domain scores for planning and debugging."""
    q = (question or "").lower()
    cols = {_normalized_name(c) for c in schema_cols}
    ranked: list[dict[str, Any]] = []
    for domain, spec in KPI_TEMPLATE_LIBRARY.items():
        if domain == DEFAULT_DOMAIN:
            continue
        score = 0
        reasons: list[str] = []
        for kw in spec.get("keywords", []):
            if re.search(rf"(?<![a-z0-9]){re.escape(str(kw).lower())}(?![a-z0-9])", q):
                score += 3
                reasons.append(f"question:{kw}")
        for metric in spec.get("metric_defs", []):
            for cand in metric.get("candidates", []):
                if _normalized_name(cand) in cols:
                    score += 2
                    reasons.append(f"column:{cand}")
                    break
        segment_hits = [c for c in spec.get("segment_candidates", []) if _normalized_name(c) in cols]
        if segment_hits:
            score += min(2, len(segment_hits))
            reasons.extend(f"segment:{c}" for c in segment_hits[:2])
        ranked.append({"domain": domain, "score": score, "reasons": reasons})
    ranked.sort(key=lambda item: (-int(item["score"]), str(item["domain"])))
    total = sum(max(0, int(item["score"])) for item in ranked)
    for item in ranked:
        item["confidence"] = float(item["score"] / total) if total else 0.0
    return ranked


def detect_business_domain(question: str, schema_cols: list[str]) -> str:
    ranked = score_business_domains(question, schema_cols)
    if not ranked or int(ranked[0]["score"]) <= 0:
        return DEFAULT_DOMAIN
    return str(ranked[0]["domain"])


def pick_template_dimension(domain: str, schema_cols: list[str]) -> str | None:
    spec = KPI_TEMPLATE_LIBRARY.get(domain, {})
    cols = {_normalized_name(c): str(c) for c in schema_cols}
    for c in spec.get("segment_candidates", []):
        match = cols.get(_normalized_name(c))
        if match is not None:
            return match
    return None


def pick_cohort_columns(domain: str, schema_cols: list[str]) -> tuple[str | None, str | None]:
    spec = KPI_TEMPLATE_LIBRARY.get(domain, {})
    cols = {_normalized_name(c): str(c) for c in schema_cols}
    entity_col = None
    date_col = None
    for c in spec.get("cohort_entity_candidates", []):
        if _normalized_name(c) in cols:
            entity_col = cols[_normalized_name(c)]
            break
    for c in spec.get("cohort_date_candidates", []):
        if _normalized_name(c) in cols:
            date_col = cols[_normalized_name(c)]
            break
    return entity_col, date_col


def infer_metric_kind(metric_name: str) -> str:
    name = (metric_name or "").strip().lower()
    if not name:
        return "unknown"

    if "_id" in name or name.endswith("id"):
        return "identifier"

    if any(tok in name for tok in _RATE_TOKENS):
        if "rate" in name or "pct" in name or "percent" in name or "share" in name:
            return "rate"
        return "ratio"

    if any(tok in name for tok in _COUNT_TOKENS):
        return "count"

    if any(tok in name for tok in _DURATION_TOKENS):
        return "duration"

    if any(tok in name for tok in _ADDITIVE_TOKENS):
        return "additive"

    return "unknown"


def allowed_aggs_for_metric(metric_name: str, metric_kind: str | None = None) -> list[str]:
    kind = (metric_kind or infer_metric_kind(metric_name)).strip().lower()
    return list(_SEMANTIC_AGGS.get(kind, _SEMANTIC_AGGS["unknown"]))


def default_agg_for_metric(metric_name: str, preferred: str | None = None) -> str:
    kind = infer_metric_kind(metric_name)
    allowed = allowed_aggs_for_metric(metric_name, metric_kind=kind)
    if preferred:
        pref = str(preferred).strip().lower()
        if pref in allowed:
            return pref
    if kind == "identifier" and "nunique" in allowed:
        return "nunique"
    if "sum" in allowed:
        return "sum"
    if "mean" in allowed:
        return "mean"
    if "count" in allowed:
        return "count"
    return allowed[0] if allowed else "sum"


def is_agg_allowed_for_metric(metric_name: str, agg: str, metric_kind: str | None = None) -> bool:
    a = str(agg or "").strip().lower()
    if not a:
        return False
    return a in set(allowed_aggs_for_metric(metric_name, metric_kind=metric_kind))
