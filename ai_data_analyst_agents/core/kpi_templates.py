from __future__ import annotations

from typing import Any, Dict, List, Tuple


KPI_TEMPLATE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "ecommerce": {
        "keywords": ["order", "orders", "revenue", "gmv", "basket", "cart", "product", "aov", "country"],
        "metric_defs": [
            {"name": "total_revenue", "agg": "sum", "candidates": ["revenue", "total_amount", "sales", "amount", "price"]},
            {"name": "order_count", "agg": "count", "candidates": ["order_id", "transaction_id", "invoice_id"]},
            {"name": "units_sold", "agg": "sum", "candidates": ["quantity", "units"]},
            {"name": "avg_discount_pct", "agg": "mean", "candidates": ["discount_pct", "discount"]},
        ],
        "derived_defs": [
            {"name": "avg_order_value", "expr": "total_revenue / order_count"},
        ],
        "segment_candidates": ["country", "region", "market", "product_category", "category", "channel", "customer_id"],
        "cohort_entity_candidates": ["customer_id"],
        "cohort_date_candidates": ["order_date", "purchase_date", "date", "created_at"],
    },
    "saas": {
        "keywords": ["mrr", "arr", "subscription", "plan", "seat", "churn", "retention", "expansion"],
        "metric_defs": [
            {"name": "total_mrr", "agg": "sum", "candidates": ["mrr", "monthly_recurring_revenue", "revenue", "amount"]},
            {"name": "customer_count", "agg": "count", "candidates": ["customer_id", "account_id", "subscription_id"]},
            {"name": "churn_rate_signal", "agg": "mean", "candidates": ["churned_flag", "is_churned", "churn_rate"]},
        ],
        "derived_defs": [
            {"name": "arpu_like", "expr": "total_mrr / customer_count"},
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
        ],
        "derived_defs": [
            {"name": "cpa", "expr": "spend / conversions"},
            {"name": "ctr", "expr": "clicks / impressions"},
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
        ],
        "derived_defs": [],
        "segment_candidates": ["team", "region", "queue", "service", "priority"],
        "cohort_entity_candidates": ["ticket_id", "incident_id", "customer_id"],
        "cohort_date_candidates": ["created_at", "opened_at", "date"],
    },
}


DEFAULT_DOMAIN = "ecommerce"


def detect_business_domain(question: str, schema_cols: List[str]) -> str:
    q = (question or "").lower()
    cols = {str(c).lower() for c in schema_cols}
    best_domain = DEFAULT_DOMAIN
    best_score = -1
    for domain, spec in KPI_TEMPLATE_LIBRARY.items():
        score = 0
        for kw in spec.get("keywords", []):
            if kw in q:
                score += 3
        for metric in spec.get("metric_defs", []):
            for cand in metric.get("candidates", []):
                if str(cand).lower() in cols:
                    score += 1
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


def pick_template_dimension(domain: str, schema_cols: List[str]) -> str | None:
    spec = KPI_TEMPLATE_LIBRARY.get(domain, {})
    cols = set(schema_cols)
    for c in spec.get("segment_candidates", []):
        if c in cols:
            return c
    return None


def pick_cohort_columns(domain: str, schema_cols: List[str]) -> Tuple[str | None, str | None]:
    spec = KPI_TEMPLATE_LIBRARY.get(domain, {})
    cols = set(schema_cols)
    entity_col = None
    date_col = None
    for c in spec.get("cohort_entity_candidates", []):
        if c in cols:
            entity_col = c
            break
    for c in spec.get("cohort_date_candidates", []):
        if c in cols:
            date_col = c
            break
    return entity_col, date_col

