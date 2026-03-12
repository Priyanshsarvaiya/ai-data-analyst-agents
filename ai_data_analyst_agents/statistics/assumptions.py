from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from ai_data_analyst_agents.statistics.models import AssumptionCheck


def _series_name(series: pd.Series, fallback: str) -> str:
    name = getattr(series, "name", None)
    if isinstance(name, str) and name.strip():
        return name
    return fallback


def to_numeric_series(series: pd.Series, *, name: str | None = None) -> tuple[pd.Series, AssumptionCheck]:
    original_n = int(series.shape[0])
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    kept_n = int(numeric.shape[0])
    dropped = original_n - kept_n
    detail = f"Converted to numeric and kept {kept_n}/{original_n} rows."
    if dropped > 0:
        detail += f" Dropped {dropped} non-numeric or missing rows."
    check = AssumptionCheck(
        name="missingness_impact",
        passed=dropped == 0,
        detail=detail,
        severity="warn" if dropped > 0 else "info",
        metric=name or _series_name(series, "series"),
    )
    return numeric, check


def check_sample_size(series: pd.Series, *, min_n: int, label: str) -> AssumptionCheck:
    n = int(series.shape[0])
    return AssumptionCheck(
        name="sample_size",
        passed=n >= min_n,
        detail=f"{label} has n={n}; recommended minimum is {min_n}.",
        severity="fail" if n < max(3, min_n // 2) else ("warn" if n < min_n else "info"),
        metric=label,
    )


def check_group_balance(n_a: int, n_b: int, *, label_a: str, label_b: str) -> AssumptionCheck:
    larger = max(n_a, n_b)
    smaller = min(n_a, n_b)
    ratio = float(larger / smaller) if smaller else float("inf")
    if not np.isfinite(ratio):
        passed = False
        severity = "fail"
    else:
        passed = ratio <= 3.0
        severity = "warn" if ratio > 3.0 else "info"
    return AssumptionCheck(
        name="group_balance",
        passed=passed,
        detail=f"Group sizes are {label_a}={n_a}, {label_b}={n_b}; imbalance ratio={ratio:.2f}.",
        severity=severity,
    )


def check_near_zero_variance(series: pd.Series, *, label: str) -> AssumptionCheck:
    variance = float(pd.to_numeric(series, errors="coerce").dropna().var(ddof=1)) if series.shape[0] > 1 else 0.0
    passed = variance > 1e-12
    return AssumptionCheck(
        name="near_zero_variance",
        passed=passed,
        detail=f"{label} variance={variance:.6g}.",
        severity="fail" if not passed else "info",
        metric=label,
    )


def check_normality(series: pd.Series, *, alpha: float, label: str) -> AssumptionCheck:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = int(clean.shape[0])
    if n < 3:
        return AssumptionCheck(
            name="normality",
            passed=None,
            detail=f"{label} has fewer than 3 observations; normality cannot be assessed.",
            severity="warn",
            metric=label,
        )
    if n <= 5000:
        stat, p_value = stats.shapiro(clean.to_numpy())
        passed = bool(p_value >= alpha)
        return AssumptionCheck(
            name="normality",
            passed=passed,
            detail=f"Shapiro-Wilk p-value for {label} is {p_value:.4f} (statistic={stat:.4f}).",
            severity="warn" if not passed else "info",
            metric=label,
        )

    skew = float(stats.skew(clean.to_numpy(), bias=False)) if n > 2 else 0.0
    passed = abs(skew) < 1.0
    return AssumptionCheck(
        name="normality",
        passed=passed,
        detail=f"Large-sample normality proxy for {label}: skew={skew:.4f} with n={n}.",
        severity="warn" if not passed else "info",
        metric=label,
    )


def check_equal_variance(a: pd.Series, b: pd.Series, *, alpha: float) -> AssumptionCheck:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return AssumptionCheck(
            name="equal_variance",
            passed=None,
            detail="Equal-variance test skipped because one or both groups have fewer than 2 observations.",
            severity="warn",
        )
    stat, p_value = stats.levene(a.to_numpy(), b.to_numpy(), center="median")
    passed = bool(p_value >= alpha)
    return AssumptionCheck(
        name="equal_variance",
        passed=passed,
        detail=f"Levene test p-value={p_value:.4f} (statistic={stat:.4f}).",
        severity="warn" if not passed else "info",
    )


def check_outlier_sensitivity(series: pd.Series, *, label: str) -> AssumptionCheck:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.shape[0] < 4:
        return AssumptionCheck(
            name="outlier_sensitivity",
            passed=None,
            detail=f"{label} has fewer than 4 observations; outlier sensitivity not assessed.",
            severity="warn",
            metric=label,
        )
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    if iqr == 0:
        count = 0
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((clean < lower) | (clean > upper)).sum())
    frac = count / max(1, int(clean.shape[0]))
    return AssumptionCheck(
        name="outlier_sensitivity",
        passed=frac <= 0.05,
        detail=f"{label} has {count} Tukey outliers ({frac:.1%} of analyzed rows).",
        severity="warn" if frac > 0.05 else "info",
        metric=label,
    )


def coerce_binary(series: pd.Series, *, success_value: Any) -> tuple[pd.Series, AssumptionCheck]:
    raw = series.dropna()
    binary = raw.map(lambda x: 1 if x == success_value else (0 if x in {0, 1, True, False} or x != success_value else 0))
    unique_vals = set(binary.dropna().unique().tolist())
    passed = unique_vals.issubset({0, 1})
    return binary.astype(int), AssumptionCheck(
        name="binary_metric_validity",
        passed=passed,
        detail=f"Binary coercion produced values {sorted(unique_vals)} using success_value={success_value!r}.",
        severity="fail" if not passed else "info",
        metric=_series_name(series, "binary_metric"),
    )


def check_binary_values(series: pd.Series, *, success_value: Any) -> AssumptionCheck:
    raw = series.dropna()
    unique_vals = set(raw.unique().tolist())
    passed = unique_vals.issubset({0, 1, True, False, success_value})
    return AssumptionCheck(
        name="binary_metric_validity",
        passed=passed,
        detail=f"Observed unique values={sorted(repr(v) for v in unique_vals)} with success_value={success_value!r}.",
        severity="fail" if not passed else "info",
        metric=_series_name(series, "binary_metric"),
    )


def check_expected_counts(observed: pd.DataFrame) -> tuple[AssumptionCheck, pd.DataFrame]:
    stat, p_value, dof, expected = stats.chi2_contingency(observed.to_numpy())
    expected_df = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    min_expected = float(expected_df.min().min()) if not expected_df.empty else 0.0
    passed = min_expected >= 5.0
    detail = (
        f"Chi-square expected-count check: min expected cell count={min_expected:.2f}; "
        f"chi-square statistic={stat:.4f}, dof={dof}, p-value={p_value:.4f}."
    )
    return AssumptionCheck(
        name="expected_counts",
        passed=passed,
        detail=detail,
        severity="warn" if not passed else "info",
    ), expected_df


def check_multiple_comparisons(n_tests: int) -> AssumptionCheck:
    passed = n_tests <= 1
    return AssumptionCheck(
        name="multiple_comparisons",
        passed=passed,
        detail=f"This run executed {n_tests} statistical test(s). Adjust for multiplicity if many related hypotheses are interpreted together.",
        severity="warn" if n_tests > 1 else "info",
    )


def summarize_failures(checks: Iterable[AssumptionCheck]) -> list[str]:
    lines: list[str] = []
    for check in checks:
        if check.passed is False and check.severity in {"warn", "fail"}:
            lines.append(check.detail)
    return lines


def regression_design_checks(design: pd.DataFrame, target: pd.Series) -> list[AssumptionCheck]:
    checks: list[AssumptionCheck] = []
    checks.append(check_sample_size(target, min_n=max(20, design.shape[1] * 5), label="regression target"))
    rank = int(np.linalg.matrix_rank(design.to_numpy())) if not design.empty else 0
    full_rank = rank == design.shape[1]
    checks.append(
        AssumptionCheck(
            name="design_matrix_rank",
            passed=full_rank,
            detail=f"Design matrix rank={rank}; columns={design.shape[1]}.",
            severity="fail" if not full_rank else "info",
        )
    )
    for col in design.columns:
        if col == "const":
            continue
        checks.append(check_near_zero_variance(design[col], label=col))
    return checks
