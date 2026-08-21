from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats

from ai_data_analyst_agents.statistics.models import ConfidenceInterval


def _numeric_clean(series: pd.Series) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(series, errors="coerce")).dropna()


def _validate_confidence_level(confidence_level: float) -> None:
    if not math.isfinite(float(confidence_level)) or not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be a finite number strictly between 0 and 1")


def single_mean_ci(series: pd.Series, *, confidence_level: float = 0.95, parameter: str | None = None) -> ConfidenceInterval:
    _validate_confidence_level(confidence_level)
    clean = _numeric_clean(series)
    n = int(clean.shape[0])
    if n == 0:
        raise ValueError("At least one numeric observation is required for a mean confidence interval")
    mean = float(clean.mean())
    if n <= 1:
        return ConfidenceInterval(
            parameter=parameter or "mean",
            point_estimate=mean,
            lower_bound=mean,
            upper_bound=mean,
            confidence_level=confidence_level,
            interpretation="Insufficient sample size for a non-degenerate confidence interval.",
        )
    sem = float(stats.sem(clean.to_numpy(), nan_policy="omit"))
    critical = float(stats.t.ppf((1.0 + confidence_level) / 2.0, df=n - 1))
    margin = critical * sem
    return ConfidenceInterval(
        parameter=parameter or "mean",
        point_estimate=mean,
        lower_bound=mean - margin,
        upper_bound=mean + margin,
        confidence_level=confidence_level,
        interpretation=f"With {confidence_level:.0%} confidence, the mean lies between {mean - margin:.4f} and {mean + margin:.4f}.",
    )


def difference_in_means_ci(
    group_a: pd.Series,
    group_b: pd.Series,
    *,
    confidence_level: float = 0.95,
    parameter: str = "difference_in_means",
) -> ConfidenceInterval:
    _validate_confidence_level(confidence_level)
    a = _numeric_clean(group_a)
    b = _numeric_clean(group_b)
    n_a = int(a.shape[0])
    n_b = int(b.shape[0])
    mean_diff = float(a.mean() - b.mean())
    if n_a == 0 or n_b == 0:
        raise ValueError("Both groups require numeric observations for a difference-in-means interval")
    if n_a <= 1 or n_b <= 1:
        return ConfidenceInterval(
            parameter=parameter,
            point_estimate=mean_diff,
            lower_bound=mean_diff,
            upper_bound=mean_diff,
            confidence_level=confidence_level,
            interpretation="Insufficient sample size for a non-degenerate difference-in-means interval.",
        )

    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    se = math.sqrt((var_a / n_a) + (var_b / n_b))
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = 0.0
    if n_a > 1:
        denominator += ((var_a / n_a) ** 2) / (n_a - 1)
    if n_b > 1:
        denominator += ((var_b / n_b) ** 2) / (n_b - 1)
    df = numerator / denominator if denominator else max(1, min(n_a, n_b) - 1)
    critical = float(stats.t.ppf((1.0 + confidence_level) / 2.0, df=df))
    margin = critical * se
    return ConfidenceInterval(
        parameter=parameter,
        point_estimate=mean_diff,
        lower_bound=mean_diff - margin,
        upper_bound=mean_diff + margin,
        confidence_level=confidence_level,
        interpretation=(
            f"With {confidence_level:.0%} confidence, the difference in means lies between "
            f"{mean_diff - margin:.4f} and {mean_diff + margin:.4f}."
        ),
    )


def single_proportion_ci(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
    parameter: str = "proportion",
) -> ConfidenceInterval:
    _validate_confidence_level(confidence_level)
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    if trials <= 0:
        return ConfidenceInterval(
            parameter=parameter,
            point_estimate=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            confidence_level=confidence_level,
            interpretation="No trials were available to estimate a proportion.",
        )
    point = successes / trials
    alpha = 1.0 - confidence_level
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    denom = 1.0 + (z**2 / trials)
    center = (point + z**2 / (2.0 * trials)) / denom
    margin = z * math.sqrt((point * (1 - point) / trials) + (z**2 / (4.0 * trials**2))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return ConfidenceInterval(
        parameter=parameter,
        point_estimate=point,
        lower_bound=lower,
        upper_bound=upper,
        confidence_level=confidence_level,
        interpretation=f"With {confidence_level:.0%} confidence, the proportion lies between {lower:.4f} and {upper:.4f}.",
    )


def difference_in_proportions_ci(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
    *,
    confidence_level: float = 0.95,
    parameter: str = "difference_in_proportions",
) -> ConfidenceInterval:
    _validate_confidence_level(confidence_level)
    if not 0 <= successes_a <= trials_a or not 0 <= successes_b <= trials_b:
        raise ValueError("success counts must be between 0 and their trial counts")
    p_a = successes_a / trials_a if trials_a else 0.0
    p_b = successes_b / trials_b if trials_b else 0.0
    diff = p_a - p_b
    if trials_a <= 0 or trials_b <= 0:
        return ConfidenceInterval(
            parameter=parameter,
            point_estimate=diff,
            lower_bound=diff,
            upper_bound=diff,
            confidence_level=confidence_level,
            interpretation="Both groups require positive trial counts to estimate a proportion difference interval.",
        )
    alpha = 1.0 - confidence_level
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    se = math.sqrt((p_a * (1.0 - p_a) / trials_a) + (p_b * (1.0 - p_b) / trials_b))
    margin = z * se
    return ConfidenceInterval(
        parameter=parameter,
        point_estimate=diff,
        lower_bound=diff - margin,
        upper_bound=diff + margin,
        confidence_level=confidence_level,
        interpretation=(
            f"With {confidence_level:.0%} confidence, the proportion difference lies between "
            f"{diff - margin:.4f} and {diff + margin:.4f}."
        ),
    )


def relative_lift_ci(
    baseline: float,
    interval: ConfidenceInterval,
    *,
    parameter: str = "relative_lift",
) -> ConfidenceInterval | None:
    if abs(baseline) < 1e-12:
        return None
    return ConfidenceInterval(
        parameter=parameter,
        point_estimate=interval.point_estimate / baseline,
        lower_bound=interval.lower_bound / baseline,
        upper_bound=interval.upper_bound / baseline,
        confidence_level=interval.confidence_level,
        interpretation=(
            f"Relative change versus baseline is estimated between {interval.lower_bound / baseline:.4f} "
            f"and {interval.upper_bound / baseline:.4f}."
        ),
    )


def regression_coefficient_cis(conf_int: pd.DataFrame, *, confidence_level: float = 0.95) -> list[ConfidenceInterval]:
    _validate_confidence_level(confidence_level)
    out: list[ConfidenceInterval] = []
    for name, row in conf_int.iterrows():
        lower = float(row.iloc[0])
        upper = float(row.iloc[1])
        point = float(np.mean([lower, upper]))
        out.append(
            ConfidenceInterval(
                parameter=f"coefficient:{name}",
                point_estimate=point,
                lower_bound=lower,
                upper_bound=upper,
                confidence_level=confidence_level,
                interpretation=(
                    f"With {confidence_level:.0%} confidence, coefficient {name} lies between {lower:.4f} and {upper:.4f}."
                ),
            )
        )
    return out


def bootstrap_difference_ci(
    group_a: pd.Series,
    group_b: pd.Series,
    *,
    statistic: str = "median",
    confidence_level: float = 0.95,
    n_resamples: int = 2000,
    random_seed: int = 0,
    parameter: str = "difference",
) -> ConfidenceInterval:
    """Deterministic percentile bootstrap CI for a mean or median difference."""
    _validate_confidence_level(confidence_level)
    a = pd.to_numeric(group_a, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(group_b, errors="coerce").dropna().to_numpy(dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError("Bootstrap difference intervals require at least two observations per group")
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'")
    reducer = np.mean if statistic == "mean" else np.median
    point = float(reducer(a) - reducer(b))
    rng = np.random.default_rng(random_seed)
    draws = max(200, min(int(n_resamples), 20000))
    estimates = np.empty(draws, dtype=float)
    for idx in range(draws):
        estimates[idx] = reducer(rng.choice(a, size=a.size, replace=True)) - reducer(
            rng.choice(b, size=b.size, replace=True)
        )
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return ConfidenceInterval(
        parameter=parameter,
        point_estimate=point,
        lower_bound=float(lower),
        upper_bound=float(upper),
        confidence_level=confidence_level,
        interpretation=(
            f"A deterministic percentile bootstrap estimates the {statistic} difference "
            f"between {float(lower):.4f} and {float(upper):.4f}."
        ),
    )
