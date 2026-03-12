from __future__ import annotations

import math

import pandas as pd

from ai_data_analyst_agents.statistics.models import EffectSize


def _interpret_cohens_d(value: float) -> str:
    abs_v = abs(value)
    if abs_v < 0.2:
        return "negligible"
    if abs_v < 0.5:
        return "small"
    if abs_v < 0.8:
        return "medium"
    return "large"


def cohens_d(group_a: pd.Series, group_b: pd.Series) -> EffectSize:
    a = pd.to_numeric(group_a, errors="coerce").dropna()
    b = pd.to_numeric(group_b, errors="coerce").dropna()
    n_a = int(a.shape[0])
    n_b = int(b.shape[0])
    if n_a < 2 or n_b < 2:
        value = 0.0
    else:
        var_a = float(a.var(ddof=1))
        var_b = float(b.var(ddof=1))
        pooled = math.sqrt((((n_a - 1) * var_a) + ((n_b - 1) * var_b)) / max(1, (n_a + n_b - 2)))
        value = 0.0 if pooled == 0 else float((a.mean() - b.mean()) / pooled)
    return EffectSize(
        name="cohens_d",
        value=value,
        interpretation=_interpret_cohens_d(value),
        caveat="Magnitude thresholds are heuristic and context-dependent.",
    )


def relative_lift(treatment: float, control: float) -> EffectSize:
    if abs(control) < 1e-12:
        value = 0.0
        caveat = "Control mean is zero; lift is not stable and is reported as 0.0."
    else:
        value = float((treatment - control) / control)
        caveat = "Relative lift should be interpreted alongside the absolute difference and CI."
    magnitude = "negative" if value < 0 else ("flat" if abs(value) < 0.01 else "positive")
    return EffectSize(name="relative_lift", value=value, interpretation=magnitude, caveat=caveat)


def percent_change(treatment: float, control: float) -> EffectSize:
    eff = relative_lift(treatment, control)
    return EffectSize(
        name="percent_change",
        value=eff.value * 100.0,
        interpretation=eff.interpretation,
        caveat=eff.caveat,
    )


def cramers_v(chi2_stat: float, observed: pd.DataFrame) -> EffectSize:
    n = int(observed.to_numpy().sum())
    rows, cols = observed.shape
    denom = min(rows - 1, cols - 1)
    value = 0.0 if n <= 0 or denom <= 0 else math.sqrt(chi2_stat / (n * denom))
    if value < 0.1:
        label = "weak"
    elif value < 0.3:
        label = "small-to-moderate"
    elif value < 0.5:
        label = "moderate"
    else:
        label = "strong"
    return EffectSize(
        name="cramers_v",
        value=float(value),
        interpretation=label,
        caveat="Association strength thresholds are heuristic and sensitive to table size.",
    )


def odds_ratio(success_treatment: int, failure_treatment: int, success_control: int, failure_control: int) -> EffectSize:
    num = (success_treatment + 0.5) * (failure_control + 0.5)
    den = (failure_treatment + 0.5) * (success_control + 0.5)
    value = float(num / den) if den else 0.0
    if value < 0.9:
        label = "lower odds in treatment"
    elif value > 1.1:
        label = "higher odds in treatment"
    else:
        label = "roughly neutral odds"
    return EffectSize(
        name="odds_ratio",
        value=value,
        interpretation=label,
        caveat="Odds ratios are most interpretable for binary outcomes and can exaggerate rare-event changes.",
    )


def regression_fit_effects(r_squared: float, adjusted_r_squared: float) -> list[EffectSize]:
    return [
        EffectSize(
            name="r_squared",
            value=float(r_squared),
            interpretation="share of target variance explained by the fitted model",
            caveat="High R-squared does not imply causality or out-of-sample usefulness.",
        ),
        EffectSize(
            name="adjusted_r_squared",
            value=float(adjusted_r_squared),
            interpretation="variance explained after penalizing additional predictors",
            caveat="Adjusted R-squared is still descriptive and depends on the chosen specification.",
        ),
    ]
