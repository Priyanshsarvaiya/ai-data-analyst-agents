from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional
import math


AssumptionSeverity = Literal["info", "warn", "fail"]
DecisionLabel = Literal["reject_null", "fail_to_reject_null", "not_reliable"]


@dataclass(slots=True)
class AssumptionCheck:
    name: str
    passed: bool | None
    detail: str
    severity: AssumptionSeverity = "warn"
    metric: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ConfidenceInterval:
    parameter: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EffectSize:
    name: str
    value: float
    interpretation: str
    caveat: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HypothesisTestRequest:
    group_col: str
    metric: str
    group_a: str | int | float | None = None
    group_b: str | int | float | None = None
    compare_to_rest: bool = False
    paired: bool = False
    pair_id_col: str | None = None
    success_value: Any = 1
    alpha: float = 0.05
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"
    metric_type: Literal["auto", "binary", "continuous", "categorical"] = "auto"

    def __post_init__(self) -> None:
        _validate_alpha(self.alpha)
        if self.group_a is not None and self.group_b is not None and str(self.group_a) == str(self.group_b):
            raise ValueError("group_a and group_b must identify different groups")
        if self.paired and not self.pair_id_col:
            raise ValueError("paired tests require pair_id_col")


@dataclass(slots=True)
class ABTestRequest:
    group_col: str
    control: str | int | float
    treatment: str | int | float
    metric: str
    metric_type: Literal["auto", "binary", "continuous"] = "auto"
    success_value: Any = 1
    alpha: float = 0.05
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"

    def __post_init__(self) -> None:
        _validate_alpha(self.alpha)
        if str(self.control) == str(self.treatment):
            raise ValueError("control and treatment must identify different groups")


@dataclass(slots=True)
class RegressionRequest:
    target: str
    predictors: List[str]
    alpha: float = 0.05
    include_standardized: bool = True

    def __post_init__(self) -> None:
        _validate_alpha(self.alpha)
        self.predictors = list(dict.fromkeys(str(x) for x in self.predictors if str(x).strip()))
        if not self.predictors:
            raise ValueError("Regression requires at least one predictor")
        if self.target in self.predictors:
            raise ValueError("Regression target cannot also be a predictor")


@dataclass(slots=True)
class StatisticalSelection:
    analysis_type: Literal["hypothesis_test", "ab_test", "regression"]
    method: str
    reason: str
    warnings: List[str] = field(default_factory=list)
    fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StatisticalResult:
    analysis_id: str
    analysis_type: str
    method: str
    method_reason: str
    null_hypothesis: str | None
    alternative_hypothesis: str | None
    sample_sizes: Dict[str, int]
    alpha: float
    test_statistic: float | None
    p_value: float | None
    decision: DecisionLabel
    interpretation: str
    plain_language: str
    assumptions: List[AssumptionCheck] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence_intervals: List[ConfidenceInterval] = field(default_factory=list)
    effect_sizes: List[EffectSize] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    extra_outputs: Dict[str, Any] = field(default_factory=dict)
    status: Literal["completed", "skipped", "not_reliable"] = "completed"

    def __post_init__(self) -> None:
        _validate_alpha(self.alpha)
        if self.p_value is not None and not math.isfinite(float(self.p_value)):
            self.p_value = None
        if self.test_statistic is not None and not math.isfinite(float(self.test_statistic)):
            self.test_statistic = None
        if self.p_value is None and self.decision != "not_reliable":
            self.decision = "not_reliable"
            self.status = "not_reliable"
        if self.decision == "not_reliable":
            self.status = "not_reliable"
        if any(int(n) < 0 for n in self.sample_sizes.values()):
            raise ValueError("sample sizes must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["assumptions"] = [x.to_dict() for x in self.assumptions]
        data["confidence_intervals"] = [x.to_dict() for x in self.confidence_intervals]
        data["effect_sizes"] = [x.to_dict() for x in self.effect_sizes]
        return data


@dataclass(slots=True)
class StatisticalArtifactBundle:
    summary_path: str
    assumptions_path: str
    results_path: str
    diagnostics_path: str | None = None
    coefficients_path: str | None = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


def _validate_alpha(alpha: float) -> None:
    if not math.isfinite(float(alpha)) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be a finite number strictly between 0 and 1")
