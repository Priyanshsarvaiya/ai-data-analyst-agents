from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


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


@dataclass(slots=True)
class ABTestRequest:
    group_col: str
    control: str | int | float
    treatment: str | int | float
    metric: str
    metric_type: Literal["auto", "binary", "continuous"] = "auto"
    success_value: Any = 1
    alpha: float = 0.05


@dataclass(slots=True)
class RegressionRequest:
    target: str
    predictors: List[str]
    alpha: float = 0.05
    include_standardized: bool = True


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
