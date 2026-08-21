from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AnalysisType = Literal[
    "descriptive",
    "trend",
    "segment_comparison",
    "diagnostic",
    "experiment_ab",
    "forecasting_unsupported",
    "impossible",
]

TaskType = Literal[
    "groupby_agg",
    "groupby2_agg",
    "filter_agg",
    "correlation",
    "distribution",
    "group_distribution",
    "recency_by_group",
    "topk",
    "timeseries_agg",
    "sql_query",
    "sql_join_profile",
    "kpi_template_apply",
    "metric_definition",
    "segment_analysis",
    "gap_decomposition",
    "cohort_analysis",
    "statistical_test",
    "ab_test",
    "ols_regression",
]

ArtifactSchemaVersion = Literal["2026.04.1"]
ARTIFACT_SCHEMA_VERSION: ArtifactSchemaVersion = "2026.04.1"


class FramingContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_metric: str | None = None
    aggregation_level: str
    metric_aggregation: str
    time_column: str | None = None
    segment_columns: List[str] = Field(default_factory=list)
    comparison_logic: str
    success_criterion: str
    analysis_limitations: List[str] = Field(default_factory=list)
    analysis_type: AnalysisType
    feasibility_status: Literal["feasible", "partially_feasible", "infeasible"]


class AnalysisPlanContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: ArtifactSchemaVersion
    business_question: str
    source_type: str
    suggested_domain: str
    domain_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_type: AnalysisType
    routing_reason: str
    routing_confidence: float = Field(ge=0.0, le=1.0)
    blocked_requirements: List[str] = Field(default_factory=list)
    feasibility_status: Literal["feasible", "partially_feasible", "infeasible"]
    framing: FramingContract
    assumptions: List[str] = Field(default_factory=list)
    suggested_slices: List[str] = Field(default_factory=list)
    requested_metrics: List[str] = Field(default_factory=list)


class TaskProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stages: List[Literal["seed", "heuristic", "sql_heuristic", "llm", "followup"]] = Field(default_factory=list)
    route: AnalysisType
    semantic_key: str


class AnalysisTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: TaskType
    params: Dict[str, Any] = Field(default_factory=dict)
    provenance: TaskProvenance


class TaskBudgetContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tasks: int = Field(ge=0)
    planned_tasks: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_budget(self) -> "TaskBudgetContract":
        if self.max_tasks > 0 and self.planned_tasks > self.max_tasks:
            raise ValueError("planned_tasks cannot exceed max_tasks")
        return self


class PlanningContract(FramingContract):
    model_config = ConfigDict(extra="forbid")

    experiment_group_column: str | None = None
    experiment_control_label: str | None = None
    experiment_treatment_label: str | None = None


class AnalysisTasksContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: ArtifactSchemaVersion
    analysis_type: AnalysisType
    routing_reason: str
    blocked_requirements: List[str] = Field(default_factory=list)
    feasibility_status: Literal["feasible", "partially_feasible", "infeasible"]
    planning_contract: PlanningContract
    task_budget: TaskBudgetContract
    tasks: List[AnalysisTask] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def validate_task_set(self) -> "AnalysisTasksContract":
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("analysis task IDs must be unique")
        semantic_keys = [task.provenance.semantic_key for task in self.tasks]
        if len(semantic_keys) != len(set(semantic_keys)):
            raise ValueError("analysis task semantic keys must be unique")
        if self.task_budget.planned_tasks != len(self.tasks):
            raise ValueError("task_budget.planned_tasks must equal the number of tasks")
        return self


class AnalysisReadinessContract(BaseModel):
    """Question-level coverage gate produced before report generation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: ArtifactSchemaVersion
    analysis_type: AnalysisType | None = None
    answer_status: Literal["ready", "partial", "blocked"]
    required_capabilities: List[str] = Field(default_factory=list)
    computed_capabilities: List[str] = Field(default_factory=list)
    missing_capabilities: List[str] = Field(default_factory=list)
    failed_or_skipped_capabilities: List[str] = Field(default_factory=list)
    reporting_guidance: List[str] = Field(default_factory=list)


class MetricsComputedItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: str
    artifact: str
    evidence_id: str | None = None


class MetricsFailedItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: str
    reason: str


class MetricsSkippedItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: str
    reason: str


class SemanticValidationItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: str
    status: Literal["ok", "invalid"]
    analysis_type: AnalysisType | None = None
    reason: str | None = None


class MetricsOutputsContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: ArtifactSchemaVersion
    analysis_type: AnalysisType | None = None
    planning_contract: PlanningContract | None = None
    computed: List[MetricsComputedItem] = Field(default_factory=list)
    failed: List[MetricsFailedItem] = Field(default_factory=list)
    skipped: List[MetricsSkippedItem] = Field(default_factory=list)
    semantic_validation: List[SemanticValidationItem] = Field(default_factory=list)
    metric_registry: Dict[str, Any] = Field(default_factory=dict)
    followup: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_task_outcomes(self) -> "MetricsOutputsContract":
        computed = [item.task_id for item in self.computed]
        failed = [item.task_id for item in self.failed]
        skipped = [item.task_id for item in self.skipped]
        for label, ids in (("computed", computed), ("failed", failed), ("skipped", skipped)):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate task IDs in metrics {label} outcomes")
        overlaps = (set(computed) & set(failed)) | (set(computed) & set(skipped)) | (set(failed) & set(skipped))
        if overlaps:
            raise ValueError(f"metric tasks cannot have multiple terminal outcomes: {sorted(overlaps)}")
        return self


class ReportMetadataContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: ArtifactSchemaVersion
    analysis_type: AnalysisType | None = None
    report_path: str
    used_fallback: bool
    unresolved_ev_placeholders: int = 0
    unsupported_numeric_claim_lines: int = 0
    section_completeness_ok: bool
    contradiction_count: int = 0
    consistency_issues: List[str] = Field(default_factory=list)


class RunScorecardContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: ArtifactSchemaVersion
    analysis_type: AnalysisType | None = None
    feasibility_status: Literal["feasible", "partially_feasible", "infeasible"] | None = None
    routing_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    framing_completeness_pct: float = Field(ge=0.0, le=1.0)
    task_summary: Dict[str, Any] = Field(default_factory=dict)
    evidence_coverage: Dict[str, Any] = Field(default_factory=dict)
    quality_gates: Dict[str, Any] = Field(default_factory=dict)
    final_quality_status: Literal["pass", "fail"]


def validate_analysis_plan_contract(obj: Dict[str, Any]) -> AnalysisPlanContract:
    return AnalysisPlanContract.model_validate(obj)


def validate_analysis_tasks_contract(obj: Dict[str, Any]) -> AnalysisTasksContract:
    return AnalysisTasksContract.model_validate(obj)


def validate_metrics_outputs_contract(obj: Dict[str, Any]) -> MetricsOutputsContract:
    return MetricsOutputsContract.model_validate(obj)


def validate_analysis_readiness_contract(obj: Dict[str, Any]) -> AnalysisReadinessContract:
    return AnalysisReadinessContract.model_validate(obj)


def validate_report_metadata_contract(obj: Dict[str, Any]) -> ReportMetadataContract:
    return ReportMetadataContract.model_validate(obj)


def validate_run_scorecard_contract(obj: Dict[str, Any]) -> RunScorecardContract:
    return RunScorecardContract.model_validate(obj)
