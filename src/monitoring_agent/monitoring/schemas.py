"""Pydantic contracts for deterministic monitoring results and evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EvidenceDomain = Literal[
    "data_quality",
    "feature_drift",
    "prediction_drift",
    "performance",
    "system",
]
EvidenceStatus = Literal["pass", "warning", "critical", "not_evaluated"]
EvidenceSeverity = Literal["info", "low", "medium", "high", "critical"]
IncidentCandidate = Literal[
    "normal_operation",
    "data_quality_failure",
    "feature_drift",
    "prediction_drift",
    "performance_degradation",
    "mixed_incident",
    "insufficient_evidence",
]


class MonitoringSchema(BaseModel):
    """Strict base contract for monitoring records."""

    model_config = ConfigDict(extra="forbid")


class EvidenceItem(MonitoringSchema):
    """One traceable deterministic monitoring conclusion."""

    evidence_id: str = Field(min_length=1)
    model_id: str = "credit_default"
    domain_id: str = "credit_risk"
    domain: EvidenceDomain
    metric: str
    status: EvidenceStatus
    severity: EvidenceSeverity
    observed_value: Any = None
    reference_value: Any = None
    threshold: Any = None
    feature: str | None = None
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    source: str


class DataQualityResult(MonitoringSchema):
    """Structural and value-level batch validation result."""

    batch_valid: bool
    batch_blocked: bool
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    missing_required_columns: list[str]
    unexpected_columns: list[str]
    duplicate_record_count: int = Field(ge=0)
    duplicate_record_rate: float = Field(ge=0.0)
    feature_results: list[dict[str, Any]]
    evidence: list[EvidenceItem]


class DriftResult(MonitoringSchema):
    """Feature- and prediction-distribution comparison result."""

    evaluated_feature_count: int = Field(ge=0)
    warning_feature_count: int = Field(ge=0)
    critical_feature_count: int = Field(ge=0)
    feature_metrics: dict[str, dict[str, Any]]
    prediction_metrics: dict[str, Any]
    evidence: list[EvidenceItem]


class PerformanceResult(MonitoringSchema):
    """Labelled performance comparison at policy and default thresholds."""

    evaluated: bool
    reason_not_evaluated: str | None
    sample_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    feature_row_count: int = Field(default=0, ge=0)
    labelled_row_count: int = Field(default=0, ge=0)
    label_coverage_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_required_sample_size: int = Field(default=0, ge=0)
    metrics_at_default_threshold: dict[str, Any]
    metrics_at_operating_threshold: dict[str, Any]
    metric_deltas: dict[str, Any]
    evidence: list[EvidenceItem]


class MonitoringRunResult(MonitoringSchema):
    """Complete evidence record for one replayed monitoring batch."""

    run_id: str
    scenario_name: str
    created_at_utc: datetime
    model_id: str = "credit_default"
    display_name: str = "Credit Default XGBoost"
    model_name: str
    model_version: str
    task_type: str = "binary_classification"
    domain_id: str = "credit_risk"
    bundle_mode: str = "live_inference"
    reference_sample_count: int = Field(default=6002, ge=1)
    feature_count: int = Field(default=36, ge=1)
    use_case: str | None = "Academic next-month credit-default risk monitoring"
    positive_outcome: str | None = "next-month credit default"
    prediction_unit: str | None = "credit-card account record"
    allowed_action_types: list[str] = Field(
        default_factory=lambda: [
            "no_action",
            "continue_monitoring",
            "quarantine_batch",
            "investigate_data_pipeline",
            "investigate_feature_drift",
            "collect_more_labels",
            "evaluate_threshold",
            "evaluate_recalibration",
            "evaluate_retraining",
            "run_challenger_in_shadow",
            "escalate_model_governance",
        ]
    )
    prohibited_claims: list[str] = Field(default_factory=list)
    safe_business_terminology: list[str] = Field(default_factory=list)
    domain_limitations: list[str] = Field(default_factory=list)
    operating_threshold: float = Field(ge=0.0, le=1.0)
    batch_valid: bool
    batch_blocked: bool
    labels_available: bool
    feature_row_count: int = Field(default=0, ge=0)
    labelled_row_count: int = Field(default=0, ge=0)
    label_coverage_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_labelled_sample_size: int = Field(default=0, ge=0)
    data_quality: DataQualityResult
    drift: DriftResult
    performance: PerformanceResult
    incident_candidates: list[IncidentCandidate]
    overall_severity: EvidenceSeverity
    evidence: list[EvidenceItem]
    report_paths: list[str] = Field(default_factory=list)

    @field_validator("created_at_utc")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        """Reject ambiguous local timestamps."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must be timezone-aware.")
        return value
