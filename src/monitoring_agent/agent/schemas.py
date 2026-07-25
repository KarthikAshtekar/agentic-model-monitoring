"""Strict structured-output contracts for triage, recommendations, and approval."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

IncidentType = Literal[
    "normal_operation",
    "data_quality_failure",
    "feature_drift",
    "prediction_drift",
    "performance_degradation",
    "mixed_incident",
    "insufficient_evidence",
]
Severity = Literal["info", "low", "medium", "high", "critical"]
DiagnosticRoute = Literal[
    "no_additional_diagnostics",
    "data_quality_diagnostics",
    "drift_diagnostics",
    "performance_diagnostics",
    "mixed_diagnostics",
    "evidence_sufficiency_review",
]
ActionType = Literal[
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
Priority = Literal["low", "medium", "high", "urgent"]


class AgentSchema(BaseModel):
    """Base class suitable for strict provider JSON Schema."""

    model_config = ConfigDict(extra="forbid")


class AgentTriage(AgentSchema):
    """One evidence-grounded incident classification and diagnostic route."""

    incident_type: IncidentType
    severity: Severity
    diagnostic_route: DiagnosticRoute
    needs_additional_diagnostics: bool
    selected_evidence_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class EvidenceBackedClaim(AgentSchema):
    """A concise observed conclusion with its exact supporting IDs."""

    claim: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class RecommendedAction(AgentSchema):
    """A controlled, non-executing action proposed for human review."""

    action_type: ActionType
    action: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    priority: Priority
    evidence_ids: list[str] = Field(min_length=1)
    requires_human_approval: bool


class AgentRecommendation(AgentSchema):
    """Evidence-backed operational recommendation emitted by the LLM."""

    incident_type: IncidentType
    severity: Severity
    executive_summary: str = Field(min_length=1)
    claims: list[EvidenceBackedClaim] = Field(min_length=1)
    root_cause_hypothesis: str | None
    root_cause_evidence_ids: list[str]
    recommended_actions: list[RecommendedAction] = Field(min_length=1)
    uncertainties: list[str]
    overall_evidence_ids: list[str] = Field(min_length=1)
    requires_human_approval: bool
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("root_cause_hypothesis")
    @classmethod
    def root_cause_must_be_labelled_hypothesis(cls, value: str | None) -> str | None:
        """Prevent an inferred cause from being presented as an observed fact."""
        if value is not None and not value.strip().lower().startswith("hypothesis:"):
            raise ValueError("root_cause_hypothesis must begin with 'Hypothesis:'.")
        return value

    @model_validator(mode="after")
    def validate_hypothesis_evidence(self) -> AgentRecommendation:
        """Keep nullable hypotheses and their evidence lists internally consistent."""
        if self.root_cause_hypothesis is None and self.root_cause_evidence_ids:
            raise ValueError("A null root-cause hypothesis must have no evidence IDs.")
        if self.root_cause_hypothesis is not None and not self.root_cause_evidence_ids:
            raise ValueError("A root-cause hypothesis must cite evidence.")
        return self


class VerificationResult(AgentSchema):
    """Deterministic evidence and policy-verification outcome."""

    status: Literal["pass", "revise", "fallback"]
    violations: list[str]
    unsupported_evidence_ids: list[str]
    policy_checks: list[str]
    revision_feedback: str | None


class ApprovalDecision(AgentSchema):
    """Serializable reviewer decision used to resume the graph."""

    decision: Literal["approve", "reject", "request_revision"]
    comment: str | None
    reviewer: str = Field(min_length=1)
    reviewed_at_utc: datetime

    @field_validator("reviewed_at_utc")
    @classmethod
    def reviewed_at_must_be_utc(cls, value: datetime) -> datetime:
        """Require an unambiguous UTC approval timestamp."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reviewed_at_utc must be timezone-aware.")
        return value.astimezone(UTC)
