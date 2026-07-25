"""Pydantic contracts for deterministic live-agent evaluation artifacts."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvaluationSchema(BaseModel):
    """Strict base for local evaluation records."""

    model_config = ConfigDict(extra="forbid")


class ScenarioEvaluation(EvaluationSchema):
    """Transparent checks and metadata for one completed scenario run."""

    scenario_name: str
    provider: str
    model: str
    execution_mode: str
    completed: bool
    used_fallback: bool
    triage_parse_success: bool
    recommendation_parse_success: bool
    first_pass_verification: bool
    final_verification_passed: bool
    revision_count: int = Field(ge=0)
    incident_type: str
    expected_incident_types: list[str]
    incident_compatible: bool
    diagnostic_route: str
    expected_routes: list[str]
    route_compatible: bool
    all_cited_evidence_valid: bool
    all_claims_cited: bool
    all_actions_cited: bool
    policy_compliant: bool
    approval_required: bool
    approval_completed: bool
    final_status: str
    latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    violations: list[str]
    notes: list[str]


class LiveEvaluationSummary(EvaluationSchema):
    """Aggregate deterministic metrics over all required live scenarios."""

    provider: str
    model: str
    scenario_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    structured_output_success_rate: float = Field(ge=0.0, le=1.0)
    incident_compatibility_rate: float = Field(ge=0.0, le=1.0)
    route_compatibility_rate: float = Field(ge=0.0, le=1.0)
    evidence_grounding_rate: float = Field(ge=0.0, le=1.0)
    policy_compliance_rate: float = Field(ge=0.0, le=1.0)
    first_pass_verification_rate: float = Field(ge=0.0, le=1.0)
    fallback_rate: float = Field(ge=0.0, le=1.0)
    approval_completion_rate: float = Field(ge=0.0, le=1.0)
    mean_latency_ms: float | None
    median_latency_ms: float | None
    total_input_tokens: int | None
    total_output_tokens: int | None
    total_tokens: int | None
    scenario_results: list[ScenarioEvaluation]
    limitations: list[str]
    created_at_utc: datetime

    @field_validator("created_at_utc")
    @classmethod
    def created_at_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must be timezone-aware.")
        return value.astimezone(UTC)
