"""Strict onboarding inspection and validation records."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FieldStatus = Literal["confirmed", "inferred", "ambiguous", "missing"]


class OnboardingSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InspectionField(OnboardingSchema):
    status: FieldStatus
    value: Any = None
    candidate_paths: list[str] = Field(default_factory=list)
    rationale: str


class ProjectInspection(OnboardingSchema):
    source_project: str
    scanned_file_count: int = Field(ge=0)
    artifact_counts: dict[str, int]
    fields: dict[str, InspectionField]
    unresolved_business_fields: list[str]
    created_at_utc: datetime


class OnboardingValidation(OnboardingSchema):
    model_id: str
    bundle_mode: str
    valid: bool
    checks: dict[str, bool]
    source_commit: str | None
    source_worktree_clean: bool | None
    reference_sample_count: int = Field(ge=0)
    raw_feature_count: int = Field(ge=0)
    preprocessed_feature_count: int = Field(ge=0)
    maximum_probability_absolute_difference: float | None
    metric_absolute_differences: dict[str, float]
    material_discrepancies: list[str]
    warnings: list[str]
    created_at_utc: datetime
