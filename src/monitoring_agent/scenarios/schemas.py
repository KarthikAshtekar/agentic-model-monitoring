"""Pydantic contracts for generated monitoring scenarios."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ScenarioManifest(BaseModel):
    """Provenance and intended evidence contract for one replay scenario."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = "credit_default"
    scenario_name: str
    scenario_version: str
    description: str
    simulation_type: str
    random_seed: int
    source_reference_sample_count: int = Field(ge=1)
    generated_sample_count: int = Field(ge=1)
    labels_available: bool = True
    labelled_sample_count: int = Field(default=1000, ge=0)
    labels_complete: bool = True
    expected_incident_candidates: list[str]
    feature_modifications: list[dict[str, Any]]
    label_modifications: list[dict[str, Any]]
    limitations: list[str]
    created_at_utc: datetime

    @field_validator("created_at_utc")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must be timezone-aware.")
        return value
