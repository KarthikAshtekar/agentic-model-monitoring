"""Pydantic schemas for the exported credit-default monitoring bundle."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    """Base model that rejects undocumented bundle fields."""

    model_config = ConfigDict(extra="forbid")


class ModelMetadata(StrictSchema):
    """Identity, provenance, compatibility, and runtime status for the model."""

    model_name: str
    model_family: str
    model_version: str
    source_repository: str
    source_repository_path: str
    source_artifact_paths: list[str]
    dataset_name: str
    target_column: str
    positive_class: int
    feature_count: int = Field(ge=1)
    operating_threshold: float = Field(ge=0.0, le=1.0)
    default_threshold: float = Field(ge=0.0, le=1.0)
    reference_split: str
    reference_sample_count: int = Field(ge=1)
    inference_artifact_path: str | None
    inference_available: bool
    integration_strategy: str
    training_library_versions: dict[str, str]
    export_library_versions: dict[str, str]
    bundle_created_at_utc: str
    production_status: str
    compatibility_warnings: list[str]
    limitations: list[str]


class TargetDefinition(StrictSchema):
    """Definition of the supervised target kept outside inference features."""

    name: str
    exported_name: str
    dtype: str
    positive_class: int
    meaning: str


class FeatureDefinition(StrictSchema):
    """Observed reference metadata for one ordered inference feature."""

    name: str
    position: int = Field(ge=0)
    dtype: str
    role: str
    nullable: bool
    allowed_values: list[Any] | None
    observed_min: float | int | None
    observed_max: float | int | None


class FeatureSchema(StrictSchema):
    """Complete ordered inference schema and target boundary."""

    schema_version: str
    ordered_features: list[str]
    required_columns: list[str]
    identifier_columns: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    integer_constrained_features: list[str]
    target: TargetDefinition
    features: list[FeatureDefinition]
    source_reference_split: str
    observed_range_policy: str


class ManifestFile(StrictSchema):
    """Integrity and provenance entry for one exported payload file."""

    relative_path: str
    purpose: str
    source_path: str | None
    generation_method: str | None
    row_count: int | None = Field(default=None, ge=0)
    column_count: int | None = Field(default=None, ge=0)
    file_size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)


class BundleManifest(StrictSchema):
    """Top-level bundle integrity manifest."""

    bundle_version: str
    created_at_utc: str
    source_repository: str
    source_repository_path: str
    source_git_commit: str | None
    source_git_dirty: bool | None
    export_status: str
    integration_strategy: str
    inference_available: bool
    compatibility_warnings: list[str]
    manifest_self_checksum: str | None
    manifest_self_checksum_reason: str
    files: list[ManifestFile]


class ConfusionMatrix(StrictSchema):
    """Binary confusion-matrix counts."""

    true_negative: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    true_positive: int = Field(ge=0)


class ThresholdMetrics(StrictSchema):
    """Reference metrics for one decision threshold."""

    threshold: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    positive_rate: float = Field(ge=0.0, le=1.0)
    predicted_positive_count: int = Field(ge=0)
    predicted_positive_rate: float = Field(ge=0.0, le=1.0)
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    roc_auc: float | None
    pr_auc: float | None
    brier_score: float | None
    confusion_matrix: ConfusionMatrix
    undefined_metrics: list[str]


class ReferenceMetrics(StrictSchema):
    """Calculated reference metrics plus comparison with source reports."""

    metrics_version: str
    reference_split: str
    sample_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    positive_rate: float = Field(ge=0.0, le=1.0)
    thresholds: dict[str, ThresholdMetrics]
    authoritative_source_metrics: dict[str, dict[str, float]]
    metric_absolute_differences: dict[str, dict[str, float]]
    material_discrepancies: list[str]
    comparison_tolerance: float = Field(ge=0.0)


class ValidationIssue(StrictSchema):
    """One structured validation error or warning."""

    check: str
    message: str


class BundleValidationResult(StrictSchema):
    """Structured result returned by complete bundle validation."""

    valid: bool
    checks: dict[str, bool]
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]
    details: dict[str, Any]
