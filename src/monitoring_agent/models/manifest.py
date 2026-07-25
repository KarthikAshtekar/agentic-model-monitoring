"""Strict registered-model manifest contracts."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskType = Literal["binary_classification"]
AdapterType = Literal["sklearn_binary_classifier"]
ScoreMethod = Literal[
    "predict_proba",
    "decision_function_transformed",
    "precomputed_scores",
]
BundleMode = Literal["live_inference", "scored_reference_only"]


class ManifestSection(BaseModel):
    """Reject undocumented fields in every manifest section."""

    model_config = ConfigDict(extra="forbid")


class ModelIdentity(ManifestSection):
    model_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    task_type: TaskType
    adapter_type: AdapterType
    domain_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")


class BusinessContext(ManifestSection):
    use_case: str | None
    prediction_unit: str | None
    positive_outcome: str | None
    prediction_horizon: str | None
    decision_context: str | None
    business_owner: str | None


class BundlePaths(ManifestSection):
    model_artifact: str | None
    feature_schema: str
    reference_features: str
    reference_labels: str
    reference_predictions: str
    reference_metrics: str
    reference_feature_summary: str
    bundle_manifest: str

    @field_validator("*")
    @classmethod
    def paths_must_be_repository_relative(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalised = value.replace("\\", "/")
        if (
            PurePosixPath(normalised).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or ".." in PurePosixPath(normalised).parts
        ):
            raise ValueError("Bundle paths must be repository-relative and cannot escape.")
        return normalised


class PredictionContract(ManifestSection):
    score_method: ScoreMethod
    positive_class: int | str
    positive_class_index: int = Field(ge=0)
    default_threshold: float = Field(ge=0.0, le=1.0)
    operating_threshold: float = Field(ge=0.0, le=1.0)


class DataContract(ManifestSection):
    identifier_column: str
    target_column: str
    ordered_features: list[str] = Field(min_length=1)
    numeric_features: list[str]
    categorical_features: list[str]
    integer_constrained_features: list[str]
    nullable_features: list[str]
    preprocessed_feature_count: int = Field(ge=1)
    preprocessing_representation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_feature_groups(self) -> DataContract:
        ordered = self.ordered_features
        if len(ordered) != len(set(ordered)):
            raise ValueError("ordered_features contains duplicates.")
        known = set(ordered)
        for field_name in (
            "numeric_features",
            "categorical_features",
            "integer_constrained_features",
            "nullable_features",
        ):
            values = getattr(self, field_name)
            unknown = sorted(set(values) - known)
            if unknown:
                raise ValueError(f"{field_name} contains unknown features: {unknown}")
        overlap = set(self.numeric_features) & set(self.categorical_features)
        if overlap:
            raise ValueError(
                "numeric_features and categorical_features must not overlap: "
                f"{sorted(overlap)}"
            )
        covered = set(self.numeric_features) | set(self.categorical_features)
        if covered != known:
            raise ValueError(
                "numeric_features and categorical_features must cover ordered_features."
            )
        return self


class MonitoringPolicy(ManifestSection):
    minimum_batch_size: int = Field(ge=1)
    minimum_labelled_samples: int = Field(ge=1)
    primary_metrics: list[str] = Field(min_length=1)
    feature_drift_thresholds: dict[str, float | int]
    prediction_drift_thresholds: dict[str, float]
    performance_thresholds: dict[str, float]


class Governance(ManifestSection):
    require_human_approval: bool
    protected_attributes: list[str]
    allowed_action_types: list[str] = Field(min_length=1)
    prohibited_automatic_actions: list[str] = Field(min_length=1)


class Provenance(ManifestSection):
    source_project: str = Field(min_length=1)
    source_commit: str | None
    source_worktree_clean: bool | None
    source_artifact_paths: list[str] = Field(min_length=1)
    bundle_created_at_utc: str
    bundle_mode: BundleMode
    limitations: list[str]

    @field_validator("source_artifact_paths")
    @classmethod
    def source_artifacts_must_not_be_runtime_absolute(
        cls,
        values: list[str],
    ) -> list[str]:
        for value in values:
            if PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute():
                raise ValueError(
                    "source_artifact_paths must be source-repository-relative paths."
                )
        return [value.replace("\\", "/") for value in values]


class RegisteredModelManifest(BaseModel):
    """Complete explicit contract for one registered model."""

    model_config = ConfigDict(extra="forbid")

    identity: ModelIdentity
    business_context: BusinessContext
    bundle_paths: BundlePaths
    prediction_contract: PredictionContract
    data_contract: DataContract
    monitoring_policy: MonitoringPolicy
    governance: Governance
    provenance: Provenance

    @model_validator(mode="after")
    def validate_bundle_mode(self) -> RegisteredModelManifest:
        live = self.provenance.bundle_mode == "live_inference"
        if live and self.bundle_paths.model_artifact is None:
            raise ValueError("live_inference requires model_artifact.")
        if (
            self.prediction_contract.score_method == "precomputed_scores"
            and live
        ):
            raise ValueError("precomputed_scores cannot declare live_inference.")
        if self.data_contract.target_column in self.data_contract.ordered_features:
            raise ValueError("The target column cannot be an inference feature.")
        if self.data_contract.identifier_column in self.data_contract.ordered_features:
            raise ValueError("The identifier column cannot be an inference feature.")
        return self

    @property
    def model_id(self) -> str:
        return self.identity.model_id

    @property
    def inference_available(self) -> bool:
        return self.provenance.bundle_mode == "live_inference"
