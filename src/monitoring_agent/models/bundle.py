"""Generic registered-model bundle loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from monitoring_agent.adapters.binary_classification import (
    AdapterInputError,
    BinaryClassificationAdapter,
)
from monitoring_agent.bundle.schemas import (
    BundleManifest,
    FeatureSchema,
    ModelMetadata,
    ReferenceMetrics,
)
from monitoring_agent.models.manifest import RegisteredModelManifest
from monitoring_agent.models.registry import ModelRegistry
from monitoring_agent.paths import PROJECT_ROOT

BundleInputError = AdapterInputError


class RegisteredModelBundle:
    """Load a manifest-backed, self-contained binary-classification bundle."""

    def __init__(
        self,
        model_id: str | None = None,
        *,
        project_root: Path | str = PROJECT_ROOT,
        manifest: RegisteredModelManifest | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.registry = registry or ModelRegistry()
        self.registered_manifest = manifest or self.registry.load_manifest(model_id)
        paths = self.registered_manifest.bundle_paths
        self.model_path = self._path(paths.model_artifact)
        self.metadata_path = self._path_from_manifest_neighbor(
            paths.feature_schema,
            "model_metadata.json",
        )
        self.feature_schema_path = self._path(paths.feature_schema)
        self.manifest_path = self._path(paths.bundle_manifest)
        self.reference_metrics_path = self._path(paths.reference_metrics)
        self.reference_features_path = self._path(paths.reference_features)
        self.reference_labels_path = self._path(paths.reference_labels)
        self.reference_predictions_path = self._path(paths.reference_predictions)
        self.reference_feature_summary_path = self._path(
            paths.reference_feature_summary
        )
        self._model: Any | None = None
        self._adapter: BinaryClassificationAdapter | None = None

    def _path(self, relative: str | None) -> Path | None:
        if relative is None:
            return None
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Bundle path escapes the project root.") from exc
        return candidate

    def _path_from_manifest_neighbor(self, relative: str, name: str) -> Path:
        path = self._path(relative)
        if path is None:
            raise ValueError("Feature schema path cannot be null.")
        return path.with_name(name)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected a JSON object in {path}.")
        return payload

    @property
    def model_id(self) -> str:
        return self.registered_manifest.model_id

    @property
    def inference_available(self) -> bool:
        return self.registered_manifest.inference_available

    def load_metadata(self) -> ModelMetadata:
        return ModelMetadata.model_validate(self._load_json(self.metadata_path))

    def load_feature_schema(self) -> FeatureSchema:
        return FeatureSchema.model_validate(self._load_json(self.feature_schema_path))

    def load_manifest(self) -> BundleManifest:
        return BundleManifest.model_validate(self._load_json(self.manifest_path))

    def load_reference_metrics(self) -> ReferenceMetrics:
        return ReferenceMetrics.model_validate(
            self._load_json(self.reference_metrics_path)
        )

    def load_inference_artifact(self) -> Any:
        if not self.inference_available or self.model_path is None:
            raise RuntimeError(
                "The registered model marks live model inference as unavailable."
            )
        if self._model is None:
            self._model = joblib.load(self.model_path)
        return self._model

    def adapter(self) -> BinaryClassificationAdapter:
        if self._adapter is None:
            model = self.load_inference_artifact() if self.inference_available else None
            self._adapter = BinaryClassificationAdapter(
                self.registered_manifest,
                model,
            )
        return self._adapter

    def load_reference_features(self) -> pd.DataFrame:
        return pd.read_parquet(self.reference_features_path)

    def load_reference_labels(self) -> pd.DataFrame:
        return pd.read_parquet(self.reference_labels_path)

    def load_reference_predictions(self) -> pd.DataFrame:
        return pd.read_parquet(self.reference_predictions_path)

    def load_reference_feature_summary(self) -> pd.DataFrame:
        return pd.read_parquet(self.reference_feature_summary_path)

    def ordered_inference_features(self) -> list[str]:
        return list(self.registered_manifest.data_contract.ordered_features)

    def predict_scores(self, frame: pd.DataFrame) -> pd.Series:
        return self.adapter().predict_scores(frame)

    def predict_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        """Backward-compatible NumPy probability output."""
        return self.predict_scores(frame).to_numpy(dtype=float)


BinaryClassificationBundle = RegisteredModelBundle


class CreditDefaultBundle(RegisteredModelBundle):
    """Backward-compatible wrapper for the registered credit-default model."""

    def __init__(self, bundle_root: Path | str | None = None) -> None:
        registry = ModelRegistry()
        super().__init__(
            project_root=bundle_root or PROJECT_ROOT,
            manifest=registry.load_manifest("credit_default"),
            registry=registry,
        )
