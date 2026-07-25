"""Self-contained loader for the exported credit-default monitoring bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from monitoring_agent.bundle.schemas import (
    BundleManifest,
    FeatureSchema,
    ModelMetadata,
    ReferenceMetrics,
)
from monitoring_agent.paths import PROJECT_ROOT


class BundleInputError(ValueError):
    """Raised when an inference frame violates the exported schema."""


class CreditDefaultBundle:
    """Load bundle artifacts without importing the source model repository."""

    def __init__(self, bundle_root: Path | str | None = None) -> None:
        self.root = Path(bundle_root or PROJECT_ROOT).resolve()
        self.model_path = self.root / "artifacts/models/credit_default_pipeline.joblib"
        self.metadata_path = self.root / "artifacts/metadata/model_metadata.json"
        self.feature_schema_path = self.root / "artifacts/metadata/feature_schema.json"
        self.manifest_path = self.root / "artifacts/metadata/bundle_manifest.json"
        self.reference_metrics_path = (
            self.root / "artifacts/baselines/reference_metrics.json"
        )
        self.reference_features_path = self.root / "data/reference/reference_features.parquet"
        self.reference_labels_path = self.root / "data/reference/reference_labels.parquet"
        self.reference_predictions_path = (
            self.root / "data/reference/reference_predictions.parquet"
        )
        self.reference_feature_summary_path = (
            self.root / "artifacts/baselines/reference_feature_summary.parquet"
        )
        self._model: Any | None = None

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected a JSON object in {path}.")
        return payload

    def load_metadata(self) -> ModelMetadata:
        """Load and validate model metadata."""
        return ModelMetadata.model_validate(self._load_json(self.metadata_path))

    def load_feature_schema(self) -> FeatureSchema:
        """Load and validate the ordered feature contract."""
        return FeatureSchema.model_validate(self._load_json(self.feature_schema_path))

    def load_manifest(self) -> BundleManifest:
        """Load and validate the integrity manifest."""
        return BundleManifest.model_validate(self._load_json(self.manifest_path))

    def load_reference_metrics(self) -> ReferenceMetrics:
        """Load and validate calculated reference metrics."""
        return ReferenceMetrics.model_validate(self._load_json(self.reference_metrics_path))

    def load_inference_artifact(self) -> Any:
        """Load the exported fitted pipeline, preserving compatibility warnings."""
        metadata = self.load_metadata()
        if not metadata.inference_available or metadata.inference_artifact_path is None:
            raise RuntimeError("The bundle explicitly marks live model inference as unavailable.")
        if self._model is None:
            artifact_path = self.root / metadata.inference_artifact_path
            self._model = joblib.load(artifact_path)
        return self._model

    def load_reference_features(self) -> pd.DataFrame:
        """Load reference identifiers and ordered inference features."""
        return pd.read_parquet(self.reference_features_path)

    def load_reference_labels(self) -> pd.DataFrame:
        """Load reference identifiers and actual labels."""
        return pd.read_parquet(self.reference_labels_path)

    def load_reference_predictions(self) -> pd.DataFrame:
        """Load reference probabilities and thresholded predictions."""
        return pd.read_parquet(self.reference_predictions_path)

    def load_reference_feature_summary(self) -> pd.DataFrame:
        """Load the one-row-per-feature reference summary."""
        return pd.read_parquet(self.reference_feature_summary_path)

    def ordered_inference_features(self) -> list[str]:
        """Return the exact feature order required by the fitted pipeline."""
        return self.load_feature_schema().ordered_features

    def _validate_input_frame(self, frame: pd.DataFrame) -> None:
        schema = self.load_feature_schema()
        target_names = {schema.target.name, schema.target.exported_name}
        leaked_targets = [column for column in frame.columns if column in target_names]
        if leaked_targets:
            raise BundleInputError(
                "Inference input contains target data: " + ", ".join(leaked_targets)
            )

        expected = schema.ordered_features
        actual = frame.columns.tolist()
        if actual == expected:
            return

        missing = [column for column in expected if column not in actual]
        extra = [column for column in actual if column not in expected]
        if not missing and not extra:
            raise BundleInputError(
                "Inference feature order does not match the exported schema. "
                "Columns must be supplied in the exact recorded order."
            )
        raise BundleInputError(
            "Inference columns do not match the exported schema. "
            f"Missing={missing or 'none'}; extra={extra or 'none'}."
        )

    def predict_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        """Generate positive-class probabilities for an exactly compatible frame."""
        self._validate_input_frame(frame)
        model = self.load_inference_artifact()
        probabilities = np.asarray(model.predict_proba(frame), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            raise RuntimeError("Inference artifact did not return binary class probabilities.")
        positive_probabilities = probabilities[:, 1]
        if not np.isfinite(positive_probabilities).all():
            raise RuntimeError("Inference artifact returned non-finite probabilities.")
        return positive_probabilities
