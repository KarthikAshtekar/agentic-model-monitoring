"""Reusable scikit-learn binary-classification adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from monitoring_agent.models.manifest import RegisteredModelManifest


class AdapterInputError(ValueError):
    """Raised when features or labels violate a registered contract."""


class CalibratedBinaryClassifier:
    """Portable fitted classifier combining a pipeline and sigmoid calibrator."""

    def __init__(
        self,
        base_estimator: Any,
        calibration_model: Any,
        *,
        positive_class_index: int = 1,
    ) -> None:
        self.base_estimator = base_estimator
        self.calibration_model = calibration_model
        self.positive_class_index = positive_class_index
        self.classes_ = np.asarray([0, 1])

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.base_estimator.predict_proba(features), dtype=float)
        positive_raw = raw[:, self.positive_class_index]
        positive = np.asarray(
            self.calibration_model.predict_proba(self._logit(positive_raw))[:, 1],
            dtype=float,
        )
        return np.column_stack([1.0 - positive, positive])


class BinaryClassificationAdapter:
    """Validate features and expose deterministic binary metrics."""

    def __init__(
        self,
        manifest: RegisteredModelManifest,
        model: Any | None,
    ) -> None:
        self.manifest = manifest
        self.model = model

    def validate_features(self, features: pd.DataFrame) -> None:
        contract = self.manifest.data_contract
        targets = {contract.target_column, "actual_label"}
        leaked = [column for column in features.columns if column in targets]
        if leaked:
            raise AdapterInputError(
                "Inference input contains target data: " + ", ".join(leaked)
            )
        expected = contract.ordered_features
        actual = features.columns.tolist()
        if actual == expected:
            return
        missing = [item for item in expected if item not in actual]
        extra = [item for item in actual if item not in expected]
        if not missing and not extra:
            raise AdapterInputError(
                "Inference feature order does not match the registered manifest."
            )
        raise AdapterInputError(
            "Inference columns do not match the registered manifest. "
            f"Missing={missing or 'none'}; extra={extra or 'none'}."
        )

    def predict_scores(self, features: pd.DataFrame) -> pd.Series:
        self.validate_features(features)
        method = self.manifest.prediction_contract.score_method
        if method == "precomputed_scores":
            raise RuntimeError(
                "Raw-batch inference is unavailable for a scored-reference-only model."
            )
        if self.model is None:
            raise RuntimeError("No fitted inference artifact is loaded.")
        if method == "predict_proba":
            output = np.asarray(self.model.predict_proba(features), dtype=float)
            index = self.manifest.prediction_contract.positive_class_index
            if output.ndim != 2 or index >= output.shape[1]:
                raise RuntimeError(
                    "Inference artifact did not return the configured probability column."
                )
            scores = output[:, index]
        else:
            decision = np.asarray(self.model.decision_function(features), dtype=float)
            scores = 1.0 / (1.0 + np.exp(-decision))
        if len(scores) != len(features):
            raise RuntimeError("Inference score count does not match feature rows.")
        if not np.isfinite(scores).all():
            raise RuntimeError("Inference artifact returned non-finite scores.")
        if not np.logical_and(scores >= 0.0, scores <= 1.0).all():
            raise RuntimeError("Inference scores must be within [0, 1].")
        return pd.Series(scores, index=features.index, name="predicted_probability")

    def apply_threshold(
        self,
        scores: pd.Series,
        threshold: float,
    ) -> pd.Series:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be within [0, 1].")
        return (scores >= threshold).astype("int8").rename("predicted_class")

    def validate_labels(self, labels: pd.Series) -> None:
        if labels.isna().any():
            raise AdapterInputError("Binary labels must be non-null.")
        allowed = {0, 1}
        actual = set(pd.to_numeric(labels, errors="coerce").dropna().unique())
        if not actual <= allowed or len(actual) > 2:
            raise AdapterInputError("Binary labels must use the exported 0/1 contract.")

    @staticmethod
    def _safe_ranking_metrics(
        labels: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[dict[str, float | None], list[str]]:
        undefined: list[str] = []
        if len(np.unique(labels)) < 2:
            roc_auc = None
            pr_auc = None
            undefined.extend(["roc_auc", "pr_auc"])
        else:
            roc_auc = float(roc_auc_score(labels, scores))
            pr_auc = float(average_precision_score(labels, scores))
        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": float(brier_score_loss(labels, scores)),
        }, undefined

    def calculate_performance(
        self,
        labels: pd.Series,
        scores: pd.Series,
        thresholds: list[float],
    ) -> dict[str, Any]:
        self.validate_labels(labels)
        if len(labels) != len(scores):
            raise AdapterInputError("Label and score row counts must align.")
        y_true = labels.to_numpy(dtype="int8")
        y_score = scores.to_numpy(dtype=float)
        ranking, ranking_undefined = self._safe_ranking_metrics(y_true, y_score)
        results: dict[str, Any] = {}
        for threshold in thresholds:
            predictions = (y_score >= threshold).astype("int8")
            positive_count = int(y_true.sum())
            predicted_positive_count = int(predictions.sum())
            precision = (
                float(precision_score(y_true, predictions, zero_division=0))
                if predicted_positive_count
                else None
            )
            recall = (
                float(recall_score(y_true, predictions, zero_division=0))
                if positive_count
                else None
            )
            f1 = (
                float(f1_score(y_true, predictions, zero_division=0))
                if precision is not None and recall is not None
                else None
            )
            tn, fp, fn, tp = confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1],
            ).ravel()
            undefined = list(ranking_undefined)
            if precision is None:
                undefined.append("precision")
            if recall is None:
                undefined.extend(["recall", "f1"])
            results[f"{threshold:.12g}"] = {
                "threshold": float(threshold),
                "sample_count": len(y_true),
                "positive_count": positive_count,
                "positive_rate": float(positive_count / len(y_true)),
                "predicted_positive_count": predicted_positive_count,
                "predicted_positive_rate": float(
                    predicted_positive_count / len(y_true)
                ),
                "accuracy": float(accuracy_score(y_true, predictions)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                **ranking,
                "confusion_matrix": {
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                },
                "undefined_metrics": list(dict.fromkeys(undefined)),
            }
        return results
