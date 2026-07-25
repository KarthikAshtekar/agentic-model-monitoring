"""Deterministic labelled-performance evaluation against bundle baselines."""

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

from monitoring_agent.bundle.schemas import ModelMetadata, ReferenceMetrics
from monitoring_agent.monitoring.evidence import EvidenceRegistry
from monitoring_agent.monitoring.schemas import EvidenceItem, PerformanceResult


def _threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype("int8")
    predicted_positive_count = int(predictions.sum())
    positive_count = int(labels.sum())
    precision = (
        float(precision_score(labels, predictions))
        if predicted_positive_count
        else None
    )
    recall = float(recall_score(labels, predictions)) if positive_count else None
    f1 = (
        float(f1_score(labels, predictions))
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": float(predicted_positive_count / len(labels)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _not_evaluated(
    reason: str,
    evidence_id: str,
    *,
    sample_count: int = 0,
    positive_count: int = 0,
    negative_count: int = 0,
    feature_row_count: int = 0,
    minimum_required_sample_size: int = 0,
) -> PerformanceResult:
    evidence = EvidenceItem(
        evidence_id=evidence_id,
        domain="system",
        metric="performance_evaluation_available",
        status="not_evaluated",
        severity="medium",
        observed_value=False,
        reference_value=True,
        threshold=None,
        feature=None,
        message=reason,
        source="performance_preconditions",
    )
    return PerformanceResult(
        evaluated=False,
        reason_not_evaluated=reason,
        sample_count=sample_count,
        positive_count=positive_count,
        negative_count=negative_count,
        feature_row_count=feature_row_count,
        labelled_row_count=sample_count,
        label_coverage_rate=(
            sample_count / feature_row_count if feature_row_count else 0.0
        ),
        minimum_required_sample_size=minimum_required_sample_size,
        metrics_at_default_threshold={},
        metrics_at_operating_threshold={},
        metric_deltas={},
        evidence=[evidence],
    )


def _drop_level(drop: float, warning: float, critical: float) -> str:
    if drop >= critical:
        return "critical"
    if drop >= warning:
        return "warning"
    return "pass"


def _severity(level: str) -> str:
    return {"pass": "info", "warning": "medium", "critical": "high"}[level]


def evaluate_performance(
    features: pd.DataFrame,
    labels_frame: pd.DataFrame | None,
    probabilities: np.ndarray,
    metadata: ModelMetadata,
    reference_metrics: ReferenceMetrics,
    config: dict[str, Any],
) -> PerformanceResult:
    """Evaluate labelled batch performance only when all preconditions hold."""
    feature_row_count = len(features)
    minimum_samples = int(config["minimum_labelled_samples"])
    if labels_frame is None:
        return _not_evaluated(
            "Labels are not available for this batch.",
            "SYSTEM-LABELS-UNAVAILABLE",
            feature_row_count=feature_row_count,
            minimum_required_sample_size=minimum_samples,
        )
    labelled_row_count = len(labels_frame)
    if labels_frame.columns.tolist() != ["record_id", "actual_label"]:
        return _not_evaluated(
            "Label columns do not match record_id and actual_label.",
            "SYSTEM-INVALID-LABEL-SCHEMA",
            sample_count=labelled_row_count,
            feature_row_count=feature_row_count,
            minimum_required_sample_size=minimum_samples,
        )
    if not labels_frame["record_id"].is_unique:
        return _not_evaluated(
            "Label record IDs must be unique.",
            "SYSTEM-LABEL-ALIGNMENT",
            sample_count=labelled_row_count,
            feature_row_count=feature_row_count,
            minimum_required_sample_size=minimum_samples,
        )
    feature_ids = set(features["record_id"].tolist())
    unknown_label_ids = [
        record_id
        for record_id in labels_frame["record_id"].tolist()
        if record_id not in feature_ids
    ]
    if unknown_label_ids:
        return _not_evaluated(
            "One or more label record IDs are not present in the feature batch.",
            "SYSTEM-LABEL-ALIGNMENT",
            sample_count=labelled_row_count,
            feature_row_count=feature_row_count,
            minimum_required_sample_size=minimum_samples,
        )

    raw_labels = labels_frame["actual_label"]
    supported = raw_labels.dropna().isin([0, 1]).all() and not raw_labels.isna().any()
    if not supported:
        return _not_evaluated(
            "Labels must be non-null binary values 0 or 1.",
            "SYSTEM-INVALID-LABELS",
            sample_count=len(raw_labels),
            feature_row_count=feature_row_count,
            minimum_required_sample_size=minimum_samples,
        )

    labels = raw_labels.to_numpy(dtype="int8")
    sample_count = len(labels)
    positive_count = int(labels.sum())
    negative_count = sample_count - positive_count
    if sample_count < minimum_samples:
        return _not_evaluated(
            f"Only {sample_count} labels are available; at least {minimum_samples} are required.",
            "SYSTEM-INSUFFICIENT-LABELS",
            sample_count=sample_count,
            positive_count=positive_count,
            negative_count=negative_count,
            feature_row_count=feature_row_count,
            minimum_required_sample_size=minimum_samples,
        )
    if positive_count == 0 or negative_count == 0:
        return _not_evaluated(
            "Both label classes are required for ROC-AUC and PR-AUC.",
            "SYSTEM-SINGLE-LABEL-CLASS",
            sample_count=sample_count,
            positive_count=positive_count,
            negative_count=negative_count,
            feature_row_count=feature_row_count,
            minimum_required_sample_size=minimum_samples,
        )

    probability_by_id = pd.Series(
        probabilities,
        index=features["record_id"].tolist(),
        dtype=float,
    )
    aligned_probabilities = probability_by_id.loc[
        labels_frame["record_id"].tolist()
    ].to_numpy(dtype=float)
    default_metrics = _threshold_metrics(
        labels,
        aligned_probabilities,
        metadata.default_threshold,
    )
    operating_metrics = _threshold_metrics(
        labels,
        aligned_probabilities,
        metadata.operating_threshold,
    )
    ranking_metrics = {
        "roc_auc": float(roc_auc_score(labels, aligned_probabilities)),
        "pr_auc": float(average_precision_score(labels, aligned_probabilities)),
        "brier_score": float(brier_score_loss(labels, aligned_probabilities)),
    }
    default_metrics.update(ranking_metrics)
    operating_metrics.update(ranking_metrics)

    reference_default = reference_metrics.thresholds["default_0_50"]
    reference_operating = reference_metrics.thresholds["operating_0_25"]
    deltas = {
        "default_threshold": {
            metric: (
                None
                if default_metrics[metric] is None
                else float(getattr(reference_default, metric) - default_metrics[metric])
            )
            for metric in ("accuracy", "precision", "recall", "f1")
        },
        "operating_threshold": {
            metric: (
                None
                if operating_metrics[metric] is None
                else float(getattr(reference_operating, metric) - operating_metrics[metric])
            )
            for metric in ("accuracy", "precision", "recall", "f1")
        },
        "roc_auc_drop": float(reference_operating.roc_auc - ranking_metrics["roc_auc"]),
        "pr_auc_drop": float(reference_operating.pr_auc - ranking_metrics["pr_auc"]),
        "brier_score_increase": float(
            ranking_metrics["brier_score"] - reference_operating.brier_score
        ),
    }

    registry = EvidenceRegistry()
    operating_rules = {
        "recall": ("PERF-RECALL-OPERATING", "recall_drop"),
        "precision": ("PERF-PRECISION-OPERATING", "precision_drop"),
        "f1": ("PERF-F1-OPERATING", "f1_drop"),
    }
    for metric, (item_id, threshold_prefix) in operating_rules.items():
        drop = deltas["operating_threshold"][metric]
        warning_threshold = float(config[f"{threshold_prefix}_warning"])
        critical_threshold = float(config[f"{threshold_prefix}_critical"])
        level = (
            "not_evaluated"
            if drop is None
            else _drop_level(drop, warning_threshold, critical_threshold)
        )
        registry.add(
            EvidenceItem(
                evidence_id=item_id,
                domain="performance",
                metric=f"{metric}_drop_operating_threshold",
                status=level,
                severity="medium" if level == "not_evaluated" else _severity(level),
                observed_value=operating_metrics[metric],
                reference_value=getattr(reference_operating, metric),
                threshold={
                    "warning_drop": warning_threshold,
                    "critical_drop": critical_threshold,
                },
                feature=None,
                message=(
                    f"Operating-threshold {metric} is {operating_metrics[metric]:.4f}; "
                    f"drop from reference is {drop:.4f}."
                    if drop is not None
                    else f"Operating-threshold {metric} is undefined."
                ),
                source="reference_metrics.json",
            )
        )

    for metric, item_id, delta_key, threshold_prefix in (
        ("pr_auc", "PERF-PRAUC", "pr_auc_drop", "pr_auc_drop"),
        ("roc_auc", "PERF-ROCAUC", "roc_auc_drop", "roc_auc_drop"),
    ):
        drop = float(deltas[delta_key])
        warning_threshold = float(config[f"{threshold_prefix}_warning"])
        critical_threshold = float(config[f"{threshold_prefix}_critical"])
        level = _drop_level(drop, warning_threshold, critical_threshold)
        registry.add(
            EvidenceItem(
                evidence_id=item_id,
                domain="performance",
                metric=f"{metric}_drop",
                status=level,
                severity=_severity(level),
                observed_value=ranking_metrics[metric],
                reference_value=getattr(reference_operating, metric),
                threshold={
                    "warning_drop": warning_threshold,
                    "critical_drop": critical_threshold,
                },
                feature=None,
                message=(
                    f"{metric.upper()} is {ranking_metrics[metric]:.4f}; "
                    f"drop from reference is {drop:.4f}."
                ),
                source="reference_metrics.json",
            )
        )

    brier_increase = float(deltas["brier_score_increase"])
    brier_level = _drop_level(
        brier_increase,
        float(config["brier_increase_warning"]),
        float(config["brier_increase_critical"]),
    )
    registry.add(
        EvidenceItem(
            evidence_id="PERF-BRIER",
            domain="performance",
            metric="brier_score_increase",
            status=brier_level,
            severity=_severity(brier_level),
            observed_value=ranking_metrics["brier_score"],
            reference_value=reference_operating.brier_score,
            threshold={
                "warning_increase": config["brier_increase_warning"],
                "critical_increase": config["brier_increase_critical"],
            },
            feature=None,
            message=(
                f"Brier score is {ranking_metrics['brier_score']:.4f}; "
                f"increase from reference is {brier_increase:.4f}."
            ),
            source="reference_metrics.json",
        )
    )

    return PerformanceResult(
        evaluated=True,
        reason_not_evaluated=None,
        sample_count=sample_count,
        positive_count=positive_count,
        negative_count=negative_count,
        feature_row_count=feature_row_count,
        labelled_row_count=sample_count,
        label_coverage_rate=sample_count / feature_row_count,
        minimum_required_sample_size=minimum_samples,
        metrics_at_default_threshold=default_metrics,
        metrics_at_operating_threshold=operating_metrics,
        metric_deltas=deltas,
        evidence=registry.items,
    )
