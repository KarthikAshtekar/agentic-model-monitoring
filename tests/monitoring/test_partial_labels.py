"""Absent and partial label behavior in deterministic monitoring."""

from __future__ import annotations

import pandas as pd

from monitoring_agent.monitoring.engine import MonitoringEngine
from monitoring_agent.paths import PROJECT_ROOT


def _normal_features_and_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_dir = PROJECT_ROOT / "data/scenarios/normal_operation"
    return (
        pd.read_parquet(scenario_dir / "features.parquet"),
        pd.read_parquet(scenario_dir / "labels.parquet"),
    )


def test_partial_aligned_labels_abstain_below_minimum() -> None:
    features, labels = _normal_features_and_labels()
    result = MonitoringEngine().run_batch(
        "insufficient_labels",
        features,
        labels.iloc[:100].reset_index(drop=True),
    )

    assert result.feature_row_count == 1000
    assert result.labelled_row_count == 100
    assert result.label_coverage_rate == 0.1
    assert result.minimum_labelled_sample_size == 200
    assert result.performance.evaluated is False
    assert "at least 200" in result.performance.reason_not_evaluated
    assert result.incident_candidates == ["insufficient_evidence"]


def test_missing_labels_preserve_drift_and_never_claim_performance() -> None:
    features, _ = _normal_features_and_labels()
    shifted = features.copy()
    shifted["LIMIT_BAL"] = (shifted["LIMIT_BAL"] * 0.55).clip(10000, 750000)
    result = MonitoringEngine().run_batch("unlabelled_drift", shifted, None)

    assert result.batch_valid is True
    assert result.drift.evaluated_feature_count > 0
    assert result.performance.evaluated is False
    assert "performance_degradation" not in result.incident_candidates
    assert "insufficient_evidence" in result.incident_candidates


def test_labels_outside_feature_batch_are_invalid() -> None:
    features, labels = _normal_features_and_labels()
    partial = labels.iloc[:200].copy()
    partial.loc[partial.index[0], "record_id"] = (
        int(features["record_id"].max()) + 1_000_000
    )
    result = MonitoringEngine().run_batch("invalid_label_alignment", features, partial)

    assert result.performance.evaluated is False
    assert "not present in the feature batch" in result.performance.reason_not_evaluated
