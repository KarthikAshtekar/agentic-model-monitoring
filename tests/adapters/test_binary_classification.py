"""Reusable binary-classification adapter behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring_agent.adapters.binary_classification import (
    AdapterInputError,
    BinaryClassificationAdapter,
)
from monitoring_agent.models.registry import ModelRegistry


class _TwoColumnProbabilityModel:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = features["LIMIT_BAL"].to_numpy(dtype=float)
        positive = (positive - positive.min()) / max(np.ptp(positive), 1.0)
        return np.column_stack([1.0 - positive, positive])


@pytest.fixture
def adapter() -> BinaryClassificationAdapter:
    return BinaryClassificationAdapter(
        ModelRegistry().load_manifest("credit_default"),
        _TwoColumnProbabilityModel(),
    )


def _features() -> pd.DataFrame:
    manifest = ModelRegistry().load_manifest("credit_default")
    frame = pd.DataFrame(
        np.zeros((4, len(manifest.data_contract.ordered_features))),
        columns=manifest.data_contract.ordered_features,
    )
    frame["LIMIT_BAL"] = [0.0, 1.0, 2.0, 3.0]
    return frame


def test_scores_and_thresholds_are_bounded(adapter: BinaryClassificationAdapter) -> None:
    scores = adapter.predict_scores(_features())

    assert scores.between(0.0, 1.0).all()
    assert adapter.apply_threshold(scores, 0.5).tolist() == [0, 0, 1, 1]


def test_wrong_feature_order_and_target_leakage_are_rejected(
    adapter: BinaryClassificationAdapter,
) -> None:
    frame = _features()
    columns = frame.columns.tolist()
    columns[:2] = reversed(columns[:2])
    with pytest.raises(AdapterInputError, match="feature order"):
        adapter.validate_features(frame[columns])

    frame["Default_Flag"] = 0
    with pytest.raises(AdapterInputError, match="target data"):
        adapter.validate_features(frame)


def test_binary_metrics_and_undefined_metrics_are_transparent(
    adapter: BinaryClassificationAdapter,
) -> None:
    scores = pd.Series([0.1, 0.4, 0.6, 0.9])
    metrics = adapter.calculate_performance(
        pd.Series([0, 0, 1, 1]),
        scores,
        [0.5],
    )["0.5"]

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["true_positive"] == 2
    single_class = adapter.calculate_performance(
        pd.Series([0, 0, 0, 0]),
        scores,
        [0.5],
    )["0.5"]
    assert single_class["roc_auc"] is None
    assert "roc_auc" in single_class["undefined_metrics"]


def test_missing_or_nonbinary_labels_are_rejected(
    adapter: BinaryClassificationAdapter,
) -> None:
    with pytest.raises(AdapterInputError, match="non-null"):
        adapter.validate_labels(pd.Series([0, None]))
    with pytest.raises(AdapterInputError, match="0/1"):
        adapter.validate_labels(pd.Series([0, 2]))
