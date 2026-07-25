"""Protocol shared by deterministic model adapters."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class ModelAdapter(Protocol):
    def validate_features(self, features: pd.DataFrame) -> None: ...

    def predict_scores(self, features: pd.DataFrame) -> pd.Series: ...

    def apply_threshold(
        self,
        scores: pd.Series,
        threshold: float,
    ) -> pd.Series: ...

    def validate_labels(self, labels: pd.Series) -> None: ...

    def calculate_performance(
        self,
        labels: pd.Series,
        scores: pd.Series,
        thresholds: list[float],
    ) -> dict[str, Any]: ...
