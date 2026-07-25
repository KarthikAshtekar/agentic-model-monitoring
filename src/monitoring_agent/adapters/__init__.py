"""Model-adapter interfaces and binary-classification implementation."""

from monitoring_agent.adapters.base import ModelAdapter
from monitoring_agent.adapters.binary_classification import (
    BinaryClassificationAdapter,
    CalibratedBinaryClassifier,
)

__all__ = [
    "BinaryClassificationAdapter",
    "CalibratedBinaryClassifier",
    "ModelAdapter",
]
