"""Registered binary-classification model contracts and loading."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BinaryClassificationBundle",
    "ModelRegistry",
    "RegisteredModelBundle",
    "RegisteredModelManifest",
]


def __getattr__(name: str) -> Any:
    if name in {"BinaryClassificationBundle", "RegisteredModelBundle"}:
        from monitoring_agent.models.bundle import (
            BinaryClassificationBundle,
            RegisteredModelBundle,
        )

        return {
            "BinaryClassificationBundle": BinaryClassificationBundle,
            "RegisteredModelBundle": RegisteredModelBundle,
        }[name]
    if name == "RegisteredModelManifest":
        from monitoring_agent.models.manifest import RegisteredModelManifest

        return RegisteredModelManifest
    if name == "ModelRegistry":
        from monitoring_agent.models.registry import ModelRegistry

        return ModelRegistry
    raise AttributeError(name)
