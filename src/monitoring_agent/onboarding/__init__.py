"""Deterministic binary-classifier inspection and onboarding."""

from __future__ import annotations

from typing import Any

__all__ = ["inspect_model_project", "onboard_binary_classifier"]


def __getattr__(name: str) -> Any:
    if name == "inspect_model_project":
        from monitoring_agent.onboarding.inspector import inspect_model_project

        return inspect_model_project
    if name == "onboard_binary_classifier":
        from monitoring_agent.onboarding.bundle_builder import (
            onboard_binary_classifier,
        )

        return onboard_binary_classifier
    raise AttributeError(name)
