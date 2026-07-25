"""Registered bundle validation entry point for onboarding."""

from monitoring_agent.bundle.validation import validate_bundle
from monitoring_agent.models.bundle import RegisteredModelBundle

__all__ = ["RegisteredModelBundle", "validate_bundle"]
