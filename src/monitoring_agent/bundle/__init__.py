"""Compatibility namespace for registered-model bundle loading."""

from __future__ import annotations

from typing import Any

__all__ = ["BundleInputError", "CreditDefaultBundle", "validate_bundle"]


def __getattr__(name: str) -> Any:
    if name in {"BundleInputError", "CreditDefaultBundle"}:
        from monitoring_agent.bundle.loader import BundleInputError, CreditDefaultBundle

        return {
            "BundleInputError": BundleInputError,
            "CreditDefaultBundle": CreditDefaultBundle,
        }[name]
    if name == "validate_bundle":
        from monitoring_agent.bundle.validation import validate_bundle

        return validate_bundle
    raise AttributeError(name)
