"""Manifest strictness and explicit source-contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from monitoring_agent.models.registry import ModelRegistry
from monitoring_agent.onboarding.bundle_builder import (
    PROBABILITY_REPRODUCTION_TOLERANCE,
)


def test_diabetes_manifest_uses_verified_binary_contract() -> None:
    manifest = ModelRegistry().load_manifest("diabetes_risk")

    assert len(manifest.data_contract.ordered_features) == 21
    assert manifest.data_contract.preprocessed_feature_count == 52
    assert manifest.data_contract.target_column == "Diabetes_binary"
    assert manifest.prediction_contract.positive_class == 1
    assert manifest.prediction_contract.operating_threshold == 0.25
    assert manifest.provenance.bundle_mode == "live_inference"
    assert all(
        not path.startswith(("D:/", "D:\\"))
        for path in manifest.provenance.source_artifact_paths
    )


def test_manifest_rejects_extra_fields() -> None:
    manifest = ModelRegistry().load_manifest("credit_default")
    payload = manifest.model_dump()
    payload["identity"]["unsupported"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(manifest).model_validate(payload)


def test_probability_reproduction_tolerance_remains_sub_micro() -> None:
    assert 0.0 < PROBABILITY_REPRODUCTION_TOLERANCE < 1e-6
