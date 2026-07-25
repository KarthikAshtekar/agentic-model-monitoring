"""Strict registry and credit migration contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from monitoring_agent.models.bundle import RegisteredModelBundle
from monitoring_agent.models.registry import ModelRegistry
from monitoring_agent.monitoring.schemas import MonitoringRunResult
from monitoring_agent.paths import PROJECT_ROOT

AUTHORITATIVE_HASHES = {
    "artifacts/models/credit_default_pipeline.joblib": (
        "ff04b857b2548174d88156332f5af8e135d50aacba4f5a47ca913c57a1097340"
    ),
    "artifacts/metadata/model_metadata.json": (
        "0e49cbf08d7fb6428125e7b07193cdca79dfd560fe8bcc37c4a6f03cf92c4668"
    ),
    "artifacts/metadata/feature_schema.json": (
        "9fa0de1701648d4352fcbd38f721e79df4c15017dc19f3edf3420ef272b08286"
    ),
    "artifacts/metadata/bundle_manifest.json": (
        "5a8feb5eaff711a86fed179105f0fcd93f507f7faf12abb9fa1f38d1fa42e677"
    ),
    "artifacts/baselines/reference_metrics.json": (
        "0be5ae84e6269a350628e2eda982c36ab9f0c4918185706d13adedc95f8d57a5"
    ),
    "artifacts/baselines/reference_feature_summary.parquet": (
        "d93aa710f52cf0c9786e650e981779fbfa5707cdfc8865a95f421251c55eb70b"
    ),
    "data/reference/reference_features.parquet": (
        "e43e0f1813c08ece1507af2b37b6f339ea3829907794a460553c4511148ea719"
    ),
    "data/reference/reference_labels.parquet": (
        "9b82c6718d827d8f741420d4afa9283ee6bcf946644d0bc20fb7eb232e8109bc"
    ),
    "data/reference/reference_predictions.parquet": (
        "16dea09d66ffef50285a77a09f5685fc110456e48ea9e66964cb6f49be750c23"
    ),
    "reports/evaluations/live_groq_six_scenarios/live_evaluation_summary.json": (
        "b4b89328bfe80ddc378a1774a146ff4faea13bdfa2335587d53cb9648eede7bd"
    ),
    "reports/evaluations/live_groq_six_scenarios/live_evaluation_report.md": (
        "54a56d110a2fcb26751215113b5826d5adce6bdf3b3c99ee11ae67e436af2b7a"
    ),
    "reports/evaluations/live_groq_six_scenarios/live_scenario_comparison.csv": (
        "e68fa1be62f5261736f967fbb524ec00103f842c958169be3d947eb8bd83370f"
    ),
}

SCENARIO_MANIFEST_HASHES = {
    "data/scenarios/data_quality_failure/scenario_manifest.json": (
        "de4926a1546387d75db3e26848900f6d03809584a9fabc4e401a036b1c6d386b"
    ),
    "data/scenarios/feature_drift/scenario_manifest.json": (
        "6b1312a71283417e8b73d2871bb46e6fd7dd2f983e71b0a108475615be0914f9"
    ),
    "data/scenarios/insufficient_labels/scenario_manifest.json": (
        "3fe34ac14df462166533a55811c9446292650896ebf98f2adb2ef24f69aae578"
    ),
    "data/scenarios/normal_operation/scenario_manifest.json": (
        "9a158607d4c1b3f1210a3702f47a6191b5f0b1d36d75041615e22b043659a9bb"
    ),
    "data/scenarios/performance_degradation/scenario_manifest.json": (
        "07ad6cecbfc8d90f6a45d52ed92a03f6ac728a00590b78b974de056717a5bbcd"
    ),
    "data/scenarios/unlabelled_drift/scenario_manifest.json": (
        "b7c3109047b94d04a27fdfdd52e2b8410749079725ca2d488a25910c9b300181"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_registry_lists_two_enabled_binary_models() -> None:
    registry = ModelRegistry()
    manifests = registry.list_enabled_models()

    assert registry.default_model_id == "credit_default"
    assert [item.model_id for item in manifests] == [
        "credit_default",
        "diabetes_risk",
    ]
    assert {item.identity.task_type for item in manifests} == {
        "binary_classification"
    }


def test_unknown_model_id_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown model_id"):
        ModelRegistry().load_manifest("not_registered")


def test_credit_manifest_preserves_validated_contract() -> None:
    bundle = RegisteredModelBundle("credit_default")
    manifest = bundle.registered_manifest
    stored = bundle.load_reference_predictions()["predicted_probability"].to_numpy()
    frame = bundle.load_reference_features()[bundle.ordered_inference_features()]

    assert len(manifest.data_contract.ordered_features) == 36
    assert manifest.prediction_contract.operating_threshold == 0.25
    assert bundle.load_reference_metrics().sample_count == 6002
    assert np.allclose(bundle.predict_probabilities(frame), stored, atol=1e-7, rtol=1e-7)


def test_credit_authoritative_artifacts_and_evaluation_are_byte_identical() -> None:
    for relative, expected in AUTHORITATIVE_HASHES.items():
        assert _sha256(PROJECT_ROOT / relative) == expected


def test_legacy_credit_scenario_manifests_are_byte_identical() -> None:
    assert _sha256(PROJECT_ROOT / "configs/scenarios.yaml") == (
        "604c368e09d06ea9a38cdc6cfdf44a8393ae95601d3f0f32ad8706f5bad0e6ef"
    )
    for relative, expected in SCENARIO_MANIFEST_HASHES.items():
        assert _sha256(PROJECT_ROOT / relative) == expected


def test_legacy_incident_candidates_remain_unchanged() -> None:
    expected = {
        "normal_operation": ["normal_operation"],
        "feature_drift": [
            "mixed_incident",
            "performance_degradation",
            "feature_drift",
            "prediction_drift",
        ],
        "data_quality_failure": ["data_quality_failure"],
        "performance_degradation": ["performance_degradation"],
        "unlabelled_drift": [
            "feature_drift",
            "prediction_drift",
            "insufficient_evidence",
        ],
        "insufficient_labels": ["insufficient_evidence"],
    }
    for scenario, candidates in expected.items():
        path = PROJECT_ROOT / "reports/generated" / scenario / "monitoring_result.json"
        result = MonitoringRunResult.model_validate(json.loads(path.read_text()))
        assert result.incident_candidates == candidates
