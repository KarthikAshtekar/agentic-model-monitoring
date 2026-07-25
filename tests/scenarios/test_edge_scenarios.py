"""Generation contracts for delayed-label robustness scenarios."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

import monitoring_agent.scenarios.generator as generator
from monitoring_agent.scenarios.schemas import ScenarioManifest


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("scenario_name", ["unlabelled_drift", "insufficient_labels"])
def test_edge_scenario_features_generate_reproducibly(
    tmp_path,
    monkeypatch,
    scenario_name: str,
) -> None:
    monkeypatch.setattr(generator, "PROJECT_ROOT", tmp_path)
    generator.generate_scenario(scenario_name)
    scenario_dir = tmp_path / "data/scenarios" / scenario_name
    before = _sha256(scenario_dir / "features.parquet")
    generator.generate_scenario(scenario_name, overwrite=True)
    after = _sha256(scenario_dir / "features.parquet")
    assert before == after


def test_unlabelled_drift_has_no_labels_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(generator, "PROJECT_ROOT", tmp_path)
    manifest = generator.generate_scenario("unlabelled_drift")
    scenario_dir = tmp_path / "data/scenarios/unlabelled_drift"
    parsed = ScenarioManifest.model_validate_json(
        (scenario_dir / "scenario_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest.labels_available is False
    assert parsed.labelled_sample_count == 0
    assert parsed.labels_complete is False
    assert parsed.feature_modifications[-1]["linked_to"] == "PAY_2"
    assert "PAY_0" not in parsed.feature_modifications[-1]["transformation"]
    assert len(pd.read_parquet(scenario_dir / "features.parquet")) == 1000
    assert not (scenario_dir / "labels.parquet").exists()


def test_insufficient_labels_has_exact_aligned_subset(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(generator, "PROJECT_ROOT", tmp_path)
    manifest = generator.generate_scenario("insufficient_labels")
    scenario_dir = tmp_path / "data/scenarios/insufficient_labels"
    features = pd.read_parquet(scenario_dir / "features.parquet")
    labels = pd.read_parquet(scenario_dir / "labels.parquet")

    assert len(features) == 1000
    assert len(labels) == 100
    assert labels["record_id"].is_unique
    assert set(labels["record_id"]) <= set(features["record_id"])
    assert manifest.labels_available is True
    assert manifest.labelled_sample_count == 100
    assert manifest.labels_complete is False
