"""Tests against generated deterministic scenario artifacts."""

import hashlib
from pathlib import Path

import pandas as pd

import monitoring_agent.scenarios.generator as generator
from monitoring_agent.bundle.loader import CreditDefaultBundle
from monitoring_agent.paths import PROJECT_ROOT
from monitoring_agent.scenarios.generator import (
    CORE_SCENARIOS,
    generate_scenario,
)
from monitoring_agent.scenarios.schemas import ScenarioManifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generated_manifests_and_feature_order() -> None:
    """Every scenario manifest parses and every feature table preserves schema order."""
    expected_columns = [
        "record_id",
        *CreditDefaultBundle().ordered_inference_features(),
    ]
    for scenario in CORE_SCENARIOS:
        scenario_dir = PROJECT_ROOT / "data/scenarios" / scenario
        manifest = ScenarioManifest.model_validate_json(
            (scenario_dir / "scenario_manifest.json").read_text(encoding="utf-8")
        )
        features = pd.read_parquet(scenario_dir / "features.parquet")
        labels = pd.read_parquet(scenario_dir / "labels.parquet")

        assert manifest.scenario_name == scenario
        assert manifest.generated_sample_count == len(features) == len(labels) == 1000
        assert features.columns.tolist() == expected_columns


def test_normal_scenario_generation_is_reproducible(tmp_path, monkeypatch) -> None:
    """The fixed seed regenerates byte-identical feature and label Parquet files."""
    monkeypatch.setattr(generator, "PROJECT_ROOT", tmp_path)
    scenario_dir = tmp_path / "data/scenarios/credit_default/normal_operation"
    feature_path = scenario_dir / "features.parquet"
    label_path = scenario_dir / "labels.parquet"
    generate_scenario("normal_operation")
    before = (_sha256(feature_path), _sha256(label_path))

    manifest = generate_scenario("normal_operation", overwrite=True)
    after = (_sha256(feature_path), _sha256(label_path))

    assert manifest.random_seed == 20260725
    assert before == after
