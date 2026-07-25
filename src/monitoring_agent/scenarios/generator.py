"""Generate reproducible replay scenarios from the validated bundle."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml

from monitoring_agent.bundle.loader import CreditDefaultBundle
from monitoring_agent.paths import CONFIG_DIR, PROJECT_ROOT
from monitoring_agent.scenarios.schemas import ScenarioManifest

CORE_SCENARIOS = (
    "normal_operation",
    "feature_drift",
    "data_quality_failure",
    "performance_degradation",
)
SUPPORTED_SCENARIOS = (
    *CORE_SCENARIOS,
    "unlabelled_drift",
    "insufficient_labels",
)


def _load_config() -> dict[str, Any]:
    with (CONFIG_DIR / "scenarios.yaml").open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload


def _stratified_sample(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample_size > len(features):
        raise ValueError("Scenario sample size exceeds the reference sample count.")
    if features["record_id"].tolist() != labels["record_id"].tolist():
        raise ValueError("Reference features and labels are not aligned.")

    rng = np.random.default_rng(seed)
    positive_positions = np.flatnonzero(labels["actual_label"].to_numpy() == 1)
    negative_positions = np.flatnonzero(labels["actual_label"].to_numpy() == 0)
    positive_count = int(round(sample_size * len(positive_positions) / len(labels)))
    negative_count = sample_size - positive_count
    selected = np.concatenate(
        [
            rng.choice(positive_positions, size=positive_count, replace=False),
            rng.choice(negative_positions, size=negative_count, replace=False),
        ]
    )
    selected = rng.permutation(selected)
    sampled_features = features.iloc[selected].reset_index(drop=True)
    sampled_labels = labels.iloc[selected].reset_index(drop=True)
    return sampled_features, sampled_labels


def _feature_drift(
    features: pd.DataFrame,
    parameters: dict[str, Any],
    seed: int,
    observed_ranges: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    modified = features.copy()
    rng = np.random.default_rng(seed)

    continuous_feature = str(parameters["continuous_feature"])
    scale_factor = float(parameters["continuous_scale_factor"])
    continuous_fraction = float(parameters["continuous_affected_fraction"])
    continuous_count = int(round(len(modified) * continuous_fraction))
    continuous_positions = rng.choice(
        len(modified),
        size=continuous_count,
        replace=False,
    )
    observed_min, observed_max = observed_ranges[continuous_feature]
    shifted = np.round(
        modified.loc[continuous_positions, continuous_feature].to_numpy(dtype=float)
        * scale_factor
    )
    modified.loc[continuous_positions, continuous_feature] = np.clip(
        shifted,
        observed_min,
        observed_max,
    ).astype("int64")

    discrete_feature = str(parameters["discrete_feature"])
    linked_feature = str(parameters["linked_feature"])
    discrete_value = int(parameters["discrete_target_value"])
    discrete_fraction = float(parameters["discrete_affected_fraction"])
    discrete_count = int(round(len(modified) * discrete_fraction))
    discrete_positions = rng.choice(len(modified), size=discrete_count, replace=False)
    modified.loc[discrete_positions, discrete_feature] = discrete_value
    modified.loc[discrete_positions, linked_feature] = discrete_value

    modifications = [
        {
            "feature": continuous_feature,
            "transformation": "multiplicative_scale_then_observed_range_clip",
            "scale_factor": scale_factor,
            "affected_row_count": continuous_count,
            "affected_fraction": continuous_fraction,
        },
        {
            "feature": discrete_feature,
            "transformation": "set_to_higher_repayment_delay_category",
            "target_value": discrete_value,
            "affected_row_count": discrete_count,
            "affected_fraction": discrete_fraction,
        },
        {
            "feature": linked_feature,
            "transformation": "synchronise_engineered_recent_delay_with_PAY_0",
            "target_value": discrete_value,
            "affected_row_count": discrete_count,
            "affected_fraction": discrete_fraction,
        },
    ]
    return modified, modifications


def _data_quality_failure(
    features: pd.DataFrame,
    parameters: dict[str, Any],
    seed: int,
    observed_ranges: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    modified = features.copy()
    rng = np.random.default_rng(seed)

    duplicate_count = int(parameters["duplicate_record_count"])
    if duplicate_count >= len(modified):
        raise ValueError("duplicate_record_count must be smaller than the scenario batch.")
    source_record_id = modified.loc[0, "record_id"]
    modified.loc[1:duplicate_count, "record_id"] = source_record_id

    missing_feature = str(parameters["missing_feature"])
    missing_fraction = float(parameters["missing_fraction"])
    missing_count = int(round(len(modified) * missing_fraction))
    missing_positions = rng.choice(len(modified), size=missing_count, replace=False)
    modified[missing_feature] = modified[missing_feature].astype(float)
    modified.loc[missing_positions, missing_feature] = np.nan

    range_feature = str(parameters["range_violation_feature"])
    range_fraction = float(parameters["range_violation_fraction"])
    range_count = int(round(len(modified) * range_fraction))
    range_positions = rng.choice(len(modified), size=range_count, replace=False)
    _, observed_max = observed_ranges[range_feature]
    injected_value = int(
        round(observed_max * float(parameters["range_violation_multiplier"]))
    )
    modified.loc[range_positions, range_feature] = injected_value

    modifications = [
        {
            "feature": "record_id",
            "transformation": "duplicate_first_record_id",
            "duplicate_record_count": duplicate_count,
            "duplicate_rate": duplicate_count / len(modified),
        },
        {
            "feature": missing_feature,
            "transformation": "set_values_to_missing",
            "affected_row_count": missing_count,
            "affected_fraction": missing_fraction,
        },
        {
            "feature": range_feature,
            "transformation": "set_above_observed_reference_maximum",
            "injected_value": injected_value,
            "reference_observed_maximum": observed_max,
            "affected_row_count": range_count,
            "affected_fraction": range_fraction,
        },
    ]
    return modified, modifications


def _performance_degradation(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    parameters: dict[str, Any],
    bundle: CreditDefaultBundle,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    modified_labels = labels.copy()
    inference_frame = features[bundle.ordered_inference_features()]
    probabilities = bundle.predict_probabilities(inference_frame)
    negative_positions = np.flatnonzero(
        modified_labels["actual_label"].to_numpy(dtype="int8") == 0
    )
    ordered_negative_positions = negative_positions[
        np.argsort(probabilities[negative_positions], kind="mergesort")
    ]
    requested_count = int(parameters["negative_to_positive_count"])
    flip_positions = ordered_negative_positions[:requested_count]
    modified_labels.loc[flip_positions, "actual_label"] = 1
    modifications = [
        {
            "target": "actual_label",
            "transformation": "flip_lowest_probability_negative_labels_to_positive",
            "requested_flip_count": requested_count,
            "actual_flip_count": int(len(flip_positions)),
            "maximum_selected_probability": float(probabilities[flip_positions].max()),
            "selection": "lowest model probabilities among observed negative labels",
        }
    ]
    return modified_labels, modifications


def _unlabelled_drift(
    features: pd.DataFrame,
    parameters: dict[str, Any],
    seed: int,
    observed_ranges: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Create valid drift using a different repayment feature and milder limit shift."""
    modified, modifications = _feature_drift(
        features,
        parameters,
        seed + 17,
        observed_ranges,
    )
    modifications[-1]["transformation"] = "synchronise_linked_delay_feature"
    modifications[-1]["linked_to"] = str(parameters["discrete_feature"])
    return modified, modifications


def _write_scenario(
    scenario_name: str,
    features: pd.DataFrame,
    labels: pd.DataFrame | None,
    manifest: ScenarioManifest,
    *,
    overwrite: bool,
) -> None:
    scenario_dir = PROJECT_ROOT / "data/scenarios" / scenario_name
    feature_path = scenario_dir / "features.parquet"
    label_path = scenario_dir / "labels.parquet"
    manifest_path = scenario_dir / "scenario_manifest.json"
    existing = [path for path in (feature_path, label_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Scenario files already exist. Pass overwrite=True: "
            + ", ".join(str(path) for path in existing)
        )
    scenario_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feature_path, index=False)
    if labels is None:
        if label_path.is_file():
            label_path.unlink()
    else:
        labels.to_parquet(label_path, index=False)
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_scenario(
    scenario_name: str,
    *,
    overwrite: bool = False,
) -> ScenarioManifest:
    """Generate one configured scenario from the authoritative reference bundle."""
    if scenario_name not in SUPPORTED_SCENARIOS:
        raise ValueError(
            f"Unsupported scenario {scenario_name!r}. "
            f"Choose from: {', '.join(SUPPORTED_SCENARIOS)}."
        )
    config = _load_config()
    scenario_config = config["scenarios"][scenario_name]
    if not scenario_config["enabled"]:
        raise ValueError(f"Scenario is disabled in config: {scenario_name}")

    bundle = CreditDefaultBundle()
    schema = bundle.load_feature_schema()
    metadata = bundle.load_metadata()
    reference_features = bundle.load_reference_features()
    reference_labels = bundle.load_reference_labels()
    seed = int(config["defaults"]["random_seed"])
    sample_size = int(config["defaults"]["sample_size"])
    features, labels = _stratified_sample(
        reference_features,
        reference_labels,
        sample_size=sample_size,
        seed=seed,
    )
    if features.columns.tolist() != ["record_id", *schema.ordered_features]:
        raise ValueError("Reference sample does not preserve the exported feature order.")

    observed_ranges = {
        feature.name: (
            float(feature.observed_min),
            float(feature.observed_max),
        )
        for feature in schema.features
        if feature.observed_min is not None and feature.observed_max is not None
    }
    feature_modifications: list[dict[str, Any]] = []
    label_modifications: list[dict[str, Any]] = []
    parameters = scenario_config["parameters"]

    if scenario_name == "feature_drift":
        features, feature_modifications = _feature_drift(
            features,
            parameters,
            seed,
            observed_ranges,
        )
    elif scenario_name == "data_quality_failure":
        features, feature_modifications = _data_quality_failure(
            features,
            parameters,
            seed,
            observed_ranges,
        )
    elif scenario_name == "performance_degradation":
        labels, label_modifications = _performance_degradation(
            features,
            labels,
            parameters,
            bundle,
        )
    elif scenario_name == "unlabelled_drift":
        features, feature_modifications = _unlabelled_drift(
            features,
            parameters,
            seed,
            observed_ranges,
        )
        labels = None
    elif scenario_name == "insufficient_labels":
        labelled_sample_count = int(parameters["labelled_sample_count"])
        if labelled_sample_count >= len(labels):
            raise ValueError(
                "insufficient_labels must retain fewer labels than feature rows."
            )
        labels = labels.iloc[:labelled_sample_count].reset_index(drop=True)
        label_modifications = [
            {
                "target": "actual_label",
                "transformation": "retain_deterministic_aligned_subset",
                "retained_row_count": labelled_sample_count,
                "feature_row_count": len(features),
                "label_coverage_rate": labelled_sample_count / len(features),
            }
        ]

    configured_features = {
        modification["feature"]
        for modification in feature_modifications
        if "feature" in modification and modification["feature"] != "record_id"
    }
    unknown_features = configured_features.difference(schema.ordered_features)
    if unknown_features:
        raise ValueError(f"Scenario references unknown features: {sorted(unknown_features)}")

    manifest = ScenarioManifest(
        scenario_name=scenario_name,
        scenario_version="1.0.0",
        description=scenario_config["description"],
        simulation_type=scenario_config["simulation_type"],
        random_seed=seed,
        source_reference_sample_count=metadata.reference_sample_count,
        generated_sample_count=len(features),
        labels_available=labels is not None,
        labelled_sample_count=0 if labels is None else len(labels),
        labels_complete=labels is not None and len(labels) == len(features),
        expected_incident_candidates=scenario_config["expected_incident_candidates"],
        feature_modifications=feature_modifications,
        label_modifications=label_modifications,
        limitations=[
            "Replay simulation from an academic held-out reference split.",
            "Scenario evidence is synthetic and does not represent an observed incident.",
            *scenario_config.get("limitations", []),
        ],
        created_at_utc=datetime.now(UTC),
    )
    _write_scenario(
        scenario_name,
        features,
        labels,
        manifest,
        overwrite=overwrite,
    )
    return manifest


def generate_all_scenarios(*, overwrite: bool = False) -> list[ScenarioManifest]:
    """Generate every enabled MVP scenario in configured order."""
    config = _load_config()
    return [
        generate_scenario(name, overwrite=overwrite)
        for name in SUPPORTED_SCENARIOS
        if config["scenarios"][name]["enabled"]
    ]
