"""Generate reproducible model-aware replay scenarios."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from monitoring_agent.models.bundle import RegisteredModelBundle
from monitoring_agent.models.registry import ModelRegistry
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


def _load_config(model_id: str) -> dict[str, Any]:
    path = CONFIG_DIR / "scenarios" / f"{model_id}.yaml"
    if not path.is_file() and model_id == "credit_default":
        path = CONFIG_DIR / "scenarios.yaml"
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a scenario mapping in {path}.")
    return payload


def scenario_directory(
    model_id: str,
    scenario_name: str,
    *,
    project_root: Path | None = None,
) -> Path:
    root = project_root or PROJECT_ROOT
    return root / "data/scenarios" / model_id / scenario_name


def resolve_scenario_directory(
    model_id: str,
    scenario_name: str,
    *,
    project_root: Path | None = None,
) -> Path:
    """Use model-aware paths first and the legacy credit path as read fallback."""
    root = project_root or PROJECT_ROOT
    current = scenario_directory(model_id, scenario_name, project_root=root)
    if current.is_dir():
        return current
    if model_id == "credit_default":
        legacy = root / "data/scenarios" / scenario_name
        if legacy.is_dir():
            return legacy
    return current


def _stratified_sample(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    identifier_column: str,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample_size > len(features):
        raise ValueError("Scenario sample size exceeds the reference sample count.")
    if features[identifier_column].tolist() != labels[identifier_column].tolist():
        raise ValueError("Reference features and labels are not aligned.")
    rng = np.random.default_rng(seed)
    label_values = labels["actual_label"].to_numpy(dtype="int8")
    positive_positions = np.flatnonzero(label_values == 1)
    negative_positions = np.flatnonzero(label_values == 0)
    positive_count = int(round(sample_size * len(positive_positions) / len(labels)))
    negative_count = sample_size - positive_count
    selected = np.concatenate(
        [
            rng.choice(positive_positions, size=positive_count, replace=False),
            rng.choice(negative_positions, size=negative_count, replace=False),
        ]
    )
    selected = rng.permutation(selected)
    return (
        features.iloc[selected].reset_index(drop=True),
        labels.iloc[selected].reset_index(drop=True),
    )


def _set_discrete_shift(
    frame: pd.DataFrame,
    *,
    feature: str,
    target_value: int | float,
    affected_fraction: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    count = int(round(len(frame) * affected_fraction))
    positions = rng.choice(len(frame), size=count, replace=False)
    frame.loc[positions, feature] = target_value
    return {
        "feature": feature,
        "transformation": "set_to_configured_valid_risk_value",
        "target_value": target_value,
        "affected_row_count": count,
        "affected_fraction": affected_fraction,
    }


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
    affected_fraction = float(parameters["continuous_affected_fraction"])
    count = int(round(len(modified) * affected_fraction))
    positions = rng.choice(len(modified), size=count, replace=False)
    observed_min, observed_max = observed_ranges[continuous_feature]
    original_dtype = modified[continuous_feature].dtype
    shifted = (
        modified.loc[positions, continuous_feature].to_numpy(dtype=float)
        * scale_factor
    )
    clipped = np.clip(shifted, observed_min, observed_max)
    if pd.api.types.is_integer_dtype(original_dtype):
        clipped = np.round(clipped).astype(original_dtype)
    modified.loc[positions, continuous_feature] = clipped
    modifications = [
        {
            "feature": continuous_feature,
            "transformation": "multiplicative_scale_then_observed_range_clip",
            "scale_factor": scale_factor,
            "affected_row_count": count,
            "affected_fraction": affected_fraction,
        }
    ]

    configured_shifts = parameters.get("discrete_shifts")
    if configured_shifts is not None:
        for shift in configured_shifts:
            modifications.append(
                _set_discrete_shift(
                    modified,
                    feature=str(shift["feature"]),
                    target_value=shift["target_value"],
                    affected_fraction=float(shift["affected_fraction"]),
                    rng=rng,
                )
            )
    else:
        discrete_feature = str(parameters["discrete_feature"])
        linked_feature = str(parameters["linked_feature"])
        value = int(parameters["discrete_target_value"])
        fraction = float(parameters["discrete_affected_fraction"])
        count = int(round(len(modified) * fraction))
        positions = rng.choice(len(modified), size=count, replace=False)
        modified.loc[positions, discrete_feature] = value
        modified.loc[positions, linked_feature] = value
        modifications.extend(
            [
                {
                    "feature": discrete_feature,
                    "transformation": "set_to_configured_valid_risk_value",
                    "target_value": value,
                    "affected_row_count": count,
                    "affected_fraction": fraction,
                },
                {
                    "feature": linked_feature,
                    "transformation": "synchronise_linked_derived_feature",
                    "linked_to": discrete_feature,
                    "target_value": value,
                    "affected_row_count": count,
                    "affected_fraction": fraction,
                },
            ]
        )
    return modified, modifications


def _data_quality_failure(
    features: pd.DataFrame,
    parameters: dict[str, Any],
    seed: int,
    observed_ranges: dict[str, tuple[float, float]],
    *,
    identifier_column: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    modified = features.copy()
    rng = np.random.default_rng(seed)
    duplicate_count = int(parameters["duplicate_record_count"])
    if duplicate_count >= len(modified):
        raise ValueError("duplicate_record_count must be smaller than the scenario batch.")
    source_record_id = modified.loc[0, identifier_column]
    modified.loc[1:duplicate_count, identifier_column] = source_record_id

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
    return modified, [
        {
            "feature": identifier_column,
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


def _performance_degradation(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    parameters: dict[str, Any],
    bundle: RegisteredModelBundle,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    modified = labels.copy()
    probabilities = bundle.predict_probabilities(
        features[bundle.ordered_inference_features()]
    )
    negative_positions = np.flatnonzero(
        modified["actual_label"].to_numpy(dtype="int8") == 0
    )
    ordered = negative_positions[
        np.argsort(probabilities[negative_positions], kind="mergesort")
    ]
    requested_count = int(parameters["negative_to_positive_count"])
    flip_positions = ordered[:requested_count]
    modified.loc[flip_positions, "actual_label"] = 1
    return modified, [
        {
            "target": "actual_label",
            "transformation": "flip_lowest_probability_negative_labels_to_positive",
            "requested_flip_count": requested_count,
            "actual_flip_count": int(len(flip_positions)),
            "maximum_selected_probability": float(
                probabilities[flip_positions].max()
            ),
            "selection": "lowest model probabilities among observed negative labels",
        }
    ]


def _write_scenario(
    model_id: str,
    scenario_name: str,
    features: pd.DataFrame,
    labels: pd.DataFrame | None,
    manifest: ScenarioManifest,
    *,
    overwrite: bool,
) -> None:
    output = scenario_directory(model_id, scenario_name)
    feature_path = output / "features.parquet"
    label_path = output / "labels.parquet"
    manifest_path = output / "scenario_manifest.json"
    existing = [path for path in (feature_path, label_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Scenario files already exist. Pass overwrite=True: "
            + ", ".join(str(path) for path in existing)
        )
    output.mkdir(parents=True, exist_ok=True)
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
    model_id: str | None = None,
    overwrite: bool = False,
) -> ScenarioManifest:
    """Generate one configured scenario from a registered reference bundle."""
    registry = ModelRegistry()
    resolved_model_id = model_id or registry.default_model_id
    if scenario_name not in SUPPORTED_SCENARIOS:
        raise ValueError(
            f"Unsupported scenario {scenario_name!r}. "
            f"Choose from: {', '.join(SUPPORTED_SCENARIOS)}."
        )
    config = _load_config(resolved_model_id)
    scenario_config = config["scenarios"][scenario_name]
    if not scenario_config["enabled"]:
        raise ValueError(f"Scenario is disabled in config: {scenario_name}")

    bundle = RegisteredModelBundle(resolved_model_id, registry=registry)
    registered = bundle.registered_manifest
    schema = bundle.load_feature_schema()
    metadata = bundle.load_metadata()
    reference_features = bundle.load_reference_features()
    reference_labels = bundle.load_reference_labels()
    identifier = registered.data_contract.identifier_column
    seed = int(config["defaults"]["random_seed"])
    sample_size = int(config["defaults"]["sample_size"])
    features, labels = _stratified_sample(
        reference_features,
        reference_labels,
        identifier_column=identifier,
        sample_size=sample_size,
        seed=seed,
    )
    if features.columns.tolist() != [identifier, *schema.ordered_features]:
        raise ValueError("Reference sample does not preserve the registered feature order.")

    observed_ranges = {
        feature.name: (float(feature.observed_min), float(feature.observed_max))
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
            identifier_column=identifier,
        )
    elif scenario_name == "performance_degradation":
        labels, label_modifications = _performance_degradation(
            features,
            labels,
            parameters,
            bundle,
        )
    elif scenario_name == "unlabelled_drift":
        features, feature_modifications = _feature_drift(
            features,
            parameters,
            seed + 17,
            observed_ranges,
        )
        labels = None
    elif scenario_name == "insufficient_labels":
        labelled_count = int(parameters["labelled_sample_count"])
        if labelled_count >= len(labels):
            raise ValueError(
                "insufficient_labels must retain fewer labels than feature rows."
            )
        labels = labels.iloc[:labelled_count].reset_index(drop=True)
        label_modifications = [
            {
                "target": "actual_label",
                "transformation": "retain_deterministic_aligned_subset",
                "retained_row_count": labelled_count,
                "feature_row_count": len(features),
                "label_coverage_rate": labelled_count / len(features),
            }
        ]

    configured_features = {
        item["feature"]
        for item in feature_modifications
        if "feature" in item and item["feature"] != identifier
    }
    unknown = configured_features.difference(schema.ordered_features)
    if unknown:
        raise ValueError(f"Scenario references unknown features: {sorted(unknown)}")
    manifest = ScenarioManifest(
        model_id=resolved_model_id,
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
        resolved_model_id,
        scenario_name,
        features,
        labels,
        manifest,
        overwrite=overwrite,
    )
    return manifest


def generate_all_scenarios(
    *,
    model_id: str | None = None,
    overwrite: bool = False,
) -> list[ScenarioManifest]:
    registry = ModelRegistry()
    resolved_model_id = model_id or registry.default_model_id
    config = _load_config(resolved_model_id)
    return [
        generate_scenario(
            name,
            model_id=resolved_model_id,
            overwrite=overwrite,
        )
        for name in SUPPORTED_SCENARIOS
        if config["scenarios"][name]["enabled"]
    ]
