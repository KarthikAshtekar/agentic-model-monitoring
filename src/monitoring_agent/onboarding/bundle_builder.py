"""Deterministic full-inference export for a registered binary classifier."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml
from sklearn.pipeline import Pipeline

from monitoring_agent.adapters.binary_classification import (
    BinaryClassificationAdapter,
    CalibratedBinaryClassifier,
)
from monitoring_agent.bundle.schemas import (
    BundleManifest,
    FeatureDefinition,
    FeatureSchema,
    ManifestFile,
    ModelMetadata,
    ReferenceMetrics,
    TargetDefinition,
    ThresholdMetrics,
)
from monitoring_agent.domains.diabetes_screening import BRFSSFeatureEngineer
from monitoring_agent.models.manifest import RegisteredModelManifest
from monitoring_agent.models.registry import ModelRegistry
from monitoring_agent.onboarding.schemas import OnboardingValidation
from monitoring_agent.paths import PROJECT_ROOT

SOURCE_MODEL = "models/brfss_final/final_binary_xgboost.joblib"
SOURCE_CALIBRATOR = "models/brfss_final/binary_probability_calibrator.joblib"
SOURCE_TEST = "data/processed/brfss_test_set.csv.gz"
SOURCE_PREDICTIONS = "reports/brfss_final/tables/binary_test_predictions.csv"
SOURCE_SUMMARY = "reports/brfss_final/run_summary.json"
PROBABILITY_REPRODUCTION_TOLERANCE = 2e-7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(source: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _load_manifest(path: Path) -> RegisteredModelManifest:
    with path.open(encoding="utf-8") as handle:
        return RegisteredModelManifest.model_validate(yaml.safe_load(handle))


def _feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features.columns:
        series = features[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        quantiles = numeric.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        rows.append(
            {
                "feature": feature,
                "dtype": str(series.dtype),
                "count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()),
                "unique_count": int(series.nunique(dropna=True)),
                "mean": float(numeric.mean()),
                "std": float(numeric.std()),
                "min": float(numeric.min()),
                "p01": float(quantiles.loc[0.01]),
                "p05": float(quantiles.loc[0.05]),
                "p25": float(quantiles.loc[0.25]),
                "p50": float(quantiles.loc[0.50]),
                "p75": float(quantiles.loc[0.75]),
                "p95": float(quantiles.loc[0.95]),
                "p99": float(quantiles.loc[0.99]),
                "max": float(numeric.max()),
                "allowed_values_json": json.dumps(
                    sorted(series.dropna().unique().tolist())
                    if series.nunique(dropna=True) <= 20
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _threshold_models(
    metrics: dict[str, Any],
) -> dict[str, ThresholdMetrics]:
    output: dict[str, ThresholdMetrics] = {}
    for _name, values in metrics.items():
        key = (
            "default_0_50"
            if np.isclose(values["threshold"], 0.5)
            else "operating_0_25"
            if np.isclose(values["threshold"], 0.25)
            else f"threshold_{values['threshold']:.4f}".replace(".", "_")
        )
        output[key] = ThresholdMetrics.model_validate(values)
    return output


def _manifest_entry(
    path: Path,
    root: Path,
    purpose: str,
    source_path: str | None,
    row_count: int | None = None,
    column_count: int | None = None,
) -> ManifestFile:
    return ManifestFile(
        relative_path=path.relative_to(root).as_posix(),
        purpose=purpose,
        source_path=source_path,
        generation_method="Deterministic export without retraining.",
        row_count=row_count,
        column_count=column_count,
        file_size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def onboard_binary_classifier(
    source_project: Path | str,
    manifest_path: Path | str,
    model_id: str,
    *,
    overwrite: bool = False,
) -> OnboardingValidation:
    """Export the authoritative BRFSS binary model as a self-contained bundle."""
    source = Path(source_project).resolve()
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = PROJECT_ROOT / manifest_file
    manifest = _load_manifest(manifest_file.resolve())
    if manifest.model_id != model_id:
        raise ValueError("CLI model_id does not match the supplied manifest.")
    if model_id != "diabetes_risk":
        raise ValueError("This phase's deterministic exporter supports diabetes_risk.")
    ModelRegistry().load_manifest(model_id)

    required_source_paths = [
        SOURCE_MODEL,
        SOURCE_CALIBRATOR,
        SOURCE_TEST,
        SOURCE_PREDICTIONS,
        SOURCE_SUMMARY,
    ]
    missing = [item for item in required_source_paths if not (source / item).is_file()]
    if missing:
        raise FileNotFoundError(f"Required authoritative source files missing: {missing}")

    commit = _git_value(source, "rev-parse", "HEAD")
    status = _git_value(source, "status", "--porcelain")
    clean = status == "" if status is not None else None
    if manifest.provenance.source_commit and commit != manifest.provenance.source_commit:
        raise ValueError("Source commit does not match the registered provenance.")
    if clean is False:
        raise ValueError("Source project must be clean for deterministic onboarding.")

    bundle_root = PROJECT_ROOT / "registered_models" / model_id
    paths = {
        "artifacts": bundle_root / "artifacts",
        "baselines": bundle_root / "baselines",
        "reference": bundle_root / "reference",
        "onboarding": bundle_root / "onboarding",
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        PROJECT_ROOT / value
        for value in manifest.bundle_paths.model_dump().values()
        if value is not None
    ]
    if not overwrite and any(path.exists() for path in expected_outputs):
        raise FileExistsError("Bundle outputs exist; pass --overwrite to replace them.")

    source_python = source / "src"
    sys.path.insert(0, str(source_python))
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        source_model = joblib.load(source / SOURCE_MODEL)
        source_calibrator = joblib.load(source / SOURCE_CALIBRATOR)
    captured_warnings.extend(str(item.message) for item in caught)

    source_test = pd.read_csv(source / SOURCE_TEST, index_col="row_index")
    source_predictions = pd.read_csv(source / SOURCE_PREDICTIONS)
    source_summary = json.loads((source / SOURCE_SUMMARY).read_text(encoding="utf-8"))
    contract = manifest.data_contract
    features = source_test[contract.ordered_features].copy()
    labels = source_test[contract.target_column].astype("int8")
    record_ids = source_test.index.to_series().reset_index(drop=True).astype("int64")
    if source_predictions["row_index"].tolist() != record_ids.tolist():
        raise ValueError("Source prediction rows do not align with the held-out split.")
    if source_predictions["actual"].astype("int8").tolist() != labels.tolist():
        raise ValueError("Source prediction labels do not align with the held-out split.")

    portable_pipeline = Pipeline(
        [
            ("feature_engineering", BRFSSFeatureEngineer()),
            *source_model.steps[1:],
        ]
    )
    portable_model = CalibratedBinaryClassifier(
        portable_pipeline,
        source_calibrator.model,
        positive_class_index=manifest.prediction_contract.positive_class_index,
    )
    adapter = BinaryClassificationAdapter(manifest, portable_model)
    scores = adapter.predict_scores(features)
    stored_scores = source_predictions["calibrated_probability"].to_numpy(dtype=float)
    max_probability_difference = float(
        np.max(np.abs(scores.to_numpy(dtype=float) - stored_scores))
    )
    if not np.allclose(
        scores,
        stored_scores,
        atol=PROBABILITY_REPRODUCTION_TOLERANCE,
        rtol=PROBABILITY_REPRODUCTION_TOLERANCE,
    ):
        raise ValueError("Portable model does not reproduce authoritative probabilities.")

    model_path = paths["artifacts"] / "model.joblib"
    feature_schema_path = paths["artifacts"] / "feature_schema.json"
    model_metadata_path = paths["artifacts"] / "model_metadata.json"
    bundle_manifest_path = paths["artifacts"] / "bundle_manifest.json"
    reference_features_path = paths["reference"] / "features.parquet"
    reference_labels_path = paths["reference"] / "labels.parquet"
    reference_predictions_path = paths["reference"] / "predictions.parquet"
    reference_metrics_path = paths["baselines"] / "reference_metrics.json"
    reference_summary_path = paths["baselines"] / "reference_feature_summary.parquet"
    source_inventory_path = paths["onboarding"] / "source_inventory.md"
    validation_path = paths["onboarding"] / "validation_result.json"
    validation_report_path = paths["onboarding"] / "validation_report.md"

    joblib.dump(portable_model, model_path)
    reference_features = pd.concat(
        [record_ids.rename("record_id"), features.reset_index(drop=True)],
        axis=1,
    )
    reference_labels = pd.DataFrame(
        {"record_id": record_ids, "actual_label": labels.reset_index(drop=True)}
    )
    score_series = scores.reset_index(drop=True)
    reference_predictions = pd.DataFrame(
        {
            "record_id": record_ids,
            "predicted_probability": score_series,
            "predicted_class_default_threshold": adapter.apply_threshold(
                score_series,
                manifest.prediction_contract.default_threshold,
            ),
            "predicted_class_operating_threshold": adapter.apply_threshold(
                score_series,
                manifest.prediction_contract.operating_threshold,
            ),
        }
    )
    reference_features.to_parquet(reference_features_path, index=False)
    reference_labels.to_parquet(reference_labels_path, index=False)
    reference_predictions.to_parquet(reference_predictions_path, index=False)
    summary_frame = _feature_summary(features)
    summary_frame.to_parquet(reference_summary_path, index=False)

    categorical = set(contract.categorical_features)
    definitions: list[FeatureDefinition] = []
    for position, name in enumerate(contract.ordered_features):
        series = features[name]
        values = (
            sorted(series.dropna().unique().tolist())
            if name in categorical and series.nunique(dropna=True) <= 20
            else None
        )
        definitions.append(
            FeatureDefinition(
                name=name,
                position=position,
                dtype=str(series.dtype),
                role="predictor",
                nullable=name in contract.nullable_features,
                allowed_values=values,
                observed_min=float(series.min()),
                observed_max=float(series.max()),
            )
        )
    feature_schema = FeatureSchema(
        schema_version="1.0.0",
        ordered_features=contract.ordered_features,
        required_columns=contract.ordered_features,
        identifier_columns=[contract.identifier_column],
        categorical_features=contract.categorical_features,
        numeric_features=[
            name
            for name in contract.ordered_features
            if name not in categorical
        ],
        integer_constrained_features=contract.integer_constrained_features,
        target=TargetDefinition(
            name=contract.target_column,
            exported_name="actual_label",
            dtype="int8",
            positive_class=1,
            meaning="1 identifies the diabetes-positive binary screening outcome.",
        ),
        features=definitions,
        source_reference_split=(
            "stratified held-out test split (test_size=0.20, random_state=42)"
        ),
        observed_range_policy=(
            "Observed held-out minima and maxima are monitoring baselines, not "
            "clinical validity ranges."
        ),
    )
    feature_schema_path.write_text(
        feature_schema.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    metric_values = adapter.calculate_performance(
        labels.reset_index(drop=True),
        score_series,
        [
            manifest.prediction_contract.default_threshold,
            manifest.prediction_contract.operating_threshold,
        ],
    )
    threshold_models = _threshold_models(metric_values)
    source_metrics = source_summary["binary_metrics"]
    operating = next(
        item
        for item in threshold_models.values()
        if np.isclose(
            item.threshold,
            manifest.prediction_contract.operating_threshold,
        )
    )
    comparable = {
        name: float(source_metrics[name])
        for name in (
            "roc_auc",
            "pr_auc",
            "recall",
            "precision",
            "f1",
            "brier_score",
        )
    }
    differences = {
        name: abs(float(getattr(operating, name)) - source_value)
        for name, source_value in comparable.items()
    }
    material = [
        f"{name} differs by {difference:.8g}"
        for name, difference in differences.items()
        if difference > 1e-6
    ]
    reference_metrics = ReferenceMetrics(
        metrics_version="1.0.0",
        reference_split=feature_schema.source_reference_split,
        sample_count=len(features),
        positive_count=int(labels.sum()),
        positive_rate=float(labels.mean()),
        thresholds=threshold_models,
        authoritative_source_metrics={"operating_0_25": comparable},
        metric_absolute_differences={"operating_0_25": differences},
        material_discrepancies=material,
        comparison_tolerance=1e-6,
    )
    reference_metrics_path.write_text(
        reference_metrics.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    now = datetime.now(UTC)
    metadata = ModelMetadata(
        model_name=manifest.identity.display_name,
        model_family="Calibrated XGBoost binary classifier in a scikit-learn Pipeline",
        model_version=manifest.identity.model_version,
        source_repository="25BM6JP22",
        source_repository_path=(
            "External read-only source path supplied at onboarding and not persisted."
        ),
        source_artifact_paths=manifest.provenance.source_artifact_paths,
        dataset_name="CDC BRFSS 2015 health indicators",
        target_column=contract.target_column,
        positive_class=1,
        feature_count=len(contract.ordered_features),
        operating_threshold=manifest.prediction_contract.operating_threshold,
        default_threshold=manifest.prediction_contract.default_threshold,
        reference_split=feature_schema.source_reference_split,
        reference_sample_count=len(features),
        inference_artifact_path=manifest.bundle_paths.model_artifact,
        inference_available=True,
        integration_strategy="existing_fitted_pipeline_plus_fitted_calibrator",
        training_library_versions={
            "python": "3.13.7",
            "numpy": "2.5.1",
            "pandas": "3.0.5",
            "scikit-learn": "1.9.0",
            "xgboost": "3.3.0",
            "joblib": "1.5.3",
        },
        export_library_versions={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        bundle_created_at_utc=now.isoformat(),
        production_status="controlled_replay_only",
        compatibility_warnings=list(dict.fromkeys(captured_warnings)),
        limitations=manifest.provenance.limitations,
    )
    model_metadata_path.write_text(
        metadata.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    source_inventory_path.write_text(
        f"""# Diabetes source inventory

- Source project: `25BM6JP22` (external read-only path not persisted)
- Source commit: `{commit}`
- Source worktree clean: `{str(clean).lower()}`
- Dataset: CDC BRFSS 2015, `253680` rows in the maintained source project
- Task: binary diabetes-positive versus no-diabetes/prediabetes screening outcome
- Target: `{contract.target_column}`; positive class `1`
- Selected model: fitted XGBoost pipeline plus fitted sigmoid calibrator
- Raw inference features: `{len(contract.ordered_features)}`
- Preprocessed representation: `{contract.preprocessed_feature_count}`
- Split: {feature_schema.source_reference_split}; `{len(features)}` rows
- Operating threshold: `{manifest.prediction_contract.operating_threshold}`
- Bundle mode: `live_inference`
- Prediction reproduction maximum absolute difference: `{max_probability_difference:.12g}`
- Source environment: Python 3.13.7, NumPy 2.5.1, pandas 3.0.5,
  scikit-learn 1.9.0, XGBoost 3.3.0, joblib 1.5.3
- Monitoring environment XGBoost: `{xgboost.__version__}`

## Authoritative source artifacts

{chr(10).join(f"- `{item}`" for item in manifest.provenance.source_artifact_paths)}

## Strategy and limitations

Full inference is reproducible without retraining. The export replaces only the source
module references for stateless feature engineering and calibration composition; fitted
scikit-learn and XGBoost state remains unchanged. The source XGBoost joblib emits a
cross-version portability warning under the pinned monitoring environment, but reference
scores reproduce within tolerance. This is survey-based risk screening, not diagnosis,
treatment, individual patient care, or external clinical validation.
""",
        encoding="utf-8",
        newline="\n",
    )

    payload_files = [
        (model_path, "Portable fitted calibrated inference pipeline.", SOURCE_MODEL, None),
        (
            feature_schema_path,
            "Exact ordered raw inference feature and target contract.",
            None,
            None,
        ),
        (
            model_metadata_path,
            "Model identity, compatibility, and source provenance.",
            None,
            None,
        ),
        (
            reference_features_path,
            "Held-out reference identifiers and raw predictors.",
            SOURCE_TEST,
            reference_features,
        ),
        (
            reference_labels_path,
            "Held-out reference identifiers and binary outcomes.",
            SOURCE_TEST,
            reference_labels,
        ),
        (
            reference_predictions_path,
            "Calibrated probabilities and thresholded predictions.",
            SOURCE_PREDICTIONS,
            reference_predictions,
        ),
        (
            reference_metrics_path,
            "Reproduced held-out binary metrics.",
            SOURCE_SUMMARY,
            None,
        ),
        (
            reference_summary_path,
            "One-row-per-feature reference distribution summary.",
            SOURCE_TEST,
            summary_frame,
        ),
        (
            source_inventory_path,
            "Read-only source discovery and bundle-strategy record.",
            None,
            None,
        ),
    ]
    entries = [
        _manifest_entry(
            path,
            PROJECT_ROOT,
            purpose,
            source_path,
            None if frame is None else len(frame),
            None if frame is None else len(frame.columns),
        )
        for path, purpose, source_path, frame in payload_files
    ]
    bundle_manifest = BundleManifest(
        bundle_version="1.0.0",
        created_at_utc=now.isoformat(),
        source_repository="25BM6JP22",
        source_repository_path=(
            "External read-only source path supplied at onboarding and not persisted."
        ),
        source_git_commit=commit,
        source_git_dirty=None if clean is None else not clean,
        export_status="complete",
        integration_strategy="existing_fitted_pipeline_plus_fitted_calibrator",
        inference_available=True,
        compatibility_warnings=list(dict.fromkeys(captured_warnings)),
        manifest_self_checksum=None,
        manifest_self_checksum_reason=(
            "A manifest cannot contain its own stable checksum; all payload files "
            "are covered."
        ),
        files=entries,
    )
    bundle_manifest_path.write_text(
        bundle_manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    validation = OnboardingValidation(
        model_id=model_id,
        bundle_mode=manifest.provenance.bundle_mode,
        valid=(
            not material
            and max_probability_difference <= PROBABILITY_REPRODUCTION_TOLERANCE
        ),
        checks={
            "source_worktree_clean": clean is True,
            "source_commit_matches": commit == manifest.provenance.source_commit,
            "row_alignment": True,
            "feature_order": features.columns.tolist() == contract.ordered_features,
            "target_excluded": contract.target_column not in features.columns,
            "finite_scores": bool(np.isfinite(scores).all()),
            "score_range": bool(scores.between(0.0, 1.0).all()),
            "probabilities_reproduced": (
                max_probability_difference <= PROBABILITY_REPRODUCTION_TOLERANCE
            ),
            "reference_metrics_reproduced": not material,
            "direct_inference": True,
        },
        source_commit=commit,
        source_worktree_clean=clean,
        reference_sample_count=len(features),
        raw_feature_count=len(contract.ordered_features),
        preprocessed_feature_count=contract.preprocessed_feature_count,
        maximum_probability_absolute_difference=max_probability_difference,
        metric_absolute_differences=differences,
        material_discrepancies=material,
        warnings=list(dict.fromkeys(captured_warnings)),
        created_at_utc=now,
    )
    validation_path.write_text(
        validation.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    differences_text = "\n".join(
        f"- `{name}`: `{value:.12g}`" for name, value in differences.items()
    )
    validation_report_path.write_text(
        f"""# Diabetes bundle validation

- Valid: `{str(validation.valid).lower()}`
- Bundle mode: `{validation.bundle_mode}`
- Reference rows: `{validation.reference_sample_count}`
- Raw/preprocessed features: `{validation.raw_feature_count}` /
  `{validation.preprocessed_feature_count}`
- Maximum probability difference: `{max_probability_difference:.12g}`

## Source-versus-reproduced metric differences

{differences_text}

No model fitting, tuning, source-file modification, clinical action, or remediation was
performed.
""",
        encoding="utf-8",
        newline="\n",
    )
    return validation
