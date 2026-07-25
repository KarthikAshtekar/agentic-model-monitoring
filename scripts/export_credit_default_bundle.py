"""Export the selected credit-default model and held-out reference split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from monitoring_agent.paths import PROJECT_ROOT

DEFAULT_SOURCE_REPO = Path(r"D:\PGDBA\Projects\credit-default-xai")
MODEL_DESTINATION = PROJECT_ROOT / "artifacts/models/credit_default_pipeline.joblib"
METADATA_DESTINATION = PROJECT_ROOT / "artifacts/metadata/model_metadata.json"
SCHEMA_DESTINATION = PROJECT_ROOT / "artifacts/metadata/feature_schema.json"
MANIFEST_DESTINATION = PROJECT_ROOT / "artifacts/metadata/bundle_manifest.json"
METRICS_DESTINATION = PROJECT_ROOT / "artifacts/baselines/reference_metrics.json"
SUMMARY_DESTINATION = (
    PROJECT_ROOT / "artifacts/baselines/reference_feature_summary.parquet"
)
FEATURES_DESTINATION = PROJECT_ROOT / "data/reference/reference_features.parquet"
LABELS_DESTINATION = PROJECT_ROOT / "data/reference/reference_labels.parquet"
PREDICTIONS_DESTINATION = (
    PROJECT_ROOT / "data/reference/reference_predictions.parquet"
)
INVENTORY_DESTINATION = PROJECT_ROOT / "docs/credit_default_bundle_inventory.md"

GENERATED_PATHS = (
    MODEL_DESTINATION,
    METADATA_DESTINATION,
    SCHEMA_DESTINATION,
    MANIFEST_DESTINATION,
    METRICS_DESTINATION,
    SUMMARY_DESTINATION,
    FEATURES_DESTINATION,
    LABELS_DESTINATION,
    PREDICTIONS_DESTINATION,
    INVENTORY_DESTINATION,
)

DEFAULT_THRESHOLD = 0.50
METRIC_COMPARISON_TOLERANCE = 1e-6
INTEGRATION_STRATEGY = "existing_complete_fitted_inference_pipeline"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(source_repo: Path, *arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(source_repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _source_environment_versions(source_repo: Path) -> dict[str, str]:
    source_python = source_repo / ".venv/Scripts/python.exe"
    if not source_python.is_file():
        return {}
    code = (
        "import json,sys,joblib,numpy,pandas,scipy,sklearn,xgboost;"
        "print(json.dumps({'python':sys.version.split()[0],"
        "'joblib':joblib.__version__,'numpy':numpy.__version__,"
        "'pandas':pandas.__version__,'scipy':scipy.__version__,"
        "'scikit-learn':sklearn.__version__,'xgboost':xgboost.__version__}))"
    )
    completed = subprocess.run(
        [str(source_python), "-c", code],
        cwd=source_repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {}
    payload = json.loads(completed.stdout)
    return {str(name): str(version) for name, version in payload.items()}


def _export_environment_versions() -> dict[str, str]:
    names = (
        "joblib",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
    )
    versions = {"python": sys.version.split()[0]}
    for name in names:
        versions[name] = importlib.metadata.version(name)
    return versions


def _resolve_source_artifacts(source_repo: Path) -> dict[str, Path]:
    if not source_repo.is_dir():
        raise FileNotFoundError(
            f"Source repository not found: {source_repo}. "
            "Pass the verified repository location with --source-repo."
        )

    training_summary_path = (
        source_repo / "reports/model_validation/xgboost_training_summary.json"
    )
    if not training_summary_path.is_file():
        raise FileNotFoundError(
            "Authoritative XGBoost training summary was not found at "
            f"{training_summary_path}."
        )
    training_summary = _read_json(training_summary_path)
    model_relative_path = training_summary.get("model_path")
    if not isinstance(model_relative_path, str):
        raise ValueError("Training summary does not identify a model_path.")

    artifacts = {
        "model": source_repo / model_relative_path,
        "processed_dataset": (
            source_repo / "data/processed/uci_taiwan_credit_default_processed.csv"
        ),
        "stored_predictions": (
            source_repo / "reports/model_validation/xgboost_test_predictions.csv"
        ),
        "model_metrics": (
            source_repo / "reports/model_validation/xgboost_public_model_metrics.json"
        ),
        "selected_policy": (
            source_repo / "reports/model_validation/selected_recall_policy.json"
        ),
        "training_summary": training_summary_path,
        "model_card": source_repo / "docs/model_card.md",
    }
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required authoritative source artifacts are missing: {missing}")
    return artifacts


def _source_split(source_repo: Path, processed_dataset: Path) -> Any:
    sys.path.insert(0, str(source_repo))
    try:
        from src.data_preprocessing import (  # type: ignore[import-not-found]
            FEATURE_SET_APPLICATION,
            TARGET_COL,
            get_dataset_split,
        )

        source_frame = pd.read_csv(processed_dataset)
        return get_dataset_split(
            source_frame,
            target_col=TARGET_COL,
            feature_set=FEATURE_SET_APPLICATION,
        )
    finally:
        sys.path.remove(str(source_repo))


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_integer_constrained(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series.dropna(), errors="coerce")
    if numeric.empty or numeric.isna().any():
        return False
    return bool(np.isclose(numeric.to_numpy(), np.round(numeric.to_numpy())).all())


def _allowed_values(series: pd.Series) -> list[Any] | None:
    values = series.dropna().unique()
    if len(values) > 20:
        return None
    return [_json_scalar(value) for value in sorted(values.tolist())]


def _feature_schema(
    reference_features: pd.DataFrame,
    feature_names: list[str],
    reference_split: str,
) -> dict[str, Any]:
    semantic_categorical = [
        feature for feature in ("EDUCATION", "MARRIAGE") if feature in feature_names
    ]
    integer_constrained = [
        feature
        for feature in feature_names
        if _is_integer_constrained(reference_features[feature])
    ]
    feature_definitions = []
    for position, feature in enumerate(feature_names):
        series = reference_features[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        feature_definitions.append(
            {
                "name": feature,
                "position": position,
                "dtype": str(series.dtype),
                "role": "predictor",
                "nullable": bool(series.isna().any()),
                "allowed_values": _allowed_values(series),
                "observed_min": _json_scalar(numeric.min()) if numeric.notna().any() else None,
                "observed_max": _json_scalar(numeric.max()) if numeric.notna().any() else None,
            }
        )

    return {
        "schema_version": "1.0.0",
        "ordered_features": feature_names,
        "required_columns": feature_names,
        "identifier_columns": ["record_id"],
        "categorical_features": semantic_categorical,
        "numeric_features": [
            feature for feature in feature_names if feature not in semantic_categorical
        ],
        "integer_constrained_features": integer_constrained,
        "target": {
            "name": "Default_Flag",
            "exported_name": "actual_label",
            "dtype": "int64",
            "positive_class": 1,
            "meaning": "1 means next-month credit-card default / bad outcome.",
        },
        "features": feature_definitions,
        "source_reference_split": reference_split,
        "observed_range_policy": (
            "Minima and maxima are observed on the exported held-out reference split. "
            "They are monitoring baselines, not universal business-validity limits."
        ),
    }


def _category_frequencies(series: pd.Series) -> str | None:
    if series.nunique(dropna=True) > 50:
        return None
    counts = series.value_counts(dropna=False).head(50)
    payload = {
        "__missing__" if pd.isna(value) else str(_json_scalar(value)): int(count)
        for value, count in counts.items()
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _reference_feature_summary(
    reference_features: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    quantiles = {
        "p01": 0.01,
        "p05": 0.05,
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p95": 0.95,
        "p99": 0.99,
    }
    rows = []
    for feature in feature_names:
        series = reference_features[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_non_null = numeric.dropna()
        row: dict[str, Any] = {
            "feature": feature,
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
            "mean": float(numeric_non_null.mean()) if not numeric_non_null.empty else None,
            "std": float(numeric_non_null.std()) if not numeric_non_null.empty else None,
            "min": float(numeric_non_null.min()) if not numeric_non_null.empty else None,
        }
        for label, quantile in quantiles.items():
            row[label] = (
                float(numeric_non_null.quantile(quantile))
                if not numeric_non_null.empty
                else None
            )
        row["max"] = float(numeric_non_null.max()) if not numeric_non_null.empty else None
        row["category_frequencies_json"] = _category_frequencies(series)
        rows.append(row)
    return pd.DataFrame(rows)


def _threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype("int8")
    sample_count = int(len(labels))
    positive_count = int(labels.sum())
    predicted_positive_count = int(predictions.sum())
    undefined: list[str] = []

    precision: float | None
    if predicted_positive_count == 0:
        precision = None
        undefined.append("precision")
    else:
        precision = float(precision_score(labels, predictions))

    recall: float | None
    if positive_count == 0:
        recall = None
        undefined.append("recall")
    else:
        recall = float(recall_score(labels, predictions))

    if precision is None or recall is None or precision + recall == 0:
        f1: float | None = None
        undefined.append("f1")
    else:
        f1 = float(f1_score(labels, predictions))

    if np.unique(labels).size < 2:
        roc_auc: float | None = None
        undefined.append("roc_auc")
    else:
        roc_auc = float(roc_auc_score(labels, probabilities))

    if positive_count == 0:
        pr_auc: float | None = None
        undefined.append("pr_auc")
    else:
        pr_auc = float(average_precision_score(labels, probabilities))

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "sample_count": sample_count,
        "positive_count": positive_count,
        "positive_rate": float(positive_count / sample_count),
        "predicted_positive_count": predicted_positive_count,
        "predicted_positive_rate": float(predicted_positive_count / sample_count),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "undefined_metrics": undefined,
    }


def _compare_source_metrics(
    reproduced: dict[str, dict[str, Any]],
    source_default: dict[str, float],
    source_operating: dict[str, float],
) -> tuple[dict[str, dict[str, float]], list[str]]:
    source_by_threshold = {
        "default_0_50": source_default,
        "operating_0_25": source_operating,
    }
    differences: dict[str, dict[str, float]] = {}
    material: list[str] = []
    for threshold_name, source_metrics in source_by_threshold.items():
        differences[threshold_name] = {}
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            source_value = source_metrics.get(metric)
            reproduced_value = reproduced[threshold_name].get(metric)
            if source_value is None or reproduced_value is None:
                continue
            difference = abs(float(reproduced_value) - float(source_value))
            differences[threshold_name][metric] = difference
            if difference > METRIC_COMPARISON_TOLERANCE:
                material.append(
                    f"{threshold_name}.{metric}: absolute difference {difference:.12g}"
                )
    return differences, material


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _manifest_entry(
    path: Path,
    *,
    purpose: str,
    source_path: str | None,
    generation_method: str | None,
    row_count: int | None = None,
    column_count: int | None = None,
) -> dict[str, Any]:
    return {
        "relative_path": _relative(path),
        "purpose": purpose,
        "source_path": source_path,
        "generation_method": generation_method,
        "row_count": row_count,
        "column_count": column_count,
        "file_size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_inventory(
    *,
    source_repo: Path,
    source_artifacts: dict[str, Path],
    feature_names: list[str],
    operating_threshold: float,
    sample_count: int,
    source_probability_difference: float,
    source_versions: dict[str, str],
    export_versions: dict[str, str],
    compatibility_warnings: list[str],
    material_discrepancies: list[str],
) -> None:
    feature_lines = "\n".join(
        f"{position + 1}. `{feature}`" for position, feature in enumerate(feature_names)
    )
    artifact_lines = "\n".join(
        f"- `{name}`: `{path}`" for name, path in source_artifacts.items()
    )
    warning_lines = (
        "\n".join(f"- {warning}" for warning in compatibility_warnings)
        if compatibility_warnings
        else "- None after probability reproduction."
    )
    discrepancy_lines = (
        "\n".join(f"- {item}" for item in material_discrepancies)
        if material_discrepancies
        else "- None above the `1e-6` comparison tolerance."
    )
    inventory = f"""# Credit-default bundle inventory

## Source inspected

- Repository: `{source_repo}`
- Dataset: UCI Default of Credit Card Clients / Taiwan credit-card default (UCI ID 350)
- Target: `Default_Flag`; positive class `1` means next-month default
- Selected model: `xgboost_public`
- Selected feature set: `application_public`
- Operating threshold: `{operating_threshold}` from
  `reports/model_validation/selected_recall_policy.json`

## Authoritative source artifacts

{artifact_lines}

The training summary selects `models/xgboost_public.pkl`; the selected-policy JSON is
authoritative for the operating threshold; the processed UCI table and source split code
reconstruct the held-out rows; the stored prediction CSV and model reports provide
independent reproduction checks.

## Inference features ({len(feature_names)})

{feature_lines}

## Reference selection

The bundle uses the untouched group-aware stratified random held-out test split
(`test_size=0.20`, `random_state=42`) reconstructed by the source repository's split
function. It contains `{sample_count}` rows. `record_id` is the zero-based row position
from the authoritative processed CSV because that CSV does not retain the original UCI
identifier. The source stored labels match exactly and the maximum absolute difference
against its stored probabilities is `{source_probability_difference:.12g}`.

## Serialization and compatibility

The source artifact is a complete fitted scikit-learn `Pipeline` containing a
`ColumnTransformer` preprocessor and `XGBClassifier`. It is exported as
`artifacts/models/credit_default_pipeline.joblib`; normal monitoring runtime does not
import sibling-repository modules.

- Source environment at export: `{json.dumps(source_versions, sort_keys=True)}`
- Monitoring export environment: `{json.dumps(export_versions, sort_keys=True)}`

Compatibility findings:

{warning_lines}

The source repository does not contain an immutable training lockfile. The source
environment versions above are the current source `.venv` versions, not independently
provable historical training versions. Usability is therefore established by reproducing
all exported reference probabilities, not by version strings alone.

## Metric comparison

{discrepancy_lines}

Metrics reproduced from this bundle are authoritative for future monitoring baselines.

## Bundle strategy

Priority 1 was selected: reuse the existing complete fitted inference pipeline. No model
was retrained. No DNN, fairness, explainability, dashboard, or scenario workflow was run.
The bundle is for replay simulation and is not evidence of live production readiness.
"""
    INVENTORY_DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY_DESTINATION.write_text(inventory, encoding="utf-8", newline="\n")


def export_bundle(source_repo: Path, *, overwrite: bool) -> dict[str, Any]:
    """Create a self-contained monitoring bundle from authoritative source artifacts."""
    source_repo = source_repo.resolve()
    source_artifacts = _resolve_source_artifacts(source_repo)
    existing = [path for path in GENERATED_PATHS if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Bundle outputs already exist. Pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    training_summary = _read_json(source_artifacts["training_summary"])
    selected_policy = _read_json(source_artifacts["selected_policy"])
    model_metrics = _read_json(source_artifacts["model_metrics"])
    operating_threshold = float(selected_policy["selected_threshold"])
    reference_split = (
        "group-aware stratified random held-out test split "
        "(test_size=0.20, random_state=42)"
    )

    split = _source_split(source_repo, source_artifacts["processed_dataset"])
    source_stored_predictions = pd.read_csv(source_artifacts["stored_predictions"])

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        pipeline = joblib.load(source_artifacts["model"])
    load_warnings = [
        f"{type(item.message).__name__}: {item.message}" for item in caught_warnings
    ]

    if not hasattr(pipeline, "predict_proba") or not hasattr(pipeline, "steps"):
        raise TypeError("Selected source artifact is not a complete fitted inference pipeline.")
    feature_names = [str(feature) for feature in pipeline.feature_names_in_.tolist()]
    reported_features = [str(feature) for feature in training_summary["feature_columns"]]
    if feature_names != reported_features or split.X_test.columns.tolist() != feature_names:
        raise ValueError("Model, training summary, and reconstructed split feature order differ.")

    probabilities = np.asarray(pipeline.predict_proba(split.X_test)[:, 1], dtype=float)
    source_probabilities = source_stored_predictions["y_proba"].to_numpy(dtype=float)
    source_labels = source_stored_predictions["y_true"].to_numpy(dtype=int)
    labels = split.y_test.to_numpy(dtype=int)
    if not np.array_equal(labels, source_labels):
        raise ValueError("Reconstructed test labels do not match stored source predictions.")
    source_probability_difference = float(np.max(np.abs(probabilities - source_probabilities)))
    if source_probability_difference > 1e-6:
        raise ValueError(
            "Source pipeline does not reproduce stored probabilities within 1e-6; "
            f"maximum absolute difference={source_probability_difference}."
        )

    record_ids = split.X_test.index.to_numpy(dtype="int64")
    reference_features = split.X_test.copy()
    reference_features.insert(0, "record_id", record_ids)
    reference_labels = pd.DataFrame(
        {"record_id": record_ids, "actual_label": labels.astype("int8")}
    )
    reference_predictions = pd.DataFrame(
        {
            "record_id": record_ids,
            "predicted_probability": probabilities,
            "predicted_class_default_threshold": (
                probabilities >= DEFAULT_THRESHOLD
            ).astype("int8"),
            "predicted_class_operating_threshold": (
                probabilities >= operating_threshold
            ).astype("int8"),
        }
    )

    for destination in GENERATED_PATHS:
        destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_artifacts["model"], MODEL_DESTINATION)
    reference_features.to_parquet(FEATURES_DESTINATION, index=False)
    reference_labels.to_parquet(LABELS_DESTINATION, index=False)
    reference_predictions.to_parquet(PREDICTIONS_DESTINATION, index=False)

    feature_summary = _reference_feature_summary(reference_features, feature_names)
    feature_summary.to_parquet(SUMMARY_DESTINATION, index=False)

    source_versions = _source_environment_versions(source_repo)
    export_versions = _export_environment_versions()
    compatibility_warnings = list(load_warnings)
    compatibility_warnings.append(
        "Source library versions are reconstructed from the current source .venv "
        "because no immutable training lockfile is present."
    )
    for name in sorted(set(source_versions).intersection(export_versions)):
        if source_versions[name] != export_versions[name]:
            compatibility_warnings.append(
                f"{name} version differs: source environment {source_versions[name]}, "
                f"export environment {export_versions[name]}."
            )

    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    metadata = {
        "model_name": "xgboost_public",
        "model_family": "XGBoost binary classifier in a scikit-learn Pipeline",
        "model_version": "1.0.0",
        "source_repository": "credit-default-xai",
        "source_repository_path": str(source_repo),
        "source_artifact_paths": [
            path.relative_to(source_repo).as_posix() for path in source_artifacts.values()
        ],
        "dataset_name": "UCI Default of Credit Card Clients / Taiwan credit-card default",
        "target_column": "Default_Flag",
        "positive_class": 1,
        "feature_count": len(feature_names),
        "operating_threshold": operating_threshold,
        "default_threshold": DEFAULT_THRESHOLD,
        "reference_split": reference_split,
        "reference_sample_count": len(reference_features),
        "inference_artifact_path": _relative(MODEL_DESTINATION),
        "inference_available": True,
        "integration_strategy": INTEGRATION_STRATEGY,
        "training_library_versions": source_versions,
        "export_library_versions": export_versions,
        "bundle_created_at_utc": created_at,
        "production_status": "replay_simulation_only",
        "compatibility_warnings": compatibility_warnings,
        "limitations": [
            "Public academic dataset; not live bank production data.",
            "Reference split is group-aware stratified random because no true timestamp exists.",
            "record_id is the processed CSV row position, not an original customer identifier.",
            "Current source .venv versions are a proxy for historical training versions.",
            "No live scoring, drift monitoring, retraining, or automated decisions are enabled.",
        ],
    }
    _write_json(METADATA_DESTINATION, metadata)
    _write_json(
        SCHEMA_DESTINATION,
        _feature_schema(reference_features, feature_names, reference_split),
    )

    threshold_metrics = {
        "default_0_50": _threshold_metrics(labels, probabilities, DEFAULT_THRESHOLD),
        "operating_0_25": _threshold_metrics(labels, probabilities, operating_threshold),
    }
    source_default_metrics = {
        name: float(value) for name, value in model_metrics["metrics"].items()
    }
    source_operating_metrics = {
        name: float(value)
        for name, value in selected_policy["test_metrics"].items()
        if isinstance(value, int | float)
    }
    metric_differences, material_discrepancies = _compare_source_metrics(
        threshold_metrics,
        source_default_metrics,
        source_operating_metrics,
    )
    reference_metrics = {
        "metrics_version": "1.0.0",
        "reference_split": reference_split,
        "sample_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "thresholds": threshold_metrics,
        "authoritative_source_metrics": {
            "default_0_50": source_default_metrics,
            "operating_0_25": source_operating_metrics,
        },
        "metric_absolute_differences": metric_differences,
        "material_discrepancies": material_discrepancies,
        "comparison_tolerance": METRIC_COMPARISON_TOLERANCE,
    }
    _write_json(METRICS_DESTINATION, reference_metrics)

    _write_inventory(
        source_repo=source_repo,
        source_artifacts=source_artifacts,
        feature_names=feature_names,
        operating_threshold=operating_threshold,
        sample_count=len(reference_features),
        source_probability_difference=source_probability_difference,
        source_versions=source_versions,
        export_versions=export_versions,
        compatibility_warnings=compatibility_warnings,
        material_discrepancies=material_discrepancies,
    )

    manifest_files = [
        _manifest_entry(
            MODEL_DESTINATION,
            purpose="Complete fitted preprocessing and XGBoost inference pipeline.",
            source_path=source_artifacts["model"].relative_to(source_repo).as_posix(),
            generation_method="Byte-for-byte copy of the selected source artifact.",
        ),
        _manifest_entry(
            METADATA_DESTINATION,
            purpose="Model identity, threshold, environment, and provenance metadata.",
            source_path=None,
            generation_method="Generated from authoritative source reports and environments.",
        ),
        _manifest_entry(
            SCHEMA_DESTINATION,
            purpose="Exact ordered inference feature and target contract.",
            source_path=None,
            generation_method="Generated from pipeline feature_names_in_ and held-out data.",
        ),
        _manifest_entry(
            METRICS_DESTINATION,
            purpose="Reproduced held-out reference metrics at both decision thresholds.",
            source_path=None,
            generation_method="Calculated from exported labels and pipeline probabilities.",
        ),
        _manifest_entry(
            SUMMARY_DESTINATION,
            purpose="One-row-per-feature reference distribution summary.",
            source_path=None,
            generation_method="Calculated from exported held-out inference features.",
            row_count=len(feature_summary),
            column_count=len(feature_summary.columns),
        ),
        _manifest_entry(
            FEATURES_DESTINATION,
            purpose="Held-out reference identifiers and ordered inference predictors.",
            source_path=source_artifacts["processed_dataset"]
            .relative_to(source_repo)
            .as_posix(),
            generation_method="Source group-aware test split reconstructed without retraining.",
            row_count=len(reference_features),
            column_count=len(reference_features.columns),
        ),
        _manifest_entry(
            LABELS_DESTINATION,
            purpose="Held-out reference identifiers and actual default labels.",
            source_path=source_artifacts["processed_dataset"]
            .relative_to(source_repo)
            .as_posix(),
            generation_method="Target values selected by the reconstructed source test split.",
            row_count=len(reference_labels),
            column_count=len(reference_labels.columns),
        ),
        _manifest_entry(
            PREDICTIONS_DESTINATION,
            purpose="Pipeline probabilities and predictions at 0.50 and 0.25.",
            source_path=source_artifacts["stored_predictions"]
            .relative_to(source_repo)
            .as_posix(),
            generation_method="Regenerated with the exported pipeline and cross-checked.",
            row_count=len(reference_predictions),
            column_count=len(reference_predictions.columns),
        ),
        _manifest_entry(
            INVENTORY_DESTINATION,
            purpose="Factual source inventory and bundle-strategy record.",
            source_path=None,
            generation_method="Generated from the completed source inspection.",
        ),
    ]

    source_commit = _git_output(source_repo, "rev-parse", "HEAD")
    source_status = _git_output(source_repo, "status", "--porcelain")
    manifest = {
        "bundle_version": "1.0.0",
        "created_at_utc": created_at,
        "source_repository": "credit-default-xai",
        "source_repository_path": str(source_repo),
        "source_git_commit": source_commit,
        "source_git_dirty": None if source_status is None else bool(source_status),
        "export_status": "complete",
        "integration_strategy": INTEGRATION_STRATEGY,
        "inference_available": True,
        "compatibility_warnings": compatibility_warnings,
        "manifest_self_checksum": None,
        "manifest_self_checksum_reason": (
            "The manifest cannot contain its own stable SHA-256 checksum without creating "
            "a self-referential value. All other exported payload files are covered."
        ),
        "files": manifest_files,
    }
    _write_json(MANIFEST_DESTINATION, manifest)

    return {
        "model": "xgboost_public",
        "strategy": INTEGRATION_STRATEGY,
        "feature_count": len(feature_names),
        "reference_sample_count": len(reference_features),
        "operating_threshold": operating_threshold,
        "source_probability_max_abs_diff": source_probability_difference,
        "material_metric_discrepancies": material_discrepancies,
        "exported_payload_files": len(manifest_files) + 1,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a self-contained credit-default monitoring bundle."
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=DEFAULT_SOURCE_REPO,
        help=f"Source credit-default-xai repository (default: {DEFAULT_SOURCE_REPO}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated bundle files.",
    )
    return parser


def main() -> int:
    """Run the bundle export and print a compact summary."""
    args = _parser().parse_args()
    try:
        summary = export_bundle(args.source_repo, overwrite=args.overwrite)
    except Exception as exc:
        print(f"Bundle export failed: {exc}", file=sys.stderr)
        return 1
    print("Credit-default monitoring bundle exported successfully.")
    for name, value in summary.items():
        print(f"  {name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
