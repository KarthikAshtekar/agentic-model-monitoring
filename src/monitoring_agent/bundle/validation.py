"""Integrity, alignment, leakage, threshold, and inference checks for the bundle."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from monitoring_agent.bundle.loader import CreditDefaultBundle
from monitoring_agent.bundle.schemas import (
    BundleValidationResult,
    ValidationIssue,
)

REQUIRED_RELATIVE_PATHS = (
    "artifacts/models/credit_default_pipeline.joblib",
    "artifacts/metadata/model_metadata.json",
    "artifacts/metadata/feature_schema.json",
    "artifacts/metadata/bundle_manifest.json",
    "artifacts/baselines/reference_metrics.json",
    "artifacts/baselines/reference_feature_summary.parquet",
    "data/reference/reference_features.parquet",
    "data/reference/reference_labels.parquet",
    "data/reference/reference_predictions.parquet",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle(
    bundle: CreditDefaultBundle | None = None,
    *,
    probability_tolerance: float = 1e-7,
) -> BundleValidationResult:
    """Run complete bundle checks and return errors, warnings, and check status."""
    resolved_bundle = bundle or CreditDefaultBundle()
    checks: dict[str, bool] = {}
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    details: dict[str, object] = {}

    def record(check: str, condition: bool, message: str) -> None:
        checks[check] = bool(condition)
        if not condition:
            errors.append(ValidationIssue(check=check, message=message))

    missing = [
        relative
        for relative in REQUIRED_RELATIVE_PATHS
        if not (resolved_bundle.root / relative).is_file()
    ]
    record("required_files", not missing, f"Missing required files: {missing}")
    if missing:
        return BundleValidationResult(
            valid=False,
            checks=checks,
            errors=errors,
            warnings=warnings,
            details={"missing_files": missing},
        )

    try:
        metadata = resolved_bundle.load_metadata()
        feature_schema = resolved_bundle.load_feature_schema()
        manifest = resolved_bundle.load_manifest()
        reference_metrics = resolved_bundle.load_reference_metrics()
    except Exception as exc:
        record("json_schemas", False, f"JSON schema validation failed: {exc}")
        return BundleValidationResult(
            valid=False,
            checks=checks,
            errors=errors,
            warnings=warnings,
            details=details,
        )
    record("json_schemas", True, "")

    features = resolved_bundle.load_reference_features()
    labels = resolved_bundle.load_reference_labels()
    predictions = resolved_bundle.load_reference_predictions()

    expected_feature_columns = ["record_id", *feature_schema.ordered_features]
    record(
        "feature_order",
        features.columns.tolist() == expected_feature_columns,
        "Reference feature columns do not match record_id plus the exact schema order.",
    )
    record(
        "feature_schema_positions",
        [feature.name for feature in feature_schema.features]
        == feature_schema.ordered_features
        and [feature.position for feature in feature_schema.features]
        == list(range(len(feature_schema.features))),
        "Feature definitions do not match their ordered positions.",
    )

    target_names = {feature_schema.target.name, feature_schema.target.exported_name}
    leaked = target_names.intersection(features.columns)
    record(
        "target_leakage",
        not leaked,
        f"Target columns found in inference features: {sorted(leaked)}",
    )
    record(
        "reference_columns",
        labels.columns.tolist() == ["record_id", "actual_label"]
        and predictions.columns.tolist()
        == [
            "record_id",
            "predicted_probability",
            "predicted_class_default_threshold",
            "predicted_class_operating_threshold",
        ],
        "Reference label or prediction columns do not match the bundle contract.",
    )

    unique_ids = (
        features["record_id"].is_unique
        and labels["record_id"].is_unique
        and predictions["record_id"].is_unique
    )
    record("unique_record_ids", unique_ids, "Record IDs must be unique in every table.")
    aligned = (
        features["record_id"].tolist()
        == labels["record_id"].tolist()
        == predictions["record_id"].tolist()
    )
    record("record_alignment", aligned, "Reference record IDs are not aligned one-to-one.")

    probabilities = predictions["predicted_probability"].to_numpy(dtype=float)
    probability_valid = bool(
        np.isfinite(probabilities).all()
        and np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all()
    )
    record(
        "probability_bounds",
        probability_valid,
        "Stored probabilities must be finite and within [0, 1].",
    )

    expected_default = (probabilities >= metadata.default_threshold).astype("int8")
    expected_operating = (probabilities >= metadata.operating_threshold).astype("int8")
    threshold_predictions_valid = bool(
        np.array_equal(
            expected_default,
            predictions["predicted_class_default_threshold"].to_numpy(dtype="int8"),
        )
        and np.array_equal(
            expected_operating,
            predictions["predicted_class_operating_threshold"].to_numpy(dtype="int8"),
        )
    )
    record(
        "threshold_predictions",
        threshold_predictions_valid,
        "Stored class predictions do not match probabilities and recorded thresholds.",
    )

    row_counts_valid = (
        len(features)
        == len(labels)
        == len(predictions)
        == metadata.reference_sample_count
        == reference_metrics.sample_count
    )
    record("row_counts", row_counts_valid, "Reference row counts disagree with metadata.")

    manifest_paths = {entry.relative_path for entry in manifest.files}
    payload_paths = set(REQUIRED_RELATIVE_PATHS) - {
        "artifacts/metadata/bundle_manifest.json"
    }
    payload_paths.add("docs/credit_default_bundle_inventory.md")
    record(
        "manifest_coverage",
        manifest_paths == payload_paths,
        "Manifest coverage does not match the exported payload files.",
    )

    integrity_valid = True
    for entry in manifest.files:
        candidate = (resolved_bundle.root / entry.relative_path).resolve()
        try:
            candidate.relative_to(resolved_bundle.root)
        except ValueError:
            integrity_valid = False
            errors.append(
                ValidationIssue(
                    check="manifest_integrity",
                    message=f"Manifest path escapes bundle root: {entry.relative_path}",
                )
            )
            continue
        if (
            not candidate.is_file()
            or candidate.stat().st_size != entry.file_size_bytes
            or _sha256(candidate) != entry.sha256
        ):
            integrity_valid = False
            errors.append(
                ValidationIssue(
                    check="manifest_integrity",
                    message=f"Size or checksum mismatch: {entry.relative_path}",
                )
            )
    checks["manifest_integrity"] = integrity_valid

    inference_max_abs_diff: float | None = None
    if metadata.inference_available:
        try:
            inference_features = features[feature_schema.ordered_features]
            reproduced = resolved_bundle.predict_probabilities(inference_features)
            inference_max_abs_diff = float(np.max(np.abs(reproduced - probabilities)))
            inference_valid = bool(
                np.allclose(
                    reproduced,
                    probabilities,
                    atol=probability_tolerance,
                    rtol=probability_tolerance,
                )
            )
            record(
                "model_probability_reproduction",
                inference_valid,
                "Loaded pipeline does not reproduce stored reference probabilities.",
            )
        except Exception as exc:
            record(
                "model_probability_reproduction",
                False,
                f"Loaded pipeline inference failed: {exc}",
            )
    else:
        checks["model_probability_reproduction"] = True
        warnings.append(
            ValidationIssue(
                check="model_probability_reproduction",
                message="Skipped because metadata marks live inference as unavailable.",
            )
        )

    for warning in manifest.compatibility_warnings:
        warnings.append(ValidationIssue(check="compatibility", message=warning))

    details.update(
        {
            "reference_sample_count": len(features),
            "feature_count": len(feature_schema.ordered_features),
            "operating_threshold": metadata.operating_threshold,
            "inference_max_probability_abs_diff": inference_max_abs_diff,
            "manifest_file_count": len(manifest.files),
        }
    )
    return BundleValidationResult(
        valid=not errors and all(checks.values()),
        checks=checks,
        errors=errors,
        warnings=warnings,
        details=details,
    )
