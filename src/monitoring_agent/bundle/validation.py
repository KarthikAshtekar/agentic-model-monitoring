"""Generic integrity, alignment, leakage, threshold, and inference checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from monitoring_agent.bundle.schemas import BundleValidationResult, ValidationIssue
from monitoring_agent.models.bundle import CreditDefaultBundle, RegisteredModelBundle

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


def _relative(bundle: RegisteredModelBundle, path: Path) -> str:
    return path.resolve().relative_to(bundle.root).as_posix()


def _required_paths(bundle: RegisteredModelBundle) -> list[Path]:
    candidates = [
        bundle.metadata_path,
        bundle.feature_schema_path,
        bundle.manifest_path,
        bundle.reference_metrics_path,
        bundle.reference_features_path,
        bundle.reference_labels_path,
        bundle.reference_predictions_path,
        bundle.reference_feature_summary_path,
    ]
    if bundle.inference_available and bundle.model_path is not None:
        candidates.append(bundle.model_path)
    return candidates


def validate_bundle(
    bundle: RegisteredModelBundle | None = None,
    *,
    probability_tolerance: float = 2e-7,
) -> BundleValidationResult:
    """Run complete validation for any registered binary-classification bundle."""
    resolved = bundle or CreditDefaultBundle()
    checks: dict[str, bool] = {}
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    details: dict[str, object] = {}

    def record(check: str, condition: bool, message: str) -> None:
        checks[check] = bool(condition)
        if not condition:
            errors.append(ValidationIssue(check=check, message=message))

    required = _required_paths(resolved)
    missing = [_relative(resolved, path) for path in required if not path.is_file()]
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
        metadata = resolved.load_metadata()
        feature_schema = resolved.load_feature_schema()
        manifest = resolved.load_manifest()
        reference_metrics = resolved.load_reference_metrics()
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
    registered = resolved.registered_manifest
    features = resolved.load_reference_features()
    labels = resolved.load_reference_labels()
    predictions = resolved.load_reference_predictions()
    identifier = registered.data_contract.identifier_column
    expected_feature_columns = [identifier, *feature_schema.ordered_features]
    record(
        "feature_order",
        features.columns.tolist() == expected_feature_columns,
        "Reference feature columns do not match identifier plus exact schema order.",
    )
    record(
        "feature_schema_positions",
        [item.name for item in feature_schema.features]
        == feature_schema.ordered_features
        and [item.position for item in feature_schema.features]
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
        labels.columns.tolist() == [identifier, "actual_label"]
        and predictions.columns.tolist()
        == [
            identifier,
            "predicted_probability",
            "predicted_class_default_threshold",
            "predicted_class_operating_threshold",
        ],
        "Reference label or prediction columns do not match the bundle contract.",
    )
    unique_ids = (
        features[identifier].is_unique
        and labels[identifier].is_unique
        and predictions[identifier].is_unique
    )
    record("unique_record_ids", unique_ids, "Record IDs must be unique.")
    aligned = (
        features[identifier].tolist()
        == labels[identifier].tolist()
        == predictions[identifier].tolist()
    )
    record("record_alignment", aligned, "Reference IDs are not aligned one-to-one.")

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
    record(
        "threshold_predictions",
        bool(
            np.array_equal(
                expected_default,
                predictions[
                    "predicted_class_default_threshold"
                ].to_numpy(dtype="int8"),
            )
            and np.array_equal(
                expected_operating,
                predictions[
                    "predicted_class_operating_threshold"
                ].to_numpy(dtype="int8"),
            )
        ),
        "Stored class predictions do not match registered thresholds.",
    )
    row_counts_valid = (
        len(features)
        == len(labels)
        == len(predictions)
        == metadata.reference_sample_count
        == reference_metrics.sample_count
    )
    record("row_counts", row_counts_valid, "Reference row counts disagree.")
    record(
        "registered_contract",
        metadata.feature_count == len(registered.data_contract.ordered_features)
        and np.isclose(
            metadata.default_threshold,
            registered.prediction_contract.default_threshold,
        )
        and np.isclose(
            metadata.operating_threshold,
            registered.prediction_contract.operating_threshold,
        ),
        "Registered manifest and exported metadata disagree.",
    )

    manifest_paths = {entry.relative_path for entry in manifest.files}
    required_payload_paths = {
        _relative(resolved, path)
        for path in required
        if path != resolved.manifest_path
    }
    record(
        "manifest_coverage",
        required_payload_paths <= manifest_paths,
        "Bundle manifest does not cover every required payload file.",
    )
    integrity_valid = True
    for entry in manifest.files:
        candidate = (resolved.root / entry.relative_path).resolve()
        try:
            candidate.relative_to(resolved.root)
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
    if resolved.inference_available:
        try:
            inference_features = features[feature_schema.ordered_features]
            reproduced = resolved.predict_probabilities(inference_features)
            inference_max_abs_diff = float(np.max(np.abs(reproduced - probabilities)))
            record(
                "model_probability_reproduction",
                bool(
                    np.allclose(
                        reproduced,
                        probabilities,
                        atol=probability_tolerance,
                        rtol=probability_tolerance,
                    )
                ),
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
                message="Skipped because the bundle is scored-reference-only.",
            )
        )
    for warning in manifest.compatibility_warnings:
        warnings.append(ValidationIssue(check="compatibility", message=warning))
    details.update(
        {
            "model_id": resolved.model_id,
            "reference_sample_count": len(features),
            "feature_count": len(feature_schema.ordered_features),
            "operating_threshold": metadata.operating_threshold,
            "inference_max_probability_abs_diff": inference_max_abs_diff,
            "manifest_file_count": len(manifest.files),
            "bundle_mode": registered.provenance.bundle_mode,
        }
    )
    return BundleValidationResult(
        valid=not errors and all(checks.values()),
        checks=checks,
        errors=errors,
        warnings=warnings,
        details=details,
    )
