"""Deterministic schema, identifier, missingness, and range checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from monitoring_agent.bundle.schemas import FeatureSchema
from monitoring_agent.monitoring.evidence import EvidenceRegistry, evidence_id
from monitoring_agent.monitoring.schemas import DataQualityResult, EvidenceItem


def _status(
    observed: float,
    warning_threshold: float,
    critical_threshold: float,
) -> tuple[str, str]:
    if observed >= critical_threshold:
        return "critical", "high"
    if observed >= warning_threshold:
        return "warning", "medium"
    return "pass", "info"


def _dtype_compatible(series: pd.Series, expected_dtype: str) -> bool:
    if expected_dtype.startswith(("int", "float")):
        if not is_numeric_dtype(series):
            return False
        if expected_dtype.startswith("int"):
            values = pd.to_numeric(series.dropna(), errors="coerce")
            return bool(
                not values.isna().any()
                and np.isclose(values.to_numpy(), np.round(values.to_numpy())).all()
            )
        return True
    return str(series.dtype) == expected_dtype


def evaluate_data_quality(
    batch: pd.DataFrame,
    schema: FeatureSchema,
    config: dict[str, Any],
) -> DataQualityResult:
    """Evaluate a feature batch against the exported inference schema."""
    registry = EvidenceRegistry()
    row_count = len(batch)
    column_count = len(batch.columns)
    if len(schema.identifier_columns) != 1:
        raise ValueError("Binary monitoring requires exactly one identifier column.")
    identifier_column = schema.identifier_columns[0]
    expected_columns = [identifier_column, *schema.ordered_features]
    target_columns = {schema.target.name, schema.target.exported_name}

    missing_required = [column for column in expected_columns if column not in batch.columns]
    leaked_targets = [column for column in batch.columns if column in target_columns]
    unexpected = [
        column
        for column in batch.columns
        if column not in expected_columns and column not in target_columns
    ]
    blocked = False

    if row_count == 0:
        registry.add(
            EvidenceItem(
                evidence_id="DQ-EMPTY-BATCH",
                domain="data_quality",
                metric="batch_row_count",
                status="critical",
                severity="critical",
                observed_value=0,
                reference_value=None,
                threshold=1,
                feature=None,
                message="The monitoring batch is empty.",
                source="batch_structure",
            )
        )
        blocked = True

    for column in missing_required:
        registry.add(
            EvidenceItem(
                evidence_id=evidence_id("DQ", "MISSING", column),
                domain="data_quality",
                metric="required_column_present",
                status="critical",
                severity="critical",
                observed_value=False,
                reference_value=True,
                threshold="required",
                feature=column,
                message=f"Required column {column} is missing.",
                source="feature_schema.json",
            )
        )
    if missing_required and config.get("block_on_missing_required_columns", True):
        blocked = True

    if leaked_targets:
        registry.add(
            EvidenceItem(
                evidence_id="DQ-TARGET-LEAKAGE",
                domain="data_quality",
                metric="target_columns_in_inference_batch",
                status="critical",
                severity="critical",
                observed_value=leaked_targets,
                reference_value=[],
                threshold=0,
                feature=None,
                message="Target data is present in the inference feature batch.",
                source="feature_schema.json",
            )
        )
        if config.get("block_on_target_leakage", True):
            blocked = True

    if unexpected:
        status = "critical" if config.get("block_on_unexpected_columns", True) else "warning"
        severity = "high" if status == "critical" else "medium"
        registry.add(
            EvidenceItem(
                evidence_id="DQ-UNEXPECTED-COLUMNS",
                domain="data_quality",
                metric="unexpected_column_count",
                status=status,
                severity=severity,
                observed_value=len(unexpected),
                reference_value=0,
                threshold=0,
                feature=None,
                message=f"Unexpected inference columns were found: {unexpected}.",
                source="feature_schema.json",
            )
        )
        if config.get("block_on_unexpected_columns", True):
            blocked = True

    if not missing_required and not unexpected and not leaked_targets:
        order_matches = batch.columns.tolist() == expected_columns
        if not order_matches:
            registry.add(
                EvidenceItem(
                    evidence_id="DQ-FEATURE-ORDER",
                    domain="data_quality",
                    metric="exact_feature_order",
                    status="critical",
                    severity="high",
                    observed_value=batch.columns.tolist(),
                    reference_value=expected_columns,
                    threshold="exact_match",
                    feature=None,
                    message="Batch columns do not follow the exported feature order.",
                    source="feature_schema.json",
                )
            )
            if config.get("block_on_feature_order_mismatch", True):
                blocked = True

    duplicate_count = 0
    duplicate_rate = 0.0
    if identifier_column in batch.columns and row_count:
        duplicate_count = int(batch[identifier_column].duplicated().sum())
        duplicate_rate = duplicate_count / row_count
        status, severity = _status(
            duplicate_rate,
            float(config["duplicate_rate_warning"]),
            float(config["duplicate_rate_critical"]),
        )
        registry.add(
            EvidenceItem(
                evidence_id="DQ-DUPLICATE-RECORD-ID",
                domain="data_quality",
                metric="duplicate_record_rate",
                status=status,
                severity="critical" if status == "critical" else severity,
                observed_value=duplicate_rate,
                reference_value=0.0,
                threshold={
                    "warning": config["duplicate_rate_warning"],
                    "critical": config["duplicate_rate_critical"],
                },
                feature=identifier_column,
                message=(
                    f"Duplicate record rate is {duplicate_rate:.4f} "
                    f"({duplicate_count} duplicate IDs)."
                ),
                source="batch_identifiers",
            )
        )
        if duplicate_count and config.get("block_on_duplicate_record_ids", True):
            blocked = True

    minimum_batch_size = int(config["minimum_batch_size"])
    if row_count < minimum_batch_size:
        registry.add(
            EvidenceItem(
                evidence_id="DQ-LOW-ROW-COUNT",
                domain="data_quality",
                metric="batch_row_count",
                status="warning",
                severity="medium",
                observed_value=row_count,
                reference_value=None,
                threshold=minimum_batch_size,
                feature=None,
                message=f"Batch has fewer than {minimum_batch_size} rows.",
                source="monitoring.yaml",
            )
        )

    schema_by_name = {feature.name: feature for feature in schema.features}
    feature_results: list[dict[str, Any]] = []
    for feature_name in schema.ordered_features:
        if feature_name not in batch.columns:
            continue
        feature_schema = schema_by_name[feature_name]
        series = batch[feature_name]
        numeric = pd.to_numeric(series, errors="coerce")
        missing_rate = float(series.isna().mean()) if row_count else 0.0
        finite_mask = np.isfinite(numeric.fillna(0).to_numpy(dtype=float))
        infinite_count = int((~finite_mask & numeric.notna().to_numpy()).sum())

        missing_status, missing_severity = _status(
            missing_rate,
            float(config["missing_rate_warning"]),
            float(config["missing_rate_critical"]),
        )
        registry.add(
            EvidenceItem(
                evidence_id=evidence_id("DQ", "MISSING", feature_name),
                domain="data_quality",
                metric="missing_rate",
                status=missing_status,
                severity=missing_severity,
                observed_value=missing_rate,
                reference_value=0.0,
                threshold={
                    "warning": config["missing_rate_warning"],
                    "critical": config["missing_rate_critical"],
                },
                feature=feature_name,
                message=f"{feature_name} missing rate is {missing_rate:.4f}.",
                source="batch_values",
            )
        )

        dtype_compatible = _dtype_compatible(series, feature_schema.dtype)
        if not dtype_compatible:
            registry.add(
                EvidenceItem(
                    evidence_id=evidence_id("DQ", "DTYPE", feature_name),
                    domain="data_quality",
                    metric="dtype_compatible",
                    status="critical",
                    severity="high",
                    observed_value=str(series.dtype),
                    reference_value=feature_schema.dtype,
                    threshold="compatible",
                    feature=feature_name,
                    message=f"{feature_name} has an incompatible dtype.",
                    source="feature_schema.json",
                )
            )

        if series.isna().all():
            registry.add(
                EvidenceItem(
                    evidence_id=evidence_id("DQ", "EMPTY", feature_name),
                    domain="data_quality",
                    metric="completely_empty_feature",
                    status="critical",
                    severity="high",
                    observed_value=True,
                    reference_value=False,
                    threshold=False,
                    feature=feature_name,
                    message=f"{feature_name} is completely empty.",
                    source="batch_values",
                )
            )

        if infinite_count:
            registry.add(
                EvidenceItem(
                    evidence_id=evidence_id("DQ", "INFINITE", feature_name),
                    domain="data_quality",
                    metric="infinite_value_count",
                    status="critical",
                    severity="high",
                    observed_value=infinite_count,
                    reference_value=0,
                    threshold=0,
                    feature=feature_name,
                    message=f"{feature_name} contains infinite numeric values.",
                    source="batch_values",
                )
            )

        valid_numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
        range_violation_count = 0
        range_violation_rate = 0.0
        if (
            feature_schema.observed_min is not None
            and feature_schema.observed_max is not None
            and not valid_numeric.empty
        ):
            violations = (valid_numeric < feature_schema.observed_min) | (
                valid_numeric > feature_schema.observed_max
            )
            range_violation_count = int(violations.sum())
            range_violation_rate = range_violation_count / row_count
            range_status, range_severity = _status(
                range_violation_rate,
                float(config["range_violation_rate_warning"]),
                float(config["range_violation_rate_critical"]),
            )
            registry.add(
                EvidenceItem(
                    evidence_id=evidence_id("DQ", "RANGE", feature_name),
                    domain="data_quality",
                    metric="observed_range_violation_rate",
                    status=range_status,
                    severity=range_severity,
                    observed_value=range_violation_rate,
                    reference_value={
                        "min": feature_schema.observed_min,
                        "max": feature_schema.observed_max,
                    },
                    threshold={
                        "warning": config["range_violation_rate_warning"],
                        "critical": config["range_violation_rate_critical"],
                    },
                    feature=feature_name,
                    message=(
                        f"{feature_name} observed-range violation rate is "
                        f"{range_violation_rate:.4f}."
                    ),
                    details={"violation_count": range_violation_count},
                    source="feature_schema.json",
                )
            )

        integer_violation_count = 0
        if feature_name in schema.integer_constrained_features and not valid_numeric.empty:
            integer_violations = ~np.isclose(
                valid_numeric.to_numpy(), np.round(valid_numeric.to_numpy())
            )
            integer_violation_count = int(integer_violations.sum())
            if integer_violation_count:
                registry.add(
                    EvidenceItem(
                        evidence_id=evidence_id("DQ", "INTEGER", feature_name),
                        domain="data_quality",
                        metric="non_integer_value_count",
                        status="critical",
                        severity="high",
                        observed_value=integer_violation_count,
                        reference_value=0,
                        threshold=0,
                        feature=feature_name,
                        message=f"{feature_name} contains non-integer values.",
                        source="feature_schema.json",
                    )
                )

        feature_results.append(
            {
                "feature": feature_name,
                "dtype": str(series.dtype),
                "dtype_compatible": dtype_compatible,
                "missing_count": int(series.isna().sum()),
                "missing_rate": missing_rate,
                "infinite_count": infinite_count,
                "range_violation_count": range_violation_count,
                "range_violation_rate": range_violation_rate,
                "integer_violation_count": integer_violation_count,
                "completely_empty": bool(series.isna().all()),
            }
        )

    evidence = registry.items
    batch_valid = not blocked and not any(item.status == "critical" for item in evidence)
    return DataQualityResult(
        batch_valid=batch_valid,
        batch_blocked=blocked,
        row_count=row_count,
        column_count=column_count,
        missing_required_columns=missing_required,
        unexpected_columns=unexpected,
        duplicate_record_count=duplicate_count,
        duplicate_record_rate=duplicate_rate,
        feature_results=feature_results,
        evidence=evidence,
    )
