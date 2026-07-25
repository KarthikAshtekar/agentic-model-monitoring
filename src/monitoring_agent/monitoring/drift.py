"""Deterministic feature and prediction distribution comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from monitoring_agent.bundle.schemas import FeatureSchema, ModelMetadata
from monitoring_agent.monitoring.evidence import EvidenceRegistry, evidence_id
from monitoring_agent.monitoring.schemas import DriftResult, EvidenceItem

EPSILON = 1e-6
MISSING_CATEGORY = "__MISSING__"


def _normalise_with_epsilon(counts: np.ndarray) -> np.ndarray:
    values = counts.astype(float) + EPSILON
    return values / values.sum()


def population_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    *,
    bin_count: int = 10,
) -> tuple[float, list[float]]:
    """Calculate numeric PSI using unique reference quantile bins."""
    reference_values = pd.to_numeric(pd.Series(reference), errors="coerce")
    current_values = pd.to_numeric(pd.Series(current), errors="coerce")
    reference_values = reference_values[np.isfinite(reference_values)]
    current_values = current_values[np.isfinite(current_values)]
    if reference_values.empty or current_values.empty:
        return 0.0, []

    quantiles = np.linspace(0.0, 1.0, bin_count + 1)
    edges = np.unique(np.quantile(reference_values, quantiles))
    if len(edges) == 1:
        constant = float(edges[0])
        edges = np.array([-np.inf, constant, np.inf])
    else:
        edges = edges.astype(float)
        edges[0] = -np.inf
        edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)
    reference_distribution = _normalise_with_epsilon(reference_counts)
    current_distribution = _normalise_with_epsilon(current_counts)
    psi = np.sum(
        (current_distribution - reference_distribution)
        * np.log(current_distribution / reference_distribution)
    )
    return float(psi), edges.tolist()


def categorical_distribution_metrics(
    reference: pd.Series,
    current: pd.Series,
) -> dict[str, Any]:
    """Calculate category PSI, Jensen-Shannon divergence, and unseen rate."""
    reference_values = reference.astype("object").where(reference.notna(), MISSING_CATEGORY)
    current_values = current.astype("object").where(current.notna(), MISSING_CATEGORY)
    reference_categories = set(reference_values.unique().tolist())
    categories = sorted(
        reference_categories.union(current_values.unique().tolist()),
        key=str,
    )
    reference_counts = np.array(
        [(reference_values == category).sum() for category in categories],
        dtype=float,
    )
    current_counts = np.array(
        [(current_values == category).sum() for category in categories],
        dtype=float,
    )
    reference_distribution = _normalise_with_epsilon(reference_counts)
    current_distribution = _normalise_with_epsilon(current_counts)
    psi = float(
        np.sum(
            (current_distribution - reference_distribution)
            * np.log(current_distribution / reference_distribution)
        )
    )
    js_divergence = float(
        jensenshannon(reference_distribution, current_distribution, base=2.0) ** 2
    )
    unseen_rate = float((~current_values.isin(reference_categories)).mean())
    return {
        "psi": psi,
        "js_divergence": js_divergence,
        "unseen_category_rate": unseen_rate,
        "reference_category_count": len(reference_categories),
        "current_category_count": int(current_values.nunique()),
        "categories": [str(category) for category in categories],
        "reference_distribution": reference_distribution.tolist(),
        "current_distribution": current_distribution.tolist(),
    }


def _level(
    observed: float,
    warning_threshold: float,
    critical_threshold: float,
) -> str:
    if observed >= critical_threshold:
        return "critical"
    if observed >= warning_threshold:
        return "warning"
    return "pass"


def _worst_level(*levels: str) -> str:
    rank = {"pass": 0, "warning": 1, "critical": 2}
    return max(levels, key=lambda level: rank[level])


def _severity(level: str) -> str:
    return {"pass": "info", "warning": "medium", "critical": "high"}[level]


def _numeric_metrics(reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    psi, edges = population_stability_index(reference, current)
    reference_numeric = pd.to_numeric(reference, errors="coerce")
    current_numeric = pd.to_numeric(current, errors="coerce")
    reference_finite = reference_numeric[np.isfinite(reference_numeric)]
    current_finite = current_numeric[np.isfinite(current_numeric)]
    if reference_finite.empty or current_finite.empty:
        ks_statistic = None
        ks_p_value = None
        mean_shift = None
        median_shift = None
    else:
        ks_result = ks_2samp(reference_finite, current_finite, method="auto")
        ks_statistic = float(ks_result.statistic)
        ks_p_value = float(ks_result.pvalue)
        mean_shift = float(current_finite.mean() - reference_finite.mean())
        median_shift = float(current_finite.median() - reference_finite.median())
    return {
        "feature_class": "continuous_numeric",
        "psi": psi,
        "psi_bin_edges": edges,
        "ks_statistic": ks_statistic,
        "ks_p_value": ks_p_value,
        "mean_shift": mean_shift,
        "median_shift": median_shift,
        "reference_missing_rate": float(reference.isna().mean()),
        "current_missing_rate": float(current.isna().mean()),
        "missing_rate_change": float(current.isna().mean() - reference.isna().mean()),
    }


def _risk_decile_distribution(
    reference_probabilities: np.ndarray,
    current_probabilities: np.ndarray,
) -> dict[str, Any]:
    edges = np.unique(
        np.quantile(reference_probabilities, np.linspace(0.0, 1.0, 11))
    ).astype(float)
    if len(edges) == 1:
        edges = np.array([-np.inf, float(edges[0]), np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    reference_counts, _ = np.histogram(reference_probabilities, bins=edges)
    current_counts, _ = np.histogram(current_probabilities, bins=edges)
    return {
        "bin_edges": edges.tolist(),
        "reference_distribution": (
            reference_counts / max(reference_counts.sum(), 1)
        ).tolist(),
        "current_distribution": (
            current_counts / max(current_counts.sum(), 1)
        ).tolist(),
    }


def evaluate_drift(
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    reference_probabilities: np.ndarray,
    current_probabilities: np.ndarray,
    schema: FeatureSchema,
    metadata: ModelMetadata,
    reference_summary: pd.DataFrame,
    config: dict[str, Any],
) -> DriftResult:
    """Calculate feature and prediction drift with structured evidence."""
    registry = EvidenceRegistry()
    feature_metrics: dict[str, dict[str, Any]] = {}
    warning_features = 0
    critical_features = 0
    unique_count_by_feature = dict(
        zip(reference_summary["feature"], reference_summary["unique_count"], strict=True)
    )

    feature_config = config["feature_drift"]
    for feature in schema.ordered_features:
        reference = reference_features[feature]
        current = current_features[feature]
        is_discrete = feature in schema.categorical_features or (
            feature in schema.integer_constrained_features
            and int(unique_count_by_feature[feature])
            <= int(feature_config["discrete_unique_count_max"])
        )

        if is_discrete:
            metrics = categorical_distribution_metrics(reference, current)
            metrics.update(
                {
                    "feature_class": "categorical_or_discrete",
                    "reference_missing_rate": float(reference.isna().mean()),
                    "current_missing_rate": float(current.isna().mean()),
                    "missing_rate_change": float(
                        current.isna().mean() - reference.isna().mean()
                    ),
                }
            )
            psi_level = _level(
                metrics["psi"],
                float(feature_config["psi_warning"]),
                float(feature_config["psi_critical"]),
            )
            js_level = _level(
                metrics["js_divergence"],
                float(feature_config["js_warning"]),
                float(feature_config["js_critical"]),
            )
            unseen_level = _level(
                metrics["unseen_category_rate"],
                float(feature_config["unseen_category_rate_warning"]),
                float(feature_config["unseen_category_rate_critical"]),
            )
            primary_level = _worst_level(psi_level, js_level, unseen_level)
            registry.add(
                EvidenceItem(
                    evidence_id=evidence_id("DRIFT", "PSI", feature),
                    domain="feature_drift",
                    metric="categorical_distribution_drift",
                    status=primary_level,
                    severity=_severity(primary_level),
                    observed_value={
                        "psi": metrics["psi"],
                        "js_divergence": metrics["js_divergence"],
                        "unseen_category_rate": metrics["unseen_category_rate"],
                    },
                    reference_value={"psi": 0.0, "js_divergence": 0.0},
                    threshold={
                        "psi_warning": feature_config["psi_warning"],
                        "psi_critical": feature_config["psi_critical"],
                        "js_warning": feature_config["js_warning"],
                        "js_critical": feature_config["js_critical"],
                    },
                    feature=feature,
                    message=(
                        f"{feature} categorical PSI={metrics['psi']:.4f}, "
                        f"JS={metrics['js_divergence']:.4f}."
                    ),
                    details={"primary_signal": "PSI/JS; unseen rate is supporting"},
                    source="reference_features.parquet",
                )
            )
        else:
            metrics = _numeric_metrics(reference, current)
            primary_level = _level(
                metrics["psi"],
                float(feature_config["psi_warning"]),
                float(feature_config["psi_critical"]),
            )
            registry.add(
                EvidenceItem(
                    evidence_id=evidence_id("DRIFT", "PSI", feature),
                    domain="feature_drift",
                    metric="population_stability_index",
                    status=primary_level,
                    severity=_severity(primary_level),
                    observed_value=metrics["psi"],
                    reference_value=0.0,
                    threshold={
                        "warning": feature_config["psi_warning"],
                        "critical": feature_config["psi_critical"],
                    },
                    feature=feature,
                    message=f"{feature} PSI is {metrics['psi']:.4f}.",
                    details={
                        "mean_shift": metrics["mean_shift"],
                        "median_shift": metrics["median_shift"],
                    },
                    source="reference_features.parquet",
                )
            )
            ks_statistic = metrics["ks_statistic"]
            ks_support_level = (
                "warning"
                if ks_statistic is not None
                and ks_statistic >= float(feature_config["ks_statistic_warning"])
                else "pass"
            )
            registry.add(
                EvidenceItem(
                    evidence_id=evidence_id("DRIFT", "KS", feature),
                    domain="feature_drift",
                    metric="kolmogorov_smirnov_statistic",
                    status=ks_support_level,
                    severity="medium" if ks_support_level == "warning" else "info",
                    observed_value=ks_statistic,
                    reference_value=0.0,
                    threshold={
                        "warning": feature_config["ks_statistic_warning"],
                        "critical_supporting": feature_config["ks_statistic_critical"],
                    },
                    feature=feature,
                    message=(
                        f"{feature} KS statistic is "
                        f"{ks_statistic:.4f} with p={metrics['ks_p_value']:.4g}."
                        if ks_statistic is not None
                        else f"{feature} KS statistic was not evaluable."
                    ),
                    details={
                        "p_value": metrics["ks_p_value"],
                        "interpretation": (
                            "Supporting distribution evidence only; p-value alone "
                            "does not determine material drift."
                        ),
                    },
                    source="reference_features.parquet",
                )
            )

        metrics["primary_status"] = primary_level
        feature_metrics[feature] = metrics
        if primary_level == "critical":
            critical_features += 1
        elif primary_level == "warning":
            warning_features += 1

    prediction_config = config["prediction_drift"]
    prediction_psi, prediction_edges = population_stability_index(
        reference_probabilities,
        current_probabilities,
    )
    prediction_ks = ks_2samp(
        reference_probabilities,
        current_probabilities,
        method="auto",
    )
    reference_default_rate = float(
        (reference_probabilities >= metadata.default_threshold).mean()
    )
    current_default_rate = float(
        (current_probabilities >= metadata.default_threshold).mean()
    )
    reference_operating_rate = float(
        (reference_probabilities >= metadata.operating_threshold).mean()
    )
    current_operating_rate = float(
        (current_probabilities >= metadata.operating_threshold).mean()
    )
    default_rate_change = current_default_rate - reference_default_rate
    operating_rate_change = current_operating_rate - reference_operating_rate
    prediction_metrics = {
        "reference_mean_probability": float(reference_probabilities.mean()),
        "current_mean_probability": float(current_probabilities.mean()),
        "mean_probability_change": float(
            current_probabilities.mean() - reference_probabilities.mean()
        ),
        "reference_median_probability": float(np.median(reference_probabilities)),
        "current_median_probability": float(np.median(current_probabilities)),
        "probability_psi": prediction_psi,
        "probability_psi_bin_edges": prediction_edges,
        "probability_ks_statistic": float(prediction_ks.statistic),
        "probability_ks_p_value": float(prediction_ks.pvalue),
        "reference_positive_rate_default_threshold": reference_default_rate,
        "current_positive_rate_default_threshold": current_default_rate,
        "default_threshold_rate_change": default_rate_change,
        "reference_positive_rate_operating_threshold": reference_operating_rate,
        "current_positive_rate_operating_threshold": current_operating_rate,
        "operating_threshold_rate_change": operating_rate_change,
        "risk_decile_distribution": _risk_decile_distribution(
            reference_probabilities,
            current_probabilities,
        ),
    }

    prediction_psi_level = _level(
        prediction_psi,
        float(prediction_config["probability_psi_warning"]),
        float(prediction_config["probability_psi_critical"]),
    )
    registry.add(
        EvidenceItem(
            evidence_id="PRED-PSI-PROBABILITY",
            domain="prediction_drift",
            metric="prediction_probability_psi",
            status=prediction_psi_level,
            severity=_severity(prediction_psi_level),
            observed_value=prediction_psi,
            reference_value=0.0,
            threshold={
                "warning": prediction_config["probability_psi_warning"],
                "critical": prediction_config["probability_psi_critical"],
            },
            feature=None,
            message=f"Prediction probability PSI is {prediction_psi:.4f}.",
            source="reference_predictions.parquet",
        )
    )

    for threshold_name, change in (
        ("DEFAULT", default_rate_change),
        ("OPERATING", operating_rate_change),
    ):
        rate_level = _level(
            abs(change),
            float(prediction_config["positive_rate_absolute_change_warning"]),
            float(prediction_config["positive_rate_absolute_change_critical"]),
        )
        registry.add(
            EvidenceItem(
                evidence_id=evidence_id("PRED", "RATE", threshold_name),
                domain="prediction_drift",
                metric=f"positive_rate_change_{threshold_name.lower()}",
                status=rate_level,
                severity=_severity(rate_level),
                observed_value=change,
                reference_value=0.0,
                threshold={
                    "warning": prediction_config[
                        "positive_rate_absolute_change_warning"
                    ],
                    "critical": prediction_config[
                        "positive_rate_absolute_change_critical"
                    ],
                },
                feature=None,
                message=(
                    f"{threshold_name.title()}-threshold predicted-positive-rate "
                    f"change is {change:+.4f}."
                ),
                source="reference_predictions.parquet",
            )
        )

    ks_prediction_status = (
        "warning"
        if prediction_ks.statistic
        >= float(prediction_config.get("ks_statistic_warning", 0.10))
        else "pass"
    )
    registry.add(
        EvidenceItem(
            evidence_id="PRED-KS-PROBABILITY",
            domain="prediction_drift",
            metric="prediction_probability_ks",
            status=ks_prediction_status,
            severity="medium" if ks_prediction_status == "warning" else "info",
            observed_value=float(prediction_ks.statistic),
            reference_value=0.0,
            threshold=prediction_config.get("ks_statistic_warning", 0.10),
            feature=None,
            message=(
                f"Prediction KS statistic is {prediction_ks.statistic:.4f}; "
                "it is supporting evidence only."
            ),
            details={"p_value": float(prediction_ks.pvalue)},
            source="reference_predictions.parquet",
        )
    )

    return DriftResult(
        evaluated_feature_count=len(feature_metrics),
        warning_feature_count=warning_features,
        critical_feature_count=critical_features,
        feature_metrics=feature_metrics,
        prediction_metrics=prediction_metrics,
        evidence=registry.items,
    )
