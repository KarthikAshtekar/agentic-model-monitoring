"""Model-aware deterministic monitoring orchestration."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from monitoring_agent.adapters.binary_classification import BinaryClassificationAdapter
from monitoring_agent.domains.base import load_domain_policy
from monitoring_agent.models.bundle import RegisteredModelBundle
from monitoring_agent.models.registry import ModelRegistry
from monitoring_agent.monitoring.data_quality import evaluate_data_quality
from monitoring_agent.monitoring.drift import evaluate_drift
from monitoring_agent.monitoring.evidence import EvidenceRegistry, maximum_severity
from monitoring_agent.monitoring.performance import evaluate_performance
from monitoring_agent.monitoring.schemas import (
    DriftResult,
    EvidenceItem,
    MonitoringRunResult,
    PerformanceResult,
)
from monitoring_agent.paths import CONFIG_DIR
from monitoring_agent.scenarios.generator import resolve_scenario_directory


class MonitoringEngine:
    """Run generic binary monitoring with manifest and domain context."""

    def __init__(
        self,
        bundle: RegisteredModelBundle | None = None,
        config_path: Path | str | None = None,
        *,
        model_id: str | None = None,
        adapter: BinaryClassificationAdapter | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.bundle = bundle or RegisteredModelBundle(
            model_id,
            registry=self.registry,
        )
        self.registered = self.bundle.registered_manifest
        self.model_id = self.registered.model_id
        self.adapter = adapter or self.bundle.adapter()
        self.config_path = Path(config_path or CONFIG_DIR / "monitoring.yaml")
        with self.config_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        self.config: dict[str, Any] = payload["monitoring"]
        policy = self.registered.monitoring_policy
        self.config["data_quality"]["minimum_batch_size"] = policy.minimum_batch_size
        self.config["feature_drift"] = dict(policy.feature_drift_thresholds)
        self.config["prediction_drift"] = dict(policy.prediction_drift_thresholds)
        self.config["performance"] = {
            "minimum_labelled_samples": policy.minimum_labelled_samples,
            **policy.performance_thresholds,
        }
        self.domain_policy = load_domain_policy(self.registered.identity.domain_id)
        self.metadata = self.bundle.load_metadata()
        self.schema = self.bundle.load_feature_schema()
        self.reference_metrics = self.bundle.load_reference_metrics()
        self.reference_features = self.bundle.load_reference_features()
        self.reference_predictions = self.bundle.load_reference_predictions()
        self.reference_summary = self.bundle.load_reference_feature_summary()
        self.identifier_column = self.registered.data_contract.identifier_column

    @staticmethod
    def _empty_drift() -> DriftResult:
        return DriftResult(
            evaluated_feature_count=0,
            warning_feature_count=0,
            critical_feature_count=0,
            feature_metrics={},
            prediction_metrics={},
            evidence=[],
        )

    @staticmethod
    def _blocked_performance(
        reason: str,
        *,
        feature_row_count: int,
        labelled_row_count: int,
        minimum_required_sample_size: int,
    ) -> PerformanceResult:
        evidence = EvidenceItem(
            evidence_id="SYSTEM-BATCH-BLOCKED",
            domain="system",
            metric="performance_evaluation_available",
            status="not_evaluated",
            severity="high",
            observed_value=False,
            reference_value=True,
            threshold=None,
            feature=None,
            message=reason,
            source="data_quality_gate",
        )
        return PerformanceResult(
            evaluated=False,
            reason_not_evaluated=reason,
            sample_count=0,
            positive_count=0,
            negative_count=0,
            feature_row_count=feature_row_count,
            labelled_row_count=labelled_row_count,
            label_coverage_rate=(
                labelled_row_count / feature_row_count if feature_row_count else 0.0
            ),
            minimum_required_sample_size=minimum_required_sample_size,
            metrics_at_default_threshold={},
            metrics_at_operating_threshold={},
            metric_deltas={},
            evidence=[evidence],
        )

    @staticmethod
    def _run_id(
        model_id: str,
        scenario_name: str,
        features: pd.DataFrame,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(model_id.encode())
        digest.update(scenario_name.encode())
        digest.update(pd.util.hash_pandas_object(features, index=False).values.tobytes())
        scenario = scenario_name.upper().replace("_", "-")
        model = model_id.upper().replace("_", "-")
        return f"MON-{model}-{scenario}-{digest.hexdigest()[:12].upper()}"

    @staticmethod
    def _incident_candidates(
        *,
        blocked: bool,
        evidence: list[EvidenceItem],
        drift: DriftResult,
        performance: PerformanceResult,
    ) -> list[str]:
        critical_data_quality = any(
            item.domain == "data_quality" and item.status == "critical"
            for item in evidence
        )
        critical_prediction = any(
            item.domain == "prediction_drift" and item.status == "critical"
            for item in evidence
        )
        critical_performance = any(
            item.domain == "performance" and item.status == "critical"
            for item in evidence
        )
        critical_feature = drift.critical_feature_count > 0
        if blocked or critical_data_quality:
            return ["data_quality_failure"]
        candidates: list[str] = []
        if critical_performance and (critical_feature or critical_prediction):
            candidates.append("mixed_incident")
        if critical_performance:
            candidates.append("performance_degradation")
        if critical_feature:
            candidates.append("feature_drift")
        if critical_prediction:
            candidates.append("prediction_drift")
        if not performance.evaluated:
            candidates.append("insufficient_evidence")
        return candidates or ["normal_operation"]

    def run_batch(
        self,
        scenario_name: str,
        features: pd.DataFrame,
        labels: pd.DataFrame | None,
    ) -> MonitoringRunResult:
        data_quality = evaluate_data_quality(
            features,
            self.schema,
            self.config["data_quality"],
        )
        feature_row_count = len(features)
        labelled_row_count = 0 if labels is None else len(labels)
        label_coverage_rate = (
            labelled_row_count / feature_row_count if feature_row_count else 0.0
        )
        minimum_labels = int(self.config["performance"]["minimum_labelled_samples"])
        if data_quality.batch_blocked:
            drift = self._empty_drift()
            performance = self._blocked_performance(
                "Performance and drift were not evaluated because data quality blocked inference.",
                feature_row_count=feature_row_count,
                labelled_row_count=labelled_row_count,
                minimum_required_sample_size=minimum_labels,
            )
        else:
            inference_frame = features[self.schema.ordered_features]
            probabilities = self.adapter.predict_scores(inference_frame).to_numpy(
                dtype=float
            )
            drift = evaluate_drift(
                self.reference_features,
                features,
                self.reference_predictions["predicted_probability"].to_numpy(
                    dtype=float
                ),
                probabilities,
                self.schema,
                self.metadata,
                self.reference_summary,
                self.config,
            )
            performance = evaluate_performance(
                features,
                labels,
                probabilities,
                self.metadata,
                self.reference_metrics,
                self.config["performance"],
                identifier_column=self.identifier_column,
            )

        registry = EvidenceRegistry()
        registry.extend(data_quality.evidence)
        registry.extend(drift.evidence)
        registry.extend(performance.evidence)
        evidence = registry.items
        for item in evidence:
            item.model_id = self.model_id
            item.domain_id = self.registered.identity.domain_id
        incidents = self._incident_candidates(
            blocked=data_quality.batch_blocked,
            evidence=evidence,
            drift=drift,
            performance=performance,
        )
        return MonitoringRunResult(
            run_id=self._run_id(self.model_id, scenario_name, features),
            scenario_name=scenario_name,
            created_at_utc=datetime.now(UTC),
            model_id=self.model_id,
            display_name=self.registered.identity.display_name,
            model_name=self.metadata.model_name,
            model_version=self.registered.identity.model_version,
            task_type=self.registered.identity.task_type,
            domain_id=self.registered.identity.domain_id,
            bundle_mode=self.registered.provenance.bundle_mode,
            reference_sample_count=self.metadata.reference_sample_count,
            feature_count=len(self.schema.ordered_features),
            use_case=self.registered.business_context.use_case,
            positive_outcome=self.registered.business_context.positive_outcome,
            prediction_unit=self.registered.business_context.prediction_unit,
            allowed_action_types=self.registered.governance.allowed_action_types,
            prohibited_claims=self.domain_policy.prohibited_claims,
            safe_business_terminology=self.domain_policy.safe_business_terminology,
            domain_limitations=self.domain_policy.domain_specific_limitations,
            operating_threshold=self.metadata.operating_threshold,
            batch_valid=data_quality.batch_valid,
            batch_blocked=data_quality.batch_blocked,
            labels_available=labels is not None and labelled_row_count > 0,
            feature_row_count=feature_row_count,
            labelled_row_count=labelled_row_count,
            label_coverage_rate=label_coverage_rate,
            minimum_labelled_sample_size=minimum_labels,
            data_quality=data_quality,
            drift=drift,
            performance=performance,
            incident_candidates=incidents,
            overall_severity=maximum_severity(evidence),
            evidence=evidence,
            report_paths=[],
        )

    def run_scenario(self, scenario_name: str) -> MonitoringRunResult:
        scenario_dir = resolve_scenario_directory(self.model_id, scenario_name)
        feature_path = scenario_dir / "features.parquet"
        label_path = scenario_dir / "labels.parquet"
        if not feature_path.is_file():
            raise FileNotFoundError(f"Scenario features not found: {feature_path}")
        features = pd.read_parquet(feature_path)
        labels = pd.read_parquet(label_path) if label_path.is_file() else None
        return self.run_batch(scenario_name, features, labels)
