"""Orchestration for deterministic monitoring of replayed scenario batches."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from monitoring_agent.bundle.loader import CreditDefaultBundle
from monitoring_agent.monitoring.data_quality import evaluate_data_quality
from monitoring_agent.monitoring.drift import evaluate_drift
from monitoring_agent.monitoring.evidence import (
    EvidenceRegistry,
    maximum_severity,
)
from monitoring_agent.monitoring.performance import evaluate_performance
from monitoring_agent.monitoring.schemas import (
    DriftResult,
    EvidenceItem,
    MonitoringRunResult,
    PerformanceResult,
)
from monitoring_agent.paths import CONFIG_DIR, PROJECT_ROOT


class MonitoringEngine:
    """Run data-quality, drift, performance, and incident rules without an LLM."""

    def __init__(
        self,
        bundle: CreditDefaultBundle | None = None,
        config_path: Path | str | None = None,
    ) -> None:
        self.bundle = bundle or CreditDefaultBundle()
        self.config_path = Path(config_path or CONFIG_DIR / "monitoring.yaml")
        with self.config_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        self.config: dict[str, Any] = payload["monitoring"]
        self.metadata = self.bundle.load_metadata()
        self.schema = self.bundle.load_feature_schema()
        self.reference_metrics = self.bundle.load_reference_metrics()
        self.reference_features = self.bundle.load_reference_features()
        self.reference_predictions = self.bundle.load_reference_predictions()
        self.reference_summary = self.bundle.load_reference_feature_summary()

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
    def _blocked_performance(reason: str) -> PerformanceResult:
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
            metrics_at_default_threshold={},
            metrics_at_operating_threshold={},
            metric_deltas={},
            evidence=[evidence],
        )

    @staticmethod
    def _run_id(scenario_name: str, features: pd.DataFrame) -> str:
        digest = hashlib.sha256()
        digest.update(scenario_name.encode())
        digest.update(pd.util.hash_pandas_object(features, index=False).values.tobytes())
        return f"MON-{scenario_name.upper().replace('_', '-')}-{digest.hexdigest()[:12].upper()}"

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

        if not candidates and not performance.evaluated:
            warning_signal = any(
                item.domain in {"feature_drift", "prediction_drift"}
                and item.status == "warning"
                for item in evidence
            )
            if warning_signal:
                candidates.append("insufficient_evidence")
        return candidates or ["normal_operation"]

    def run_batch(
        self,
        scenario_name: str,
        features: pd.DataFrame,
        labels: pd.DataFrame | None,
    ) -> MonitoringRunResult:
        """Run one in-memory batch through the deterministic monitoring gates."""
        data_quality = evaluate_data_quality(
            features,
            self.schema,
            self.config["data_quality"],
        )

        if data_quality.batch_blocked:
            drift = self._empty_drift()
            performance = self._blocked_performance(
                "Performance and drift were not evaluated because data quality blocked inference."
            )
        else:
            inference_frame = features.drop(columns=["record_id"])
            probabilities = self.bundle.predict_probabilities(inference_frame)
            drift = evaluate_drift(
                self.reference_features,
                features,
                self.reference_predictions["predicted_probability"].to_numpy(dtype=float),
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
            )

        registry = EvidenceRegistry()
        registry.extend(data_quality.evidence)
        registry.extend(drift.evidence)
        registry.extend(performance.evidence)
        evidence = registry.items
        incidents = self._incident_candidates(
            blocked=data_quality.batch_blocked,
            evidence=evidence,
            drift=drift,
            performance=performance,
        )
        return MonitoringRunResult(
            run_id=self._run_id(scenario_name, features),
            scenario_name=scenario_name,
            created_at_utc=datetime.now(UTC),
            model_name=self.metadata.model_name,
            model_version=self.metadata.model_version,
            operating_threshold=self.metadata.operating_threshold,
            batch_valid=data_quality.batch_valid,
            batch_blocked=data_quality.batch_blocked,
            labels_available=labels is not None,
            data_quality=data_quality,
            drift=drift,
            performance=performance,
            incident_candidates=incidents,
            overall_severity=maximum_severity(evidence),
            evidence=evidence,
            report_paths=[],
        )

    def run_scenario(self, scenario_name: str) -> MonitoringRunResult:
        """Load one generated scenario and monitor it."""
        scenario_dir = PROJECT_ROOT / "data/scenarios" / scenario_name
        feature_path = scenario_dir / "features.parquet"
        label_path = scenario_dir / "labels.parquet"
        if not feature_path.is_file():
            raise FileNotFoundError(f"Scenario features not found: {feature_path}")
        features = pd.read_parquet(feature_path)
        labels = pd.read_parquet(label_path) if label_path.is_file() else None
        return self.run_batch(scenario_name, features, labels)
