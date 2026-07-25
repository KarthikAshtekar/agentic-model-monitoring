"""Compact deterministic evidence selection for LLM consumption."""

from __future__ import annotations

from typing import Any

from monitoring_agent.monitoring.schemas import EvidenceItem, MonitoringRunResult

MAX_EVIDENCE_RECORDS = 30
NORMAL_PASS_LIMIT = 10


def _serialise_evidence(item: EvidenceItem) -> dict[str, Any]:
    return item.model_dump(mode="json")


def _system_evidence(result: MonitoringRunResult) -> list[dict[str, Any]]:
    """Materialise key run-level facts as compact, citable evidence records."""
    blocked_status = "critical" if result.batch_blocked else "pass"
    blocked_severity = "critical" if result.batch_blocked else "info"
    performance_status = "pass" if result.performance.evaluated else "not_evaluated"
    performance_severity = "info" if result.performance.evaluated else "high"
    return [
        {
            "evidence_id": "SYSTEM-BATCH-VALIDITY",
            "domain": "system",
            "metric": "batch_valid",
            "status": "pass" if result.batch_valid else "critical",
            "severity": "info" if result.batch_valid else "critical",
            "observed_value": result.batch_valid,
            "reference_value": True,
            "threshold": None,
            "feature": None,
            "message": "Deterministic data-quality batch validity.",
            "details": {},
            "source": "monitoring_run_result",
        },
        {
            "evidence_id": "SYSTEM-BATCH-BLOCKING",
            "domain": "system",
            "metric": "batch_blocked",
            "status": blocked_status,
            "severity": blocked_severity,
            "observed_value": result.batch_blocked,
            "reference_value": False,
            "threshold": None,
            "feature": None,
            "message": "Deterministic inference blocking decision.",
            "details": {},
            "source": "monitoring_run_result",
        },
        {
            "evidence_id": "SYSTEM-LABEL-AVAILABILITY",
            "domain": "system",
            "metric": "labels_available",
            "status": "pass" if result.labels_available else "not_evaluated",
            "severity": "info" if result.labels_available else "medium",
            "observed_value": result.labels_available,
            "reference_value": True,
            "threshold": None,
            "feature": None,
            "message": "Whether labels were supplied for this monitoring batch.",
            "details": {},
            "source": "monitoring_run_result",
        },
        {
            "evidence_id": "SYSTEM-LABELLED-SAMPLE-SIZE",
            "domain": "system",
            "metric": "labelled_sample_size",
            "status": performance_status,
            "severity": performance_severity,
            "observed_value": result.performance.sample_count,
            "reference_value": None,
            "threshold": 200,
            "feature": None,
            "message": "Labelled sample size used by deterministic performance evaluation.",
            "details": {
                "performance_evaluated": result.performance.evaluated,
                "reason_not_evaluated": result.performance.reason_not_evaluated,
            },
            "source": "monitoring_run_result",
        },
    ]


def _pass_priority(item: EvidenceItem) -> tuple[int, str]:
    domain_rank = {
        "performance": 0,
        "prediction_drift": 1,
        "feature_drift": 2,
        "data_quality": 3,
        "system": 4,
    }
    return domain_rank[item.domain], item.evidence_id


def build_evidence_packet(
    result: MonitoringRunResult | dict[str, Any],
    *,
    max_records: int = MAX_EVIDENCE_RECORDS,
) -> dict[str, Any]:
    """Select high-signal evidence without recalculating or modifying its values."""
    monitoring_result = (
        result
        if isinstance(result, MonitoringRunResult)
        else MonitoringRunResult.model_validate(result)
    )
    critical = sorted(
        (item for item in monitoring_result.evidence if item.status == "critical"),
        key=lambda item: item.evidence_id,
    )
    warnings = sorted(
        (item for item in monitoring_result.evidence if item.status == "warning"),
        key=lambda item: item.evidence_id,
    )
    mandatory = [_serialise_evidence(item) for item in [*critical, *warnings]]
    system = _system_evidence(monitoring_result)
    if len(mandatory) + len(system) > max_records:
        raise ValueError(
            "Evidence cap is too small to retain every critical, warning, and system record."
        )

    selected = [*mandatory, *system]
    selected_ids = {item["evidence_id"] for item in selected}
    passes = sorted(
        (
            item
            for item in monitoring_result.evidence
            if item.status == "pass" and item.evidence_id not in selected_ids
        ),
        key=_pass_priority,
    )
    pass_limit = min(NORMAL_PASS_LIMIT, max_records - len(selected))
    selected.extend(_serialise_evidence(item) for item in passes[:pass_limit])

    ids = [item["evidence_id"] for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Selected evidence IDs must remain unique.")
    return {
        "case_summary": {
            "run_id": monitoring_result.run_id,
            "scenario_name": monitoring_result.scenario_name,
            "model_name": monitoring_result.model_name,
            "model_version": monitoring_result.model_version,
            "batch_valid": monitoring_result.batch_valid,
            "batch_blocked": monitoring_result.batch_blocked,
            "labels_available": monitoring_result.labels_available,
            "performance_evaluated": monitoring_result.performance.evaluated,
            "labelled_sample_size": monitoring_result.performance.sample_count,
            "incident_candidates": monitoring_result.incident_candidates,
            "overall_severity": monitoring_result.overall_severity,
        },
        "evidence": selected,
    }
