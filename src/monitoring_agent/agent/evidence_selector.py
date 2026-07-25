"""Compact deterministic evidence selection for LLM consumption."""

from __future__ import annotations

from typing import Any

from monitoring_agent.monitoring.schemas import EvidenceItem, MonitoringRunResult

MAX_EVIDENCE_RECORDS = 30
NORMAL_PASS_LIMIT = 10


def _serialise_evidence(item: EvidenceItem) -> dict[str, Any]:
    return item.model_dump(mode="json")


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
    existing_system = sorted(
        (
            item
            for item in monitoring_result.evidence
            if item.domain == "system"
            and item.status not in {"critical", "warning"}
        ),
        key=lambda item: item.evidence_id,
    )
    mandatory_items = [*critical, *warnings, *existing_system]
    mandatory_by_id = {item.evidence_id: item for item in mandatory_items}
    mandatory = [
        _serialise_evidence(item) for item in mandatory_by_id.values()
    ]
    if len(mandatory) > max_records:
        raise ValueError(
            "Evidence cap is too small to retain every critical, warning, and existing "
            "system evidence record."
        )

    selected = list(mandatory)
    selected_ids = {item["evidence_id"] for item in selected}
    if monitoring_result.incident_candidates == ["normal_operation"]:
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
            "model_id": monitoring_result.model_id,
            "display_name": monitoring_result.display_name,
            "model_name": monitoring_result.model_name,
            "model_version": monitoring_result.model_version,
            "task_type": monitoring_result.task_type,
            "domain_id": monitoring_result.domain_id,
            "use_case": monitoring_result.use_case,
            "positive_outcome": monitoring_result.positive_outcome,
            "prediction_unit": monitoring_result.prediction_unit,
            "operating_threshold": monitoring_result.operating_threshold,
            "allowed_action_types": monitoring_result.allowed_action_types,
            "prohibited_claims": monitoring_result.prohibited_claims,
            "batch_valid": monitoring_result.batch_valid,
            "batch_blocked": monitoring_result.batch_blocked,
            "labels_available": monitoring_result.labels_available,
            "performance_evaluated": monitoring_result.performance.evaluated,
            "labelled_sample_size": monitoring_result.performance.sample_count,
            "feature_row_count": monitoring_result.feature_row_count,
            "labelled_row_count": monitoring_result.labelled_row_count,
            "label_coverage_rate": monitoring_result.label_coverage_rate,
            "minimum_labelled_sample_size": (
                monitoring_result.minimum_labelled_sample_size
            ),
            "performance_reason_not_evaluated": (
                monitoring_result.performance.reason_not_evaluated
            ),
            "incident_candidates": monitoring_result.incident_candidates,
            "overall_severity": monitoring_result.overall_severity,
        },
        "evidence": selected,
    }
