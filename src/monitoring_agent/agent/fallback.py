"""Conservative deterministic recommendation when the LLM path is unusable."""

from __future__ import annotations

from typing import Any

from monitoring_agent.agent.schemas import (
    AgentRecommendation,
    EvidenceBackedClaim,
    RecommendedAction,
)
from monitoring_agent.monitoring.schemas import MonitoringRunResult


def _ids(
    evidence: list[dict[str, Any]],
    *,
    domains: set[str] | None = None,
    statuses: set[str] | None = None,
    limit: int = 6,
) -> list[str]:
    matches = [
        item["evidence_id"]
        for item in evidence
        if (domains is None or item["domain"] in domains)
        and (statuses is None or item["status"] in statuses)
    ]
    return matches[:limit]


def build_fallback_recommendation(
    monitoring_result: MonitoringRunResult | dict[str, Any],
    evidence: list[dict[str, Any]],
) -> AgentRecommendation:
    """Build a valid conservative recommendation from deterministic facts only."""
    result = (
        monitoring_result
        if isinstance(monitoring_result, MonitoringRunResult)
        else MonitoringRunResult.model_validate(monitoring_result)
    )
    incident = result.incident_candidates[0]
    approval = incident != "normal_operation"
    material = _ids(
        evidence,
        statuses={"critical", "warning", "not_evaluated"},
    )
    if not material:
        material = _ids(
            evidence,
            domains={"system", "performance", "prediction_drift", "feature_drift"},
            statuses={"pass"},
        )
    material = material or [evidence[0]["evidence_id"]]

    if incident == "normal_operation":
        action_type = "continue_monitoring"
        action_text = "Continue scheduled deterministic monitoring."
        claim_text = "The selected deterministic checks support normal operation."
        priority = "low"
    elif incident == "data_quality_failure":
        action_type = "quarantine_batch"
        action_text = "Keep the blocked batch quarantined and investigate the data pipeline."
        claim_text = "Deterministic data-quality checks blocked the monitoring batch."
        priority = "urgent"
    elif incident in {"feature_drift", "prediction_drift", "mixed_incident"}:
        action_type = "investigate_feature_drift"
        action_text = (
            "Investigate the observed drift signals before considering a model change."
        )
        claim_text = "Deterministic monitoring found material drift-related signals."
        priority = "high"
    elif incident == "performance_degradation":
        action_type = "escalate_model_governance"
        action_text = (
            "Escalate the evaluated degradation evidence for model-governance review."
        )
        claim_text = (
            "Deterministic labelled evaluation found material performance degradation."
        )
        priority = "high"
    else:
        action_type = "collect_more_labels"
        action_text = "Collect sufficient labels before drawing a performance conclusion."
        claim_text = "The available evidence is insufficient for a stronger conclusion."
        priority = "medium"

    action = RecommendedAction(
        action_type=action_type,
        action=action_text,
        rationale=(
            "This conservative step is supported by deterministic monitoring evidence "
            "and does not execute remediation."
        ),
        priority=priority,
        evidence_ids=material,
        requires_human_approval=approval,
    )
    return AgentRecommendation(
        incident_type=incident,
        severity=result.overall_severity,
        executive_summary=(
            "The LLM recommendation was unavailable or rejected; this deterministic "
            f"fallback preserves the authoritative {result.display_name} monitoring "
            "outcome."
        ),
        claims=[
            EvidenceBackedClaim(
                claim=claim_text,
                evidence_ids=material,
            )
        ],
        root_cause_hypothesis=None,
        root_cause_evidence_ids=[],
        recommended_actions=[action],
        uncertainties=(
            [
                "Performance conclusions are limited by insufficient label coverage; "
                "the fallback does not replace human review."
            ]
            if incident == "insufficient_evidence"
            else [
                "The fallback does not infer a detailed root cause or replace human review."
            ]
        )
        if approval
        else [],
        overall_evidence_ids=material,
        requires_human_approval=approval,
        confidence=1.0,
    )
