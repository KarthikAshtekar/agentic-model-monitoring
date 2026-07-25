"""Deterministic incident and action policy for agent recommendations."""

from __future__ import annotations

from typing import Any

from monitoring_agent.agent.schemas import AgentRecommendation, AgentTriage
from monitoring_agent.monitoring.schemas import MonitoringRunResult

MINIMUM_LABELLED_SAMPLES = 200
BLOCKED_ALLOWED_ACTIONS = {
    "quarantine_batch",
    "investigate_data_pipeline",
    "escalate_model_governance",
}
NORMAL_ALLOWED_ACTIONS = {"no_action", "continue_monitoring"}
UNEVALUATED_PERFORMANCE_FORBIDDEN_ACTIONS = {
    "evaluate_threshold",
    "evaluate_recalibration",
    "evaluate_retraining",
}


def build_policy_context(
    result: MonitoringRunResult | dict[str, Any],
) -> dict[str, Any]:
    """Expose only concise deterministic constraints to the structured LLM."""
    monitoring_result = (
        result
        if isinstance(result, MonitoringRunResult)
        else MonitoringRunResult.model_validate(result)
    )
    material_performance_ids = [
        item.evidence_id
        for item in monitoring_result.evidence
        if item.domain == "performance" and item.status == "critical"
    ]
    return {
        "incident_candidates": monitoring_result.incident_candidates,
        "deterministic_overall_severity": monitoring_result.overall_severity,
        "batch_blocked": monitoring_result.batch_blocked,
        "labels_available": monitoring_result.labels_available,
        "performance_evaluated": monitoring_result.performance.evaluated,
        "labelled_sample_size": monitoring_result.performance.sample_count,
        "feature_row_count": monitoring_result.feature_row_count,
        "labelled_row_count": monitoring_result.labelled_row_count,
        "label_coverage_rate": monitoring_result.label_coverage_rate,
        "minimum_labelled_samples": (
            monitoring_result.minimum_labelled_sample_size
            or MINIMUM_LABELLED_SAMPLES
        ),
        "performance_reason_not_evaluated": (
            monitoring_result.performance.reason_not_evaluated
        ),
        "material_performance_evidence_ids": material_performance_ids,
        "normal_allowed_actions": sorted(NORMAL_ALLOWED_ACTIONS),
        "blocked_allowed_actions": sorted(BLOCKED_ALLOWED_ACTIONS),
        "all_non_normal_incidents_require_human_approval": True,
        "actions_are_recommendations_only": True,
    }


def deterministic_triage(
    result: MonitoringRunResult | dict[str, Any],
    evidence: list[dict[str, Any]],
) -> AgentTriage:
    """Provide a safe route when structured triage is unavailable or invalid."""
    monitoring_result = (
        result
        if isinstance(result, MonitoringRunResult)
        else MonitoringRunResult.model_validate(result)
    )
    incident = monitoring_result.incident_candidates[0]
    routes = {
        "normal_operation": "no_additional_diagnostics",
        "data_quality_failure": "data_quality_diagnostics",
        "feature_drift": "drift_diagnostics",
        "prediction_drift": "drift_diagnostics",
        "performance_degradation": "performance_diagnostics",
        "mixed_incident": "mixed_diagnostics",
        "insufficient_evidence": "evidence_sufficiency_review",
    }
    material_ids = [
        item["evidence_id"]
        for item in evidence
        if item["status"] in {"critical", "warning", "not_evaluated"}
    ]
    if not material_ids:
        material_ids = [
            item["evidence_id"]
            for item in evidence
            if item["evidence_id"].startswith("SYSTEM-") or item["status"] == "pass"
        ][:8]
    return AgentTriage(
        incident_type=incident,
        severity=monitoring_result.overall_severity,
        diagnostic_route=routes[incident],
        needs_additional_diagnostics=incident != "normal_operation",
        selected_evidence_ids=material_ids[:12],
        reason=(
            "Deterministic fallback triage selected the leading monitoring candidate "
            "and its compatible diagnostic route."
        ),
    )


def triage_violations(
    triage: AgentTriage,
    result: MonitoringRunResult,
    valid_evidence_ids: set[str],
) -> list[str]:
    """Check structured triage before it controls diagnostic routing."""
    violations: list[str] = []
    unsupported = set(triage.selected_evidence_ids) - valid_evidence_ids
    if unsupported:
        violations.append(f"Triage cited unsupported evidence IDs: {sorted(unsupported)}.")
    if triage.incident_type not in result.incident_candidates:
        violations.append(
            "Triage incident type is incompatible with deterministic candidates."
        )
    if result.batch_blocked and triage.incident_type != "data_quality_failure":
        violations.append("A blocked batch must route as data_quality_failure.")
    if result.incident_candidates == ["normal_operation"]:
        if triage.diagnostic_route != "no_additional_diagnostics":
            violations.append("Normal operation must not request an incident diagnostic route.")
    return violations


def _contains_any(text: str, phrases: set[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _claims_metric_degradation_without_evaluation(text: str) -> bool:
    lowered = text.lower()
    metric_terms = {"recall", "precision", "pr-auc", "pr_auc", "roc-auc", "roc_auc"}
    degradation_terms = {
        "degrad",
        "declin",
        "drop",
        "deteriorat",
        "worsen",
        "lower",
    }
    return any(metric in lowered for metric in metric_terms) and any(
        term in lowered for term in degradation_terms
    )


def evaluate_hard_policy(
    recommendation: AgentRecommendation,
    result: MonitoringRunResult,
    evidence_by_id: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return policy violations plus audit-friendly checks performed."""
    violations: list[str] = []
    checks = [
        "incident_candidate_compatibility",
        "batch_blocking_policy",
        "label_and_performance_availability_policy",
        "normal_operation_action_policy",
        "retraining_eligibility_policy",
        "feature_drift_interpretation_policy",
        "human_approval_policy",
        "no_automatic_action_policy",
    ]
    action_types = {action.action_type for action in recommendation.recommended_actions}
    claim_text = " ".join(claim.claim for claim in recommendation.claims)

    if recommendation.incident_type not in result.incident_candidates:
        violations.append(
            "Recommendation incident type is not a deterministic incident candidate."
        )

    if result.batch_blocked:
        if recommendation.incident_type != "data_quality_failure":
            violations.append("A blocked batch must be classified as data_quality_failure.")
        disallowed = action_types - BLOCKED_ALLOWED_ACTIONS
        if disallowed:
            violations.append(
                f"Blocked-batch recommendation contains disallowed actions: {sorted(disallowed)}."
            )
        cited_domains = {
            evidence_by_id[evidence_id]["domain"]
            for claim in recommendation.claims
            for evidence_id in claim.evidence_ids
            if evidence_id in evidence_by_id
        }
        if "performance" in cited_domains or _contains_any(
            claim_text,
            {"performance degradation", "model performance degraded", "concept drift"},
        ):
            violations.append("A blocked batch must not make a performance claim.")

    performance_unavailable = (
        not result.labels_available or not result.performance.evaluated
    )
    if performance_unavailable:
        if recommendation.incident_type in {
            "performance_degradation",
            "mixed_incident",
        }:
            violations.append(
                "Performance degradation cannot be concluded without evaluated labels."
            )
        forbidden = action_types & UNEVALUATED_PERFORMANCE_FORBIDDEN_ACTIONS
        if forbidden:
            violations.append(
                "Unevaluated performance cannot support threshold, calibration, or "
                f"retraining actions: {sorted(forbidden)}."
            )
        if _contains_any(
            claim_text,
            {"performance degradation", "model performance degraded", "concept drift"},
        ) or _claims_metric_degradation_without_evaluation(claim_text):
            violations.append(
                "Unevaluated performance must not be described as degradation, metric "
                "deterioration, or concept drift."
            )

    insufficient_label_evidence = (
        result.labels_available
        and not result.performance.evaluated
        and result.performance.sample_count
        < (
            result.minimum_labelled_sample_size
            or MINIMUM_LABELLED_SAMPLES
        )
        and "insufficient_evidence" in result.incident_candidates
    )
    if insufficient_label_evidence:
        if not recommendation.recommended_actions or (
            recommendation.recommended_actions[0].action_type != "collect_more_labels"
        ):
            violations.append(
                "Insufficient labels require collect_more_labels as the primary action."
            )
        uncertainty_text = " ".join(recommendation.uncertainties).lower()
        if not any(
            term in uncertainty_text
            for term in {"label", "coverage", "sample", "insufficient", "evidence"}
        ):
            violations.append(
                "Insufficient labels require an explicit evidence-sufficiency uncertainty."
            )

    if result.incident_candidates == ["normal_operation"]:
        if recommendation.incident_type != "normal_operation":
            violations.append("A stable deterministic case must remain normal_operation.")
        if recommendation.severity not in {"info", "low"}:
            violations.append("Normal-operation severity must be info or low.")
        disallowed = action_types - NORMAL_ALLOWED_ACTIONS
        if disallowed:
            violations.append(
                f"Normal operation contains unsafe actions: {sorted(disallowed)}."
            )
        if recommendation.requires_human_approval:
            violations.append("Normal operation must not require human approval.")
        if any(
            action.requires_human_approval
            for action in recommendation.recommended_actions
        ):
            violations.append("Normal-operation actions must not require approval.")

    if "evaluate_retraining" in action_types:
        material_ids = {
            item.evidence_id
            for item in result.evidence
            if item.domain == "performance" and item.status == "critical"
        }
        retraining_actions = [
            action
            for action in recommendation.recommended_actions
            if action.action_type == "evaluate_retraining"
        ]
        eligibility = (
            result.performance.evaluated
            and result.performance.sample_count >= MINIMUM_LABELLED_SAMPLES
            and bool(material_ids)
        )
        if not eligibility:
            violations.append(
                "evaluate_retraining requires evaluated, sufficiently sampled material "
                "performance degradation."
            )
        if any(not (set(action.evidence_ids) & material_ids) for action in retraining_actions):
            violations.append(
                "evaluate_retraining must cite critical performance evidence."
            )

    material_performance = any(
        item.domain == "performance" and item.status == "critical"
        for item in result.evidence
    )
    if recommendation.incident_type in {"feature_drift", "prediction_drift"} and not (
        material_performance
    ):
        combined_text = " ".join(
            [
                recommendation.executive_summary,
                claim_text,
                *[action.rationale for action in recommendation.recommended_actions],
            ]
        )
        if _contains_any(
            combined_text,
            {"proven model failure", "model has failed", "confirms model failure"},
        ):
            violations.append("Feature drift alone must not be stated as proven model failure.")

    non_normal = recommendation.incident_type != "normal_operation"
    if non_normal:
        if not recommendation.requires_human_approval:
            violations.append("Every non-normal recommendation requires human approval.")
        if any(
            not action.requires_human_approval
            for action in recommendation.recommended_actions
        ):
            violations.append("Every non-normal recommended action requires approval.")

    automatic_phrases = {
        "automatically retrain",
        "automatic retraining",
        "automatically deploy",
        "automatically rollback",
        "execute immediately without approval",
    }
    if any(
        _contains_any(f"{action.action} {action.rationale}", automatic_phrases)
        for action in recommendation.recommended_actions
    ):
        violations.append("The MVP cannot recommend automatic execution.")

    return violations, checks
