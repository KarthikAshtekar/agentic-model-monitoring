"""Deterministic grounding and policy verification for LLM recommendations."""

from __future__ import annotations

from typing import Any

from monitoring_agent.agent.policy import evaluate_hard_policy
from monitoring_agent.agent.schemas import AgentRecommendation, VerificationResult
from monitoring_agent.monitoring.evidence import SEVERITY_RANK
from monitoring_agent.monitoring.schemas import MonitoringRunResult


def verify_recommendation(
    recommendation: AgentRecommendation,
    monitoring_result: MonitoringRunResult | dict[str, Any],
    available_evidence: list[dict[str, Any]],
    *,
    revision_count: int,
    max_revision_attempts: int,
) -> VerificationResult:
    """Verify citations, severity, uncertainty, and every hard policy rule."""
    result = (
        monitoring_result
        if isinstance(monitoring_result, MonitoringRunResult)
        else MonitoringRunResult.model_validate(monitoring_result)
    )
    evidence_by_id = {item["evidence_id"]: item for item in available_evidence}
    valid_ids = set(evidence_by_id)
    cited_ids = {
        evidence_id
        for claim in recommendation.claims
        for evidence_id in claim.evidence_ids
    }
    cited_ids.update(
        evidence_id
        for action in recommendation.recommended_actions
        for evidence_id in action.evidence_ids
    )
    cited_ids.update(recommendation.root_cause_evidence_ids)
    unsupported = sorted(cited_ids - valid_ids)
    violations: list[str] = []
    checks = [
        "citation_existence_and_uniqueness",
        "claim_and_action_grounding",
        "overall_evidence_coverage",
        "severity_compatibility",
        "confidence_schema_bounds",
        "uncertainty_presence",
    ]

    if unsupported:
        violations.append(f"Unsupported evidence IDs were cited: {unsupported}.")
    if any(not claim.evidence_ids for claim in recommendation.claims):
        violations.append("Every claim must cite at least one evidence ID.")
    if any(not action.evidence_ids for action in recommendation.recommended_actions):
        violations.append("Every action must cite at least one evidence ID.")
    missing_from_overall = cited_ids - set(recommendation.overall_evidence_ids)
    if missing_from_overall:
        violations.append(
            "overall_evidence_ids does not contain all supporting citations: "
            f"{sorted(missing_from_overall)}."
        )
    unsupported_overall = set(recommendation.overall_evidence_ids) - valid_ids
    if unsupported_overall:
        unsupported = sorted(set(unsupported) | unsupported_overall)
        violations.append(
            f"overall_evidence_ids contains unsupported IDs: {sorted(unsupported_overall)}."
        )

    if result.incident_candidates == ["normal_operation"]:
        if recommendation.severity not in {"info", "low"}:
            violations.append("Normal-operation severity is incompatible with evidence.")
    elif (
        SEVERITY_RANK[recommendation.severity]
        < SEVERITY_RANK[result.overall_severity]
    ):
        violations.append(
            "Recommendation severity is lower than deterministic overall severity."
        )

    if not 0.0 <= recommendation.confidence <= 1.0:
        violations.append("Confidence must be between zero and one.")
    if (
        recommendation.incident_type != "normal_operation"
        and not recommendation.uncertainties
    ):
        violations.append(
            "A replay or inferential non-normal conclusion must state an uncertainty."
        )
    if (
        recommendation.root_cause_hypothesis is not None
        and not recommendation.root_cause_hypothesis.strip()
        .lower()
        .startswith("hypothesis:")
    ):
        violations.append(
            "root_cause_hypothesis must be explicitly labelled with 'Hypothesis:'."
        )
    if (
        recommendation.root_cause_hypothesis is None
        and recommendation.root_cause_evidence_ids
    ):
        violations.append(
            "A null root-cause hypothesis must not contain root-cause evidence IDs."
        )
    if (
        recommendation.root_cause_hypothesis is not None
        and not recommendation.root_cause_evidence_ids
    ):
        violations.append("A root-cause hypothesis must cite evidence.")

    policy_violations, policy_checks = evaluate_hard_policy(
        recommendation,
        result,
        evidence_by_id,
    )
    violations.extend(policy_violations)
    checks.extend(policy_checks)

    if violations:
        status = (
            "revise"
            if revision_count < max_revision_attempts
            else "fallback"
        )
        feedback = " ".join(dict.fromkeys(violations))
    else:
        status = "pass"
        feedback = None
    return VerificationResult(
        status=status,
        violations=list(dict.fromkeys(violations)),
        unsupported_evidence_ids=unsupported,
        policy_checks=list(dict.fromkeys(checks)),
        revision_feedback=feedback,
    )
