"""Grounding and policy-verifier behavior."""

from monitoring_agent.agent.evidence_selector import build_evidence_packet
from monitoring_agent.agent.fallback import build_fallback_recommendation
from monitoring_agent.agent.schemas import EvidenceBackedClaim, RecommendedAction
from monitoring_agent.agent.verifier import verify_recommendation
from tests.agent.helpers import load_result


def _verify(result, recommendation):
    evidence = build_evidence_packet(result)["evidence"]
    return verify_recommendation(
        recommendation,
        result,
        evidence,
        revision_count=0,
        max_revision_attempts=1,
    )


def test_rejects_hallucinated_evidence_ids() -> None:
    result = load_result("normal_operation")
    evidence = build_evidence_packet(result)["evidence"]
    recommendation = build_fallback_recommendation(result, evidence).model_copy(
        update={
            "claims": [
                EvidenceBackedClaim(
                    claim="Unsupported assertion.",
                    evidence_ids=["DOES-NOT-EXIST"],
                )
            ],
            "overall_evidence_ids": ["DOES-NOT-EXIST"],
        }
    )
    verified = _verify(result, recommendation)
    assert verified.status == "revise"
    assert verified.unsupported_evidence_ids == ["DOES-NOT-EXIST"]


def test_rejects_unsupported_retraining() -> None:
    result = load_result("data_quality_failure")
    evidence = build_evidence_packet(result)["evidence"]
    recommendation = build_fallback_recommendation(result, evidence)
    recommendation = recommendation.model_copy(
        update={
            "recommended_actions": [
                RecommendedAction(
                    action_type="evaluate_retraining",
                    action="Evaluate retraining.",
                    rationale="Unsupported for blocked data.",
                    priority="high",
                    evidence_ids=recommendation.overall_evidence_ids,
                    requires_human_approval=True,
                )
            ]
        }
    )
    assert _verify(result, recommendation).status == "revise"


def test_rejects_performance_claim_on_blocked_batch() -> None:
    result = load_result("data_quality_failure")
    evidence = build_evidence_packet(result)["evidence"]
    recommendation = build_fallback_recommendation(result, evidence).model_copy(
        update={
            "claims": [
                EvidenceBackedClaim(
                    claim="Model performance degradation is established.",
                    evidence_ids=["SYSTEM-BATCH-BLOCKING"],
                )
            ]
        }
    )
    verified = _verify(result, recommendation)
    assert any("performance claim" in item for item in verified.violations)


def test_rejects_non_normal_without_approval() -> None:
    result = load_result("performance_degradation")
    evidence = build_evidence_packet(result)["evidence"]
    recommendation = build_fallback_recommendation(result, evidence)
    actions = [
        action.model_copy(update={"requires_human_approval": False})
        for action in recommendation.recommended_actions
    ]
    recommendation = recommendation.model_copy(
        update={
            "requires_human_approval": False,
            "recommended_actions": actions,
        }
    )
    verified = _verify(result, recommendation)
    assert any("requires human approval" in item for item in verified.violations)


def test_accepts_fully_grounded_recommendation() -> None:
    result = load_result("performance_degradation")
    evidence = build_evidence_packet(result)["evidence"]
    recommendation = build_fallback_recommendation(result, evidence)
    verified = _verify(result, recommendation)
    assert verified.status == "pass"
    assert verified.violations == []
