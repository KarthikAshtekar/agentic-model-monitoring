"""Deterministic fallback safety constraints."""

from monitoring_agent.agent.evidence_selector import build_evidence_packet
from monitoring_agent.agent.fallback import build_fallback_recommendation
from tests.agent.helpers import load_result


def test_blocked_batch_fallback_is_safe_and_requires_approval() -> None:
    result = load_result("data_quality_failure")
    evidence = build_evidence_packet(result)["evidence"]
    recommendation = build_fallback_recommendation(result, evidence)
    assert recommendation.incident_type == "data_quality_failure"
    assert recommendation.requires_human_approval is True
    assert {action.action_type for action in recommendation.recommended_actions} <= {
        "quarantine_batch",
        "investigate_data_pipeline",
        "escalate_model_governance",
    }
    assert all(action.evidence_ids for action in recommendation.recommended_actions)


def test_normal_fallback_needs_no_approval() -> None:
    result = load_result("normal_operation")
    evidence = build_evidence_packet(result)["evidence"]
    recommendation = build_fallback_recommendation(result, evidence)
    assert recommendation.incident_type == "normal_operation"
    assert recommendation.requires_human_approval is False
    assert {action.action_type for action in recommendation.recommended_actions} <= {
        "no_action",
        "continue_monitoring",
    }
