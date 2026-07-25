"""Hard policy behavior independent of the LLM."""

from monitoring_agent.agent.evidence_selector import build_evidence_packet
from monitoring_agent.agent.fallback import build_fallback_recommendation
from monitoring_agent.agent.policy import evaluate_hard_policy
from monitoring_agent.agent.schemas import EvidenceBackedClaim, RecommendedAction
from tests.agent.helpers import load_result


def _policy(result, recommendation):
    evidence = build_evidence_packet(result)["evidence"]
    return evaluate_hard_policy(
        recommendation,
        result,
        {item["evidence_id"]: item for item in evidence},
    )[0]


def test_blocked_batch_prevents_performance_and_retraining_claims() -> None:
    result = load_result("data_quality_failure")
    evidence = build_evidence_packet(result)["evidence"]
    ids = [item["evidence_id"] for item in evidence[:2]]
    recommendation = build_fallback_recommendation(result, evidence).model_copy(
        update={
            "claims": [
                EvidenceBackedClaim(
                    claim="Model performance degradation is established.",
                    evidence_ids=ids,
                )
            ],
            "recommended_actions": [
                RecommendedAction(
                    action_type="evaluate_retraining",
                    action="Evaluate retraining.",
                    rationale="Claimed performance change.",
                    priority="high",
                    evidence_ids=ids,
                    requires_human_approval=True,
                )
            ],
            "overall_evidence_ids": ids,
        }
    )
    violations = _policy(result, recommendation)
    assert any("performance claim" in item for item in violations)
    assert any("disallowed actions" in item for item in violations)


def test_missing_labels_prevent_performance_claims() -> None:
    result = load_result("normal_operation").model_copy(deep=True)
    result.labels_available = False
    result.performance.evaluated = False
    result.performance.sample_count = 0
    result.performance.reason_not_evaluated = "Labels unavailable."
    result.incident_candidates = ["insufficient_evidence"]
    result.overall_severity = "medium"
    result.evidence[0] = result.evidence[0].model_copy(
        update={
            "domain": "system",
            "status": "not_evaluated",
            "severity": "medium",
        }
    )
    evidence = build_evidence_packet(result)["evidence"]
    evidence_id = evidence[0]["evidence_id"]
    recommendation = build_fallback_recommendation(result, evidence).model_copy(
        update={
            "claims": [
                EvidenceBackedClaim(
                    claim="Model performance degradation is established.",
                    evidence_ids=[evidence_id],
                )
            ]
        }
    )
    violations = _policy(result, recommendation)
    assert any("without evaluated labels" in item for item in violations) is False
    assert any("must not be described as degradation" in item for item in violations)


def test_normal_operation_permits_only_safe_actions() -> None:
    result = load_result("normal_operation")
    evidence = build_evidence_packet(result)["evidence"]
    recommendation = build_fallback_recommendation(result, evidence)
    assert not _policy(result, recommendation)

    unsafe = recommendation.model_copy(
        update={
            "recommended_actions": [
                RecommendedAction(
                    action_type="evaluate_threshold",
                    action="Evaluate a threshold change.",
                    rationale="Not needed for a stable case.",
                    priority="low",
                    evidence_ids=recommendation.overall_evidence_ids,
                    requires_human_approval=False,
                )
            ]
        }
    )
    assert any("unsafe actions" in item for item in _policy(result, unsafe))


def test_feature_drift_alone_is_not_proven_model_failure() -> None:
    result = load_result("normal_operation").model_copy(deep=True)
    result.incident_candidates = ["feature_drift"]
    result.overall_severity = "medium"
    result.evidence[0] = result.evidence[0].model_copy(
        update={
            "domain": "feature_drift",
            "status": "warning",
            "severity": "medium",
        }
    )
    evidence = build_evidence_packet(result)["evidence"]
    recommendation = build_fallback_recommendation(result, evidence)
    assert not _policy(result, recommendation)
    invalid = recommendation.model_copy(
        update={"executive_summary": "The drift confirms model failure."}
    )
    assert any("proven model failure" in item for item in _policy(result, invalid))


def test_retraining_requires_labelled_material_degradation() -> None:
    result = load_result("performance_degradation")
    evidence = build_evidence_packet(result)["evidence"]
    performance_ids = [
        item["evidence_id"]
        for item in evidence
        if item["domain"] == "performance" and item["status"] == "critical"
    ]
    recommendation = build_fallback_recommendation(result, evidence).model_copy(
        update={
            "recommended_actions": [
                RecommendedAction(
                    action_type="evaluate_retraining",
                    action="Evaluate retraining in governed offline validation.",
                    rationale="Material labelled degradation was measured.",
                    priority="high",
                    evidence_ids=performance_ids,
                    requires_human_approval=True,
                )
            ],
            "overall_evidence_ids": list(
                dict.fromkeys(
                    [
                        *build_fallback_recommendation(
                            result, evidence
                        ).overall_evidence_ids,
                        *performance_ids,
                    ]
                )
            ),
        }
    )
    assert not _policy(result, recommendation)

    ineligible = result.model_copy(deep=True)
    ineligible.performance.sample_count = 100
    assert any("requires evaluated" in item for item in _policy(ineligible, recommendation))


def test_unlabelled_metric_degradation_claim_is_rejected() -> None:
    result = load_result("normal_operation").model_copy(deep=True)
    result.labels_available = False
    result.performance.evaluated = False
    result.performance.sample_count = 0
    result.incident_candidates = ["feature_drift", "insufficient_evidence"]
    result.overall_severity = "medium"
    result.evidence[0] = result.evidence[0].model_copy(
        update={
            "domain": "feature_drift",
            "status": "warning",
            "severity": "medium",
        }
    )
    evidence = build_evidence_packet(result)["evidence"]
    recommendation = build_fallback_recommendation(result, evidence).model_copy(
        update={
            "claims": [
                EvidenceBackedClaim(
                    claim="Recall and ROC-AUC degraded in this unlabelled batch.",
                    evidence_ids=[evidence[0]["evidence_id"]],
                )
            ]
        }
    )
    assert any(
        "metric deterioration" in item for item in _policy(result, recommendation)
    )


def test_insufficient_labels_require_collection_and_explicit_uncertainty() -> None:
    result = load_result("normal_operation").model_copy(deep=True)
    result.scenario_name = "insufficient_labels"
    result.labels_available = True
    result.labelled_row_count = 100
    result.label_coverage_rate = 0.1
    result.minimum_labelled_sample_size = 200
    result.performance.evaluated = False
    result.performance.sample_count = 100
    result.performance.reason_not_evaluated = "Only 100 labels; at least 200 required."
    result.incident_candidates = ["insufficient_evidence"]
    result.overall_severity = "medium"
    result.evidence[0] = result.evidence[0].model_copy(
        update={
            "domain": "system",
            "status": "not_evaluated",
            "severity": "medium",
        }
    )
    evidence = build_evidence_packet(result)["evidence"]
    valid = build_fallback_recommendation(result, evidence)
    assert not _policy(result, valid)

    invalid = valid.model_copy(
        update={
            "recommended_actions": [
                RecommendedAction(
                    action_type="continue_monitoring",
                    action="Continue monitoring.",
                    rationale="Wait for more evidence.",
                    priority="medium",
                    evidence_ids=valid.overall_evidence_ids,
                    requires_human_approval=True,
                )
            ],
            "uncertainties": ["No causal conclusion is made."],
        }
    )
    violations = _policy(result, invalid)
    assert any("primary action" in item for item in violations)
    assert any("evidence-sufficiency uncertainty" in item for item in violations)
