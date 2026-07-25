"""Regression tests for deterministic live-agent evaluation."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from monitoring_agent.evaluation.evaluator import (
    evaluate_live_runs,
    evaluate_scenario,
    scenario_expectations,
)
from monitoring_agent.monitoring.schemas import MonitoringRunResult
from tests.agent.helpers import load_result


def _fixture(
    result: MonitoringRunResult,
    *,
    token_usage: dict[str, int] | None = None,
) -> dict:
    expectations = scenario_expectations(result.scenario_name)
    incident = expectations["incident_types"][0]
    route = expectations["routes"][0]
    material = [
        item.evidence_id
        for item in result.evidence
        if item.status in {"critical", "warning"}
    ]
    evidence_ids = material[:6] or [result.evidence[0].evidence_id]
    non_normal = incident != "normal_operation"
    if incident == "normal_operation":
        action_type = "continue_monitoring"
        action = "Continue deterministic monitoring."
        claim = "The deterministic evidence supports normal operation."
    elif incident == "data_quality_failure":
        action_type = "quarantine_batch"
        action = "Keep the blocked batch quarantined for pipeline review."
        claim = "Critical data-quality evidence blocked this batch."
    elif incident == "performance_degradation":
        action_type = "escalate_model_governance"
        action = "Escalate the degradation evidence for governed review."
        claim = "Material labelled performance degradation was measured."
    elif incident == "insufficient_evidence":
        action_type = "collect_more_labels"
        action = "Collect enough aligned labels to meet the policy minimum."
        claim = "Available labels are insufficient for performance evaluation."
    else:
        action_type = "investigate_feature_drift"
        action = "Investigate the material drift signals."
        claim = "Material drift signals were measured."
    reviewed_at = datetime.now(UTC).isoformat()
    return {
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "execution_mode": "live",
        "is_fake_llm": False,
        "run_id": result.run_id,
        "thread_id": f"fixture-{result.run_id.lower()}",
        "created_at_utc": reviewed_at,
        "scenario_name": result.scenario_name,
        "deterministic_incident_candidates": result.incident_candidates,
        "selected_evidence": {"case_summary": {}, "evidence": []},
        "triage": {
            "incident_type": incident,
            "severity": result.overall_severity,
            "diagnostic_route": route,
            "needs_additional_diagnostics": non_normal,
            "selected_evidence_ids": evidence_ids,
            "reason": "Minimal structured fixture.",
        },
        "diagnostic_context": {"route": route, "evidence": []},
        "recommendation": {
            "incident_type": incident,
            "severity": result.overall_severity,
            "executive_summary": "Minimal grounded live evaluation fixture.",
            "claims": [{"claim": claim, "evidence_ids": evidence_ids}],
            "root_cause_hypothesis": (
                "Hypothesis: The replay transformation may contribute to the signals."
                if non_normal
                else None
            ),
            "root_cause_evidence_ids": evidence_ids if non_normal else [],
            "recommended_actions": [
                {
                    "action_type": action_type,
                    "action": action,
                    "rationale": "The action is proposed for human review only.",
                    "priority": "high" if non_normal else "low",
                    "evidence_ids": evidence_ids,
                    "requires_human_approval": non_normal,
                }
            ],
            "uncertainties": (
                [
                    (
                        "Conclusions are limited by insufficient label coverage."
                        if incident == "insufficient_evidence"
                        else "Controlled replay evidence does not establish causality."
                    )
                ]
                if non_normal
                else []
            ),
            "overall_evidence_ids": evidence_ids,
            "requires_human_approval": non_normal,
            "confidence": 0.8,
        },
        "verification": {
            "status": "pass",
            "violations": [],
            "unsupported_evidence_ids": [],
            "policy_checks": ["fixture"],
            "revision_feedback": None,
        },
        "revision_count": 0,
        "fallback_used": False,
        "approval_required": non_normal,
        "approval_decision": (
            {
                "decision": "approve",
                "comment": None,
                "reviewer": "fixture-reviewer",
                "reviewed_at_utc": reviewed_at,
            }
            if non_normal
            else None
        ),
        "llm_call_metadata": [
            {
                "stage": "triage",
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "schema": "AgentTriage",
                "latency_ms": 10.0,
                "token_usage": token_usage,
                "parse_success": True,
                "is_fake": False,
            },
            {
                "stage": "recommendation",
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "schema": "AgentRecommendation",
                "latency_ms": 20.0,
                "token_usage": token_usage,
                "parse_success": True,
                "is_fake": False,
            },
        ],
        "execution_errors": [],
        "final_status": "approved" if non_normal else "completed_no_approval_required",
    }


def test_scenario_expectations_are_controlled() -> None:
    assert scenario_expectations("normal_operation") == {
        "incident_types": ["normal_operation"],
        "routes": ["no_additional_diagnostics"],
    }
    assert scenario_expectations("feature_drift")["incident_types"] == [
        "mixed_incident",
        "feature_drift",
    ]
    assert scenario_expectations("unlabelled_drift")["routes"] == [
        "drift_diagnostics",
        "mixed_diagnostics",
        "evidence_sufficiency_review",
    ]
    assert scenario_expectations("insufficient_labels") == {
        "incident_types": ["insufficient_evidence"],
        "routes": ["evidence_sufficiency_review"],
    }
    with pytest.raises(ValueError):
        scenario_expectations("unknown")


def test_grounded_result_passes() -> None:
    result = load_result("performance_degradation")
    evaluated = evaluate_scenario(_fixture(result), result)
    assert evaluated.all_cited_evidence_valid is True
    assert evaluated.policy_compliant is True
    assert evaluated.violations == []


def test_hallucinated_evidence_id_fails() -> None:
    result = load_result("performance_degradation")
    agent = _fixture(result)
    agent["recommendation"]["claims"][0]["evidence_ids"] = ["HALLUCINATED-ID"]
    evaluated = evaluate_scenario(agent, result)
    assert evaluated.all_cited_evidence_valid is False
    assert evaluated.all_claims_cited is False


def test_unsafe_data_quality_recommendation_fails_policy() -> None:
    result = load_result("data_quality_failure")
    agent = _fixture(result)
    agent["recommendation"]["recommended_actions"][0][
        "action_type"
    ] = "evaluate_retraining"
    evaluated = evaluate_scenario(agent, result)
    assert evaluated.policy_compliant is False


def test_normal_operation_requiring_approval_fails() -> None:
    result = load_result("normal_operation")
    agent = _fixture(result)
    agent["approval_required"] = True
    agent["recommendation"]["requires_human_approval"] = True
    agent["recommendation"]["recommended_actions"][0][
        "requires_human_approval"
    ] = True
    agent["approval_decision"] = {
        "decision": "approve",
        "comment": None,
        "reviewer": "fixture-reviewer",
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
    }
    agent["final_status"] = "approved"
    evaluated = evaluate_scenario(agent, result)
    assert evaluated.policy_compliant is False
    assert evaluated.approval_completed is False


def test_fallback_is_excluded_from_structured_output_success() -> None:
    result = load_result("performance_degradation")
    agent = _fixture(result)
    agent["fallback_used"] = True
    evaluated = evaluate_scenario(agent, result)
    assert evaluated.triage_parse_success is True
    assert evaluated.recommendation_parse_success is False


def test_aggregate_rates_are_calculated() -> None:
    runs = []
    for scenario in (
        "normal_operation",
        "feature_drift",
        "data_quality_failure",
        "performance_degradation",
    ):
        result = load_result(scenario)
        runs.append((_fixture(result), result))
    degraded = deepcopy(runs[-1][0])
    degraded["fallback_used"] = True
    runs[-1] = (degraded, runs[-1][1])
    summary = evaluate_live_runs(runs)
    assert summary.scenario_count == 4
    assert summary.incident_compatibility_rate == 1.0
    assert summary.structured_output_success_rate == 0.75
    assert summary.fallback_rate == 0.25


def test_missing_token_metadata_remains_nullable() -> None:
    result = load_result("normal_operation")
    evaluated = evaluate_scenario(_fixture(result, token_usage=None), result)
    assert evaluated.input_tokens is None
    assert evaluated.output_tokens is None
    assert evaluated.total_tokens is None


def test_safe_live_execution_error_is_preserved_in_notes() -> None:
    result = load_result("feature_drift")
    agent = _fixture(result)
    agent["execution_errors"] = [
        {
            "stage": "recommendation",
            "error_type": "RateLimitError",
            "error_message": "Groq request was rate-limited (HTTP 429).",
        }
    ]

    evaluated = evaluate_scenario(agent, result)

    assert any(
        "recommendation: RateLimitError" in note for note in evaluated.notes
    )


def test_insufficient_labels_fixture_requires_collection_action() -> None:
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
    evaluated = evaluate_scenario(_fixture(result), result)
    assert evaluated.incident_compatible is True
    assert evaluated.route_compatible is True
    assert evaluated.policy_compliant is True
