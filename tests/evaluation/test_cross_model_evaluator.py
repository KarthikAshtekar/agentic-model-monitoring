"""Cross-model aggregation remains domain-safe and revision-aware."""

from __future__ import annotations

from monitoring_agent.evaluation.schemas import (
    LiveEvaluationSummary,
    ScenarioEvaluation,
)
from scripts.evaluate_cross_model_agent import build_cross_model_summary


def _summary(
    model_id: str,
    domain_id: str,
    revisions: list[int],
) -> LiveEvaluationSummary:
    results = [
        ScenarioEvaluation.model_construct(
            model_id=model_id,
            revision_count=revision_count,
            latency_ms=10.0,
        )
        for revision_count in revisions
    ]
    return LiveEvaluationSummary.model_construct(
        provider="groq",
        model="openai/gpt-oss-20b",
        model_id=model_id,
        domain_id=domain_id,
        scenario_count=len(results),
        completed_count=len(results),
        structured_output_success_rate=1.0,
        incident_compatibility_rate=1.0,
        route_compatibility_rate=1.0,
        evidence_grounding_rate=1.0,
        policy_compliance_rate=1.0,
        first_pass_verification_rate=0.75,
        fallback_rate=0.0,
        approval_completion_rate=1.0,
        mean_latency_ms=10.0,
        median_latency_ms=10.0,
        total_input_tokens=100,
        total_output_tokens=50,
        total_tokens=150,
        scenario_results=results,
    )


def test_cross_model_revision_rate_is_calculated_from_scenarios() -> None:
    credit = _summary("credit_default", "credit_risk", [0, 1])
    diabetes = _summary("diabetes_risk", "diabetes_screening", [0, 0])

    payload = build_cross_model_summary([credit, diabetes])

    assert payload["revision_rate"] == 0.25
    assert payload["models"][0]["revision_rate"] == 0.5
    assert payload["models"][1]["revision_rate"] == 0.0
    assert "roc_auc" not in payload
