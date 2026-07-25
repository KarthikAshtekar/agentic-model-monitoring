"""Mostly deterministic evaluation of live structured-agent outputs."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from monitoring_agent.agent.policy import evaluate_hard_policy
from monitoring_agent.agent.schemas import AgentRecommendation
from monitoring_agent.evaluation.schemas import LiveEvaluationSummary, ScenarioEvaluation
from monitoring_agent.monitoring.schemas import MonitoringRunResult

CORE_SCENARIOS = (
    "normal_operation",
    "feature_drift",
    "data_quality_failure",
    "performance_degradation",
)
EXTENDED_SCENARIOS = (
    *CORE_SCENARIOS,
    "unlabelled_drift",
    "insufficient_labels",
)
REQUIRED_SCENARIOS = CORE_SCENARIOS

_EXPECTATIONS: dict[str, dict[str, list[str]]] = {
    "normal_operation": {
        "incident_types": ["normal_operation"],
        "routes": ["no_additional_diagnostics"],
    },
    "feature_drift": {
        "incident_types": ["mixed_incident", "feature_drift"],
        "routes": ["mixed_diagnostics", "drift_diagnostics"],
    },
    "data_quality_failure": {
        "incident_types": ["data_quality_failure"],
        "routes": ["data_quality_diagnostics"],
    },
    "performance_degradation": {
        "incident_types": ["performance_degradation"],
        "routes": ["performance_diagnostics"],
    },
    "unlabelled_drift": {
        "incident_types": [
            "feature_drift",
            "prediction_drift",
            "mixed_incident",
            "insufficient_evidence",
        ],
        "routes": [
            "drift_diagnostics",
            "mixed_diagnostics",
            "evidence_sufficiency_review",
        ],
    },
    "insufficient_labels": {
        "incident_types": ["insufficient_evidence"],
        "routes": ["evidence_sufficiency_review"],
    },
}

_COMPLETED_STATUSES = {
    "approved",
    "rejected",
    "revision_requested",
    "completed_no_approval_required",
    "fallback_pending_review",
}


def scenario_expectations(scenario_name: str) -> dict[str, list[str]]:
    """Return a defensive copy of the controlled evaluation expectations."""
    if scenario_name not in _EXPECTATIONS:
        raise ValueError(f"Unsupported evaluation scenario: {scenario_name}")
    return {
        key: list(values) for key, values in _EXPECTATIONS[scenario_name].items()
    }


def _provider_calls(
    agentic_result: dict[str, Any],
    schema: str | None = None,
) -> list[dict[str, Any]]:
    calls = [
        item
        for item in agentic_result.get("llm_call_metadata", [])
        if item.get("provider") == "groq"
        and item.get("schema") in {"AgentTriage", "AgentRecommendation"}
    ]
    if schema is not None:
        calls = [item for item in calls if item.get("schema") == schema]
    return calls


def _token_total(
    calls: list[dict[str, Any]],
    key: str,
) -> int | None:
    values = [
        usage[key]
        for item in calls
        if isinstance((usage := item.get("token_usage")), dict)
        and isinstance(usage.get(key), int)
    ]
    return sum(values) if values else None


def _all_citations(
    triage: dict[str, Any],
    recommendation: dict[str, Any],
) -> set[str]:
    citations = set(triage.get("selected_evidence_ids", []))
    citations.update(recommendation.get("overall_evidence_ids", []))
    citations.update(recommendation.get("root_cause_evidence_ids", []))
    for claim in recommendation.get("claims", []):
        citations.update(claim.get("evidence_ids", []))
    for action in recommendation.get("recommended_actions", []):
        citations.update(action.get("evidence_ids", []))
    return citations


def _contains_any(text: str, phrases: set[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _scenario_policy_violations(
    scenario_name: str,
    agentic_result: dict[str, Any],
    monitoring_result: MonitoringRunResult,
    recommendation: AgentRecommendation,
    evidence_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    violations, _ = evaluate_hard_policy(
        recommendation,
        monitoring_result,
        evidence_by_id,
    )
    action_types = {
        action.action_type for action in recommendation.recommended_actions
    }
    cited_ids = _all_citations(
        agentic_result.get("triage", {}),
        agentic_result.get("recommendation", {}),
    )
    cited_domains = {
        evidence_by_id[item]["domain"] for item in cited_ids if item in evidence_by_id
    }
    approval = agentic_result.get("approval_decision")
    final_status = agentic_result.get("final_status")

    if agentic_result.get("provider") != "groq":
        violations.append("Live evaluation report did not use the Groq provider.")
    if agentic_result.get("execution_mode") != "live":
        violations.append("Live evaluation report is not labelled execution_mode=live.")
    if agentic_result.get("is_fake_llm") is not False:
        violations.append("Fake LLM use is not permitted in live evaluation.")

    if scenario_name == "normal_operation":
        if action_types - {"no_action", "continue_monitoring"}:
            violations.append("Normal operation contains a non-monitoring action.")
        if agentic_result.get("approval_required"):
            violations.append("Normal operation incorrectly required approval.")
        if approval is not None:
            violations.append("Normal operation unexpectedly recorded an approval decision.")
        if final_status != "completed_no_approval_required":
            violations.append("Normal operation has an incompatible final status.")

    if scenario_name == "feature_drift":
        if not cited_domains & {"feature_drift", "prediction_drift"}:
            violations.append("Feature-drift result did not cite drift evidence.")
        combined = " ".join(
            [
                recommendation.executive_summary,
                *[claim.claim for claim in recommendation.claims],
            ]
        )
        if _contains_any(
            combined,
            {"proven model failure", "model has failed", "confirms model failure"},
        ):
            violations.append("Feature drift was stated as proven model failure.")

    if scenario_name == "data_quality_failure":
        blocking_ids = {
            item.evidence_id
            for item in monitoring_result.evidence
            if item.domain == "data_quality" and item.status == "critical"
        }
        if not monitoring_result.batch_blocked:
            violations.append("Deterministic batch blocking was not preserved.")
        if not cited_ids & blocking_ids:
            violations.append("Data-quality recommendation did not cite blocking evidence.")
        if action_types & {
            "evaluate_threshold",
            "evaluate_recalibration",
            "evaluate_retraining",
        }:
            violations.append("Blocked data-quality batch contains a model-change action.")
        if not action_types & {"quarantine_batch", "investigate_data_pipeline"}:
            violations.append("Data-quality actions did not prioritize quarantine or pipeline review.")

    if scenario_name == "performance_degradation":
        material_ids = {
            item.evidence_id
            for item in monitoring_result.evidence
            if item.domain == "performance" and item.status == "critical"
        }
        if not cited_ids & material_ids:
            violations.append("Performance degradation did not cite material evidence.")
        if not recommendation.uncertainties:
            violations.append("Synthetic performance degradation did not state uncertainty.")
        hypothesis = recommendation.root_cause_hypothesis or ""
        if hypothesis and not hypothesis.lower().startswith("hypothesis:"):
            violations.append("Root cause was not explicitly labelled as a hypothesis.")

    if scenario_name == "unlabelled_drift":
        if not cited_domains & {"feature_drift", "prediction_drift"}:
            violations.append("Unlabelled drift did not cite authoritative drift evidence.")
        if action_types & {
            "evaluate_threshold",
            "evaluate_recalibration",
            "evaluate_retraining",
        }:
            violations.append(
                "Unlabelled drift contains an unsupported immediate model-change action."
            )

    if scenario_name == "insufficient_labels":
        if not recommendation.recommended_actions or (
            recommendation.recommended_actions[0].action_type != "collect_more_labels"
        ):
            violations.append(
                "Insufficient labels did not prioritize collect_more_labels."
            )
        uncertainty_text = " ".join(recommendation.uncertainties).lower()
        if not any(
            term in uncertainty_text
            for term in {"label", "coverage", "sample", "insufficient", "evidence"}
        ):
            violations.append(
                "Insufficient labels did not state an evidence-sufficiency uncertainty."
            )

    if scenario_name != "normal_operation":
        if not agentic_result.get("approval_required"):
            violations.append("Non-normal scenario did not require approval.")
        if not isinstance(approval, dict) or approval.get("decision") != "approve":
            violations.append("Non-normal scenario was not resumed with approval.")
        if final_status != "approved":
            violations.append("Approved non-normal scenario has an incompatible final status.")

    return list(dict.fromkeys(violations))


def evaluate_scenario(
    agentic_result: dict[str, Any],
    monitoring_result: MonitoringRunResult | dict[str, Any],
) -> ScenarioEvaluation:
    """Evaluate one completed agentic result without an LLM judge."""
    deterministic = (
        monitoring_result
        if isinstance(monitoring_result, MonitoringRunResult)
        else MonitoringRunResult.model_validate(monitoring_result)
    )
    scenario_name = deterministic.scenario_name
    expected = scenario_expectations(scenario_name)
    triage = agentic_result.get("triage", {})
    recommendation_payload = agentic_result.get("recommendation", {})
    verification = agentic_result.get("verification", {})
    violations: list[str] = []
    notes = [
        "Evaluation used deterministic schemas, evidence IDs, and policy rules; no judge LLM.",
        "No external remediation execution is part of the agent workflow.",
    ]

    deterministic_evidence = {
        item.evidence_id: item.model_dump(mode="json")
        for item in deterministic.evidence
    }
    valid_ids = set(deterministic_evidence)
    all_citations = _all_citations(triage, recommendation_payload)
    unsupported = sorted(all_citations - valid_ids)
    all_cited_valid = not unsupported
    if unsupported:
        violations.append(f"Unsupported deterministic evidence IDs: {unsupported}.")

    claims = recommendation_payload.get("claims", [])
    actions = recommendation_payload.get("recommended_actions", [])
    all_claims_cited = bool(claims) and all(
        bool(claim.get("evidence_ids"))
        and set(claim["evidence_ids"]) <= valid_ids
        for claim in claims
    )
    all_actions_cited = bool(actions) and all(
        bool(action.get("evidence_ids"))
        and set(action["evidence_ids"]) <= valid_ids
        for action in actions
    )
    if not all_claims_cited:
        violations.append("One or more claims lack valid deterministic evidence.")
    if not all_actions_cited:
        violations.append("One or more actions lack valid deterministic evidence.")

    incident_type = str(recommendation_payload.get("incident_type", ""))
    diagnostic_route = str(triage.get("diagnostic_route", ""))
    incident_compatible = incident_type in expected["incident_types"]
    route_compatible = diagnostic_route in expected["routes"]
    if not incident_compatible:
        violations.append("Incident type is incompatible with scenario expectations.")
    if not route_compatible:
        violations.append("Diagnostic route is incompatible with scenario expectations.")

    used_fallback = bool(agentic_result.get("fallback_used"))
    triage_calls = _provider_calls(agentic_result, "AgentTriage")
    recommendation_calls = _provider_calls(agentic_result, "AgentRecommendation")
    triage_parse_success = any(
        item.get("parse_success") is True for item in triage_calls
    )
    recommendation_parse_success = (
        not used_fallback
        and any(item.get("parse_success") is True for item in recommendation_calls)
    )
    if not triage_parse_success:
        violations.append("Live triage structured output did not parse successfully.")
    if not recommendation_parse_success:
        violations.append("Live recommendation structured output did not parse successfully.")

    policy_violations: list[str]
    try:
        recommendation = AgentRecommendation.model_validate(recommendation_payload)
        policy_violations = _scenario_policy_violations(
            scenario_name,
            agentic_result,
            deterministic,
            recommendation,
            deterministic_evidence,
        )
    except ValidationError as exc:
        policy_violations = [
            f"Final recommendation failed schema validation: {exc.errors(include_input=False)}"
        ]
    violations.extend(policy_violations)
    policy_compliant = not policy_violations

    revision_count = int(agentic_result.get("revision_count", 0))
    final_verification_passed = verification.get("status") == "pass"
    first_pass = (
        final_verification_passed and revision_count == 0 and not used_fallback
    )
    final_status = str(agentic_result.get("final_status", ""))
    completed = final_status in _COMPLETED_STATUSES
    approval_required = bool(agentic_result.get("approval_required"))
    approval = agentic_result.get("approval_decision")
    if scenario_name == "normal_operation":
        approval_completed = (
            not approval_required
            and approval is None
            and final_status == "completed_no_approval_required"
        )
    else:
        approval_completed = (
            approval_required
            and isinstance(approval, dict)
            and approval.get("decision") == "approve"
            and final_status == "approved"
        )
    if not completed:
        violations.append("Agent run did not reach a completed final status.")
    if not approval_completed:
        violations.append("Approval behavior did not complete as expected.")

    calls = _provider_calls(agentic_result)
    latencies = [
        float(item["latency_ms"])
        for item in calls
        if isinstance(item.get("latency_ms"), int | float)
    ]
    latency_ms = round(sum(latencies), 2) if latencies else None
    input_tokens = _token_total(calls, "input_tokens")
    output_tokens = _token_total(calls, "output_tokens")
    total_tokens = _token_total(calls, "total_tokens")
    if input_tokens is None or output_tokens is None or total_tokens is None:
        notes.append("One or more token-usage fields were unavailable from the provider.")

    return ScenarioEvaluation(
        scenario_name=scenario_name,
        provider=str(agentic_result.get("provider", "")),
        model=str(agentic_result.get("model", "")),
        execution_mode=str(agentic_result.get("execution_mode", "")),
        completed=completed,
        used_fallback=used_fallback,
        triage_parse_success=triage_parse_success,
        recommendation_parse_success=recommendation_parse_success,
        first_pass_verification=first_pass,
        final_verification_passed=final_verification_passed,
        revision_count=revision_count,
        incident_type=incident_type,
        expected_incident_types=expected["incident_types"],
        incident_compatible=incident_compatible,
        diagnostic_route=diagnostic_route,
        expected_routes=expected["routes"],
        route_compatible=route_compatible,
        all_cited_evidence_valid=all_cited_valid,
        all_claims_cited=all_claims_cited,
        all_actions_cited=all_actions_cited,
        policy_compliant=policy_compliant,
        approval_required=approval_required,
        approval_completed=approval_completed,
        final_status=final_status,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        violations=list(dict.fromkeys(violations)),
        notes=notes,
    )


def _rate(results: list[ScenarioEvaluation], predicate: Any) -> float:
    if not results:
        return 0.0
    return round(sum(bool(predicate(item)) for item in results) / len(results), 4)


def _optional_sum(values: list[int | None]) -> int | None:
    available = [value for value in values if value is not None]
    return sum(available) if available else None


def evaluate_live_runs(
    runs: list[tuple[dict[str, Any], MonitoringRunResult | dict[str, Any]]],
) -> LiveEvaluationSummary:
    """Aggregate deterministic metrics across the required scenario results."""
    results = [evaluate_scenario(agent, monitoring) for agent, monitoring in runs]
    providers = {item.provider for item in results}
    models = {item.model for item in results}
    provider = next(iter(providers)) if len(providers) == 1 else "mixed"
    model = next(iter(models)) if len(models) == 1 else "mixed"
    latency_values = [
        item.latency_ms for item in results if item.latency_ms is not None
    ]
    return LiveEvaluationSummary(
        provider=provider,
        model=model,
        scenario_count=len(results),
        completed_count=sum(item.completed for item in results),
        structured_output_success_rate=_rate(
            results,
            lambda item: (
                item.triage_parse_success and item.recommendation_parse_success
            ),
        ),
        incident_compatibility_rate=_rate(
            results,
            lambda item: item.incident_compatible,
        ),
        route_compatibility_rate=_rate(
            results,
            lambda item: item.route_compatible,
        ),
        evidence_grounding_rate=_rate(
            results,
            lambda item: (
                item.all_cited_evidence_valid
                and item.all_claims_cited
                and item.all_actions_cited
            ),
        ),
        policy_compliance_rate=_rate(
            results,
            lambda item: item.policy_compliant,
        ),
        first_pass_verification_rate=_rate(
            results,
            lambda item: item.first_pass_verification,
        ),
        fallback_rate=_rate(results, lambda item: item.used_fallback),
        approval_completion_rate=_rate(
            results,
            lambda item: item.approval_completed,
        ),
        mean_latency_ms=(
            round(statistics.mean(latency_values), 2) if latency_values else None
        ),
        median_latency_ms=(
            round(statistics.median(latency_values), 2) if latency_values else None
        ),
        total_input_tokens=_optional_sum([item.input_tokens for item in results]),
        total_output_tokens=_optional_sum([item.output_tokens for item in results]),
        total_tokens=_optional_sum([item.total_tokens for item in results]),
        scenario_results=results,
        limitations=[
            (
                f"The evaluation contains {len(results)} controlled replay scenarios, "
                "not production traffic."
            ),
            "Monitoring thresholds and synthetic transformations are not production-validated.",
            "Deterministic checks assess schema, evidence, routing, and policy, not prose quality.",
            "A successful replay evaluation does not establish production readiness.",
        ],
        created_at_utc=datetime.now(UTC),
    )
