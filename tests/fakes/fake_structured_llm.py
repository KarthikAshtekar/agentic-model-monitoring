"""Deterministic test-only structured LLM for offline graph validation."""

from __future__ import annotations

from typing import Any

from monitoring_agent.agent.llm import StructuredCallResult
from monitoring_agent.agent.schemas import (
    AgentRecommendation,
    AgentTriage,
    EvidenceBackedClaim,
    RecommendedAction,
)


class FakeStructuredMonitoringLLM:
    """Scenario-aware fake with explicit, injectable failure modes."""

    provider_name = "fake"
    model_name = "fake-structured-llm"
    is_fake = True

    def __init__(self, failure_mode: str | None = None) -> None:
        self.failure_mode = failure_mode
        self.triage_call_count = 0
        self.recommendation_call_count = 0

    def _metadata(self, schema: str, parse_success: bool) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "schema": schema,
            "latency_ms": 0.0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "parse_success": parse_success,
            "is_fake": True,
        }

    def _failure(self, schema: str) -> StructuredCallResult | None:
        if self.failure_mode == "api_failure":
            return StructuredCallResult(
                parsed=None,
                metadata=self._metadata(schema, False),
                error={
                    "error_type": "FakeAPIError",
                    "error_message": "Injected offline API failure.",
                },
            )
        if self.failure_mode == "parsing_failure":
            return StructuredCallResult(
                parsed=None,
                metadata=self._metadata(schema, False),
                error={
                    "error_type": "FakeParsingError",
                    "error_message": "Injected offline structured parsing failure.",
                },
            )
        return None

    def triage(self, payload: dict[str, Any]) -> StructuredCallResult:
        """Return scenario-specific valid triage unless failure is injected."""
        self.triage_call_count += 1
        failure = self._failure("AgentTriage")
        if failure is not None:
            return failure
        summary = payload["monitoring_case"]["case_summary"]
        incident = summary["incident_candidates"][0]
        routes = {
            "normal_operation": "no_additional_diagnostics",
            "data_quality_failure": "data_quality_diagnostics",
            "feature_drift": "drift_diagnostics",
            "prediction_drift": "drift_diagnostics",
            "performance_degradation": "performance_diagnostics",
            "mixed_incident": "mixed_diagnostics",
            "insufficient_evidence": "evidence_sufficiency_review",
        }
        evidence = payload["monitoring_case"]["evidence"]
        ids = [
            item["evidence_id"]
            for item in evidence
            if item["status"] in {"critical", "warning", "not_evaluated"}
        ]
        if not ids:
            ids = [item["evidence_id"] for item in evidence[:6]]
        triage = AgentTriage(
            incident_type=incident,
            severity=summary["overall_severity"],
            diagnostic_route=routes[incident],
            needs_additional_diagnostics=incident != "normal_operation",
            selected_evidence_ids=ids[:12],
            reason="Offline fake selected the leading deterministic candidate and route.",
        )
        return StructuredCallResult(
            parsed=triage,
            metadata=self._metadata("AgentTriage", True),
            error=None,
        )

    @staticmethod
    def _material_ids(payload: dict[str, Any]) -> list[str]:
        evidence = payload["diagnostic_context"]["evidence"]
        ids = [
            item["evidence_id"]
            for item in evidence
            if item["status"] in {"critical", "warning", "not_evaluated"}
        ]
        if not ids:
            ids = [item["evidence_id"] for item in evidence]
        return ids[:8]

    def recommend(self, payload: dict[str, Any]) -> StructuredCallResult:
        """Return a grounded recommendation or one of the requested invalid variants."""
        self.recommendation_call_count += 1
        failure = self._failure("AgentRecommendation")
        if failure is not None:
            return failure

        incident = payload["triage"]["incident_type"]
        severity = payload["triage"]["severity"]
        policy = payload["policy_context"]
        diabetes_screening = policy.get("domain_id") == "diabetes_screening"
        approval = incident != "normal_operation"
        evidence_ids = self._material_ids(payload)
        if not evidence_ids:
            evidence_ids = payload["triage"]["selected_evidence_ids"][:4]

        if incident == "normal_operation":
            claim_text = (
                "Deterministic checks support normal operation for the survey-based "
                "diabetes-risk screening model."
                if diabetes_screening
                else "Deterministic checks support normal operation for the credit-risk model."
            )
            action_type = "continue_monitoring"
            action_text = "Continue scheduled deterministic monitoring."
            priority = "low"
            hypothesis = None
            hypothesis_ids: list[str] = []
            uncertainties: list[str] = []
        elif incident == "data_quality_failure":
            claim_text = (
                "The deterministic data-quality gate blocked these screening records."
                if diabetes_screening
                else "The deterministic data-quality gate blocked this credit-risk batch."
            )
            action_type = "quarantine_batch"
            action_text = "Keep the blocked batch quarantined pending pipeline investigation."
            priority = "urgent"
            hypothesis = (
                "Hypothesis: Upstream batch construction may explain the observed "
                "data-quality violations."
            )
            hypothesis_ids = evidence_ids
            uncertainties = [
                "The replay evidence does not establish the upstream implementation cause."
            ]
        elif incident == "performance_degradation":
            claim_text = (
                "Confirmed outcome labels found material screening-model performance "
                "degradation."
                if diabetes_screening
                else "Labelled evaluation found material credit-risk performance degradation."
            )
            action_type = "evaluate_retraining"
            action_text = "Evaluate retraining under governed offline validation."
            priority = "high"
            hypothesis = (
                "Hypothesis: The synthetic outcome shift may explain the measured "
                "performance degradation."
            )
            hypothesis_ids = evidence_ids
            uncertainties = [
                "The replayed label shift is synthetic and may not represent production."
            ]
        elif incident == "insufficient_evidence":
            claim_text = (
                "Only a partial labelled sample is available, so performance was not evaluated."
            )
            action_type = "collect_more_labels"
            action_text = "Collect enough aligned outcomes to meet the policy minimum."
            priority = "medium"
            hypothesis = None
            hypothesis_ids = []
            uncertainties = [
                "Conclusions are limited by insufficient label coverage."
            ]
        else:
            claim_text = (
                "Deterministic screening-model monitoring found material drift signals."
                if diabetes_screening
                else "Deterministic credit-risk monitoring found material drift signals."
            )
            action_type = "investigate_feature_drift"
            action_text = "Investigate the observed feature and prediction shifts."
            priority = "high"
            hypothesis = (
                "Hypothesis: The replayed input shift may contribute to the observed "
                "monitoring signals."
            )
            hypothesis_ids = evidence_ids
            uncertainties = [
                "Distribution shift does not by itself establish causality or model failure."
            ]

        invalid_first = self.failure_mode in {
            "invalid_evidence_ids",
            "unsupported_retraining",
        } and self.recommendation_call_count == 1
        invalid_always = self.failure_mode == "invalid_always"
        if (invalid_first or invalid_always) and self.failure_mode != "unsupported_retraining":
            evidence_ids = [*evidence_ids, "FAKE-HALLUCINATED-EVIDENCE"]
            if hypothesis is not None:
                hypothesis_ids = evidence_ids
        if (
            self.failure_mode == "unsupported_retraining"
            and self.recommendation_call_count == 1
        ):
            action_type = "evaluate_retraining"
            action_text = "Evaluate retraining despite unavailable performance evidence."

        action = RecommendedAction(
            action_type=action_type,
            action=action_text,
            rationale="This is a recommendation for human review, not automatic execution.",
            priority=priority,
            evidence_ids=evidence_ids,
            requires_human_approval=approval,
        )
        recommendation = AgentRecommendation(
            incident_type=incident,
            severity=severity,
            executive_summary=(
                "Offline fake recommendation grounded in deterministic replay evidence."
            ),
            claims=[
                EvidenceBackedClaim(
                    claim=claim_text,
                    evidence_ids=evidence_ids,
                )
            ],
            root_cause_hypothesis=hypothesis,
            root_cause_evidence_ids=hypothesis_ids,
            recommended_actions=[action],
            uncertainties=uncertainties,
            overall_evidence_ids=evidence_ids,
            requires_human_approval=approval,
            confidence=0.8,
        )
        return StructuredCallResult(
            parsed=recommendation,
            metadata=self._metadata("AgentRecommendation", True),
            error=None,
        )
