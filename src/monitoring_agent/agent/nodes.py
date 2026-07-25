"""Node implementations for the single LangGraph monitoring orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langgraph.types import interrupt
from pydantic import ValidationError

from monitoring_agent.agent.config import AgentSettings
from monitoring_agent.agent.evidence_selector import build_evidence_packet
from monitoring_agent.agent.fallback import build_fallback_recommendation
from monitoring_agent.agent.llm import StructuredMonitoringLLM
from monitoring_agent.agent.policy import (
    build_policy_context,
    deterministic_triage,
    triage_violations,
)
from monitoring_agent.agent.reporting import write_agentic_reports
from monitoring_agent.agent.schemas import (
    AgentRecommendation,
    AgentTriage,
    ApprovalDecision,
    VerificationResult,
)
from monitoring_agent.agent.state import AgentState
from monitoring_agent.agent.verifier import verify_recommendation
from monitoring_agent.monitoring.schemas import MonitoringRunResult


def _error(stage: str, error_type: str, message: str) -> dict[str, str]:
    return {
        "stage": stage,
        "error_type": error_type,
        "error_message": message,
    }


def _rank_evidence(item: dict[str, Any]) -> tuple[int, int, str]:
    status_rank = {"critical": 0, "warning": 1, "not_evaluated": 2, "pass": 3}
    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    return (
        status_rank[item["status"]],
        severity_rank[item["severity"]],
        item["evidence_id"],
    )


def _diagnostic_evidence(
    route: str,
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    route_domains = {
        "data_quality_diagnostics": {"data_quality", "system"},
        "drift_diagnostics": {"feature_drift", "prediction_drift", "system"},
        "performance_diagnostics": {"performance", "system"},
        "mixed_diagnostics": {
            "feature_drift",
            "prediction_drift",
            "performance",
            "system",
        },
        "evidence_sufficiency_review": {
            "data_quality",
            "feature_drift",
            "prediction_drift",
            "performance",
            "system",
        },
        "no_additional_diagnostics": {
            "data_quality",
            "feature_drift",
            "prediction_drift",
            "performance",
            "system",
        },
    }
    filtered = [
        item for item in evidence if item["domain"] in route_domains[route]
    ]
    if route == "evidence_sufficiency_review":
        filtered = [
            item
            for item in filtered
            if item["status"] in {"warning", "critical", "not_evaluated"}
            or item["domain"] == "system"
        ]
    return sorted(filtered, key=_rank_evidence)[:16]


class AgentNodes:
    """Dependency-injected graph nodes with no hidden mutable graph state."""

    def __init__(
        self,
        llm: StructuredMonitoringLLM,
        settings: AgentSettings,
    ) -> None:
        self.llm = llm
        self.settings = settings

    @staticmethod
    def load_monitoring_case(state: AgentState) -> dict[str, Any]:
        """Validate the authoritative monitoring result and initialise run state."""
        result = MonitoringRunResult.model_validate(state["monitoring_result"])
        return {
            "run_id": result.run_id,
            "scenario_name": result.scenario_name,
            "monitoring_result": result.model_dump(mode="json"),
            "revision_count": int(state.get("revision_count", 0)),
            "checkpoint_backend": state.get("checkpoint_backend", "memory"),
            "checkpoint_database": state.get("checkpoint_database"),
            "resumed_from_checkpoint": bool(
                state.get("resumed_from_checkpoint", False)
            ),
            "pause_count": int(state.get("pause_count", 0)),
            "resume_count": int(state.get("resume_count", 0)),
            "approval_required": False,
            "approval_decision": None,
            "final_report_paths": [],
        }

    @staticmethod
    def select_evidence(state: AgentState) -> dict[str, Any]:
        """Build the bounded LLM packet from authoritative evidence."""
        packet = build_evidence_packet(state["monitoring_result"])
        return {
            "available_evidence": packet["evidence"],
            "selected_evidence": packet,
        }

    def llm_triage(self, state: AgentState) -> dict[str, Any]:
        """Call structured triage, falling back to deterministic routing on failure."""
        result = MonitoringRunResult.model_validate(state["monitoring_result"])
        payload = {
            "monitoring_case": state["selected_evidence"],
            "policy_context": build_policy_context(result),
        }
        call = self.llm.triage(payload)
        metadata = [{"stage": "triage", **call.metadata}]
        errors: list[dict[str, str]] = []
        triage: AgentTriage | None = None
        if call.error is not None:
            errors.append(
                _error(
                    "triage",
                    call.error["error_type"],
                    call.error["error_message"],
                )
            )
        elif call.parsed is not None:
            try:
                triage = AgentTriage.model_validate(call.parsed)
                violations = triage_violations(
                    triage,
                    result,
                    {item["evidence_id"] for item in state["available_evidence"]},
                )
                if violations:
                    errors.append(
                        _error("triage", "TriageVerificationError", " ".join(violations))
                    )
                    triage = None
            except ValidationError as exc:
                errors.append(_error("triage", type(exc).__name__, str(exc)))

        if triage is None:
            triage = deterministic_triage(result, state["available_evidence"])
            metadata.append(
                {
                    "stage": "triage_fallback",
                    "provider": "deterministic",
                    "model": None,
                    "schema": "AgentTriage",
                    "latency_ms": 0.0,
                    "token_usage": None,
                    "parse_success": True,
                    "is_fake": False,
                }
            )
        return {
            "triage": triage.model_dump(mode="json"),
            "llm_call_metadata": metadata,
            "execution_errors": errors,
        }

    @staticmethod
    def route_diagnostics(state: AgentState) -> dict[str, Any]:
        """Record the one controlled deterministic route selected by triage."""
        return {
            "diagnostic_context": {
                "route": state["triage"]["diagnostic_route"],
                "evidence": [],
            }
        }

    @staticmethod
    def prepare_diagnostic_context(state: AgentState) -> dict[str, Any]:
        """Select already-calculated route evidence without recomputing metrics."""
        route = state["triage"]["diagnostic_route"]
        return {
            "diagnostic_context": {
                "route": route,
                "evidence": _diagnostic_evidence(
                    route,
                    state["available_evidence"],
                ),
                "recalculated_metrics": False,
            }
        }

    def llm_recommendation(self, state: AgentState) -> dict[str, Any]:
        """Request a strict recommendation, including feedback on the one revision."""
        payload = {
            "monitoring_case": state["selected_evidence"]["case_summary"],
            "triage": state["triage"],
            "diagnostic_context": state["diagnostic_context"],
            "policy_context": build_policy_context(state["monitoring_result"]),
            "verifier_feedback": state.get("revision_feedback"),
            "revision_attempt": state.get("revision_count", 0),
        }
        call = self.llm.recommend(payload)
        metadata = [{"stage": "recommendation", **call.metadata}]
        errors: list[dict[str, str]] = []
        if call.error is not None:
            errors.append(
                _error(
                    "recommendation",
                    call.error["error_type"],
                    call.error["error_message"],
                )
            )
            return {
                "recommendation": {},
                "llm_call_metadata": metadata,
                "execution_errors": errors,
            }
        try:
            recommendation = AgentRecommendation.model_validate(call.parsed)
        except ValidationError as exc:
            errors.append(_error("recommendation", type(exc).__name__, str(exc)))
            return {
                "recommendation": {},
                "llm_call_metadata": metadata,
                "execution_errors": errors,
            }
        return {
            "recommendation": recommendation.model_dump(mode="json"),
            "approval_required": recommendation.requires_human_approval,
            "llm_call_metadata": metadata,
            "execution_errors": errors,
        }

    def verify_recommendation(self, state: AgentState) -> dict[str, Any]:
        """Verify the recommendation or direct provider failures to fallback."""
        if not state.get("recommendation"):
            result = VerificationResult(
                status="fallback",
                violations=["No parseable structured recommendation was available."],
                unsupported_evidence_ids=[],
                policy_checks=["structured_recommendation_available"],
                revision_feedback=None,
            )
        else:
            recommendation = AgentRecommendation.model_validate(state["recommendation"])
            result = verify_recommendation(
                recommendation,
                state["monitoring_result"],
                state["available_evidence"],
                revision_count=state.get("revision_count", 0),
                max_revision_attempts=min(
                    self.settings.max_revision_attempts,
                    1,
                ),
            )
        return {
            "verification": result.model_dump(mode="json"),
            "revision_feedback": result.revision_feedback,
        }

    @staticmethod
    def revise_recommendation(state: AgentState) -> dict[str, Any]:
        """Increment the bounded revision counter before one more structured call."""
        return {"revision_count": state.get("revision_count", 0) + 1}

    @staticmethod
    def deterministic_fallback(state: AgentState) -> dict[str, Any]:
        """Replace an unavailable or rejected recommendation with a safe fallback."""
        recommendation = build_fallback_recommendation(
            state["monitoring_result"],
            state["available_evidence"],
        )
        return {
            "recommendation": recommendation.model_dump(mode="json"),
            "approval_required": recommendation.requires_human_approval,
            "llm_call_metadata": [
                {
                    "stage": "deterministic_fallback",
                    "provider": "deterministic",
                    "model": None,
                    "schema": "AgentRecommendation",
                    "latency_ms": 0.0,
                    "token_usage": None,
                    "parse_success": True,
                    "is_fake": False,
                    "fallback_used": True,
                }
            ],
        }

    @staticmethod
    def prepare_human_approval(state: AgentState) -> dict[str, Any]:
        """Commit pause provenance before entering the interrupting node."""
        return {
            "approval_required": True,
            "pause_count": int(state.get("pause_count", 0)) + 1,
        }

    @staticmethod
    def human_approval(state: AgentState) -> dict[str, Any]:
        """Pause non-normal runs and validate the reviewer response on resume."""
        recommendation = AgentRecommendation.model_validate(state["recommendation"])
        response = interrupt(
            {
                "scenario": state["scenario_name"],
                "incident_type": recommendation.incident_type,
                "severity": recommendation.severity,
                "executive_summary": recommendation.executive_summary,
                "recommended_actions": [
                    action.model_dump(mode="json")
                    for action in recommendation.recommended_actions
                ],
                "evidence_ids": recommendation.overall_evidence_ids,
                "allowed_decisions": ["approve", "reject", "request_revision"],
            }
        )
        resume_context: dict[str, Any] = {}
        decision_payload = response
        if isinstance(response, dict) and "approval_decision" in response:
            decision_payload = response["approval_decision"]
            context_payload = response.get("resume_context", {})
            if isinstance(context_payload, dict):
                resume_context = context_payload
        decision = ApprovalDecision.model_validate(decision_payload)
        return {
            "approval_decision": decision.model_dump(mode="json"),
            "resumed_from_checkpoint": bool(
                resume_context.get("resumed_from_checkpoint", False)
            ),
            "resume_count": int(state.get("resume_count", 0)) + 1,
        }

    @staticmethod
    def finalize(state: AgentState) -> dict[str, Any]:
        """Assign the terminal status and write both agentic artifacts."""
        approval = state.get("approval_decision")
        if not state["approval_required"]:
            status = "completed_no_approval_required"
        elif approval is None:
            status = "fallback_pending_review"
        else:
            status = {
                "approve": "approved",
                "reject": "rejected",
                "request_revision": "revision_requested",
            }[approval["decision"]]
        report_state = {**state, "final_status": status}
        paths = write_agentic_reports(report_state)
        return {"final_status": status, "final_report_paths": paths}


def utc_approval_payload(
    decision: str,
    reviewer: str,
    comment: str | None,
) -> dict[str, Any]:
    """Build the exact serializable payload expected by Command(resume=...)."""
    return {
        "decision": decision,
        "comment": comment,
        "reviewer": reviewer,
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
    }


def approval_resume_payload(
    approval: dict[str, Any],
    *,
    resumed_from_checkpoint: bool,
) -> dict[str, Any]:
    """Wrap reviewer data with non-user workflow provenance for the interrupt node."""
    return {
        "approval_decision": approval,
        "resume_context": {
            "resumed_from_checkpoint": resumed_from_checkpoint,
        },
    }
