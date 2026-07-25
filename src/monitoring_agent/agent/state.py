"""Serializable LangGraph state for the monitoring orchestrator."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    """State passed between deterministic and LLM-backed graph nodes."""

    run_id: str
    thread_id: str
    scenario_name: str
    checkpoint_backend: str
    checkpoint_database: str | None
    resumed_from_checkpoint: bool
    pause_count: int
    resume_count: int
    monitoring_result: dict[str, Any]
    available_evidence: list[dict[str, Any]]
    selected_evidence: dict[str, Any]
    triage: dict[str, Any]
    diagnostic_context: dict[str, Any]
    recommendation: dict[str, Any]
    verification: dict[str, Any]
    revision_feedback: str | None
    revision_count: int
    approval_required: bool
    approval_decision: dict[str, Any] | None
    llm_call_metadata: Annotated[list[dict[str, Any]], operator.add]
    execution_errors: Annotated[list[dict[str, Any]], operator.add]
    final_status: str
    final_report_paths: list[str]
