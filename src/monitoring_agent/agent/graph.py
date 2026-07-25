"""LangGraph construction for the bounded single-agent workflow."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from monitoring_agent.agent.config import AgentSettings
from monitoring_agent.agent.llm import StructuredMonitoringLLM
from monitoring_agent.agent.nodes import AgentNodes
from monitoring_agent.agent.state import AgentState


def _after_verification(state: AgentState) -> str:
    status = state["verification"]["status"]
    if status == "revise":
        return "revise_recommendation"
    if status == "fallback":
        return "deterministic_fallback"
    return (
        "prepare_human_approval"
        if state["recommendation"]["requires_human_approval"]
        else "finalize"
    )


def _after_fallback(state: AgentState) -> str:
    return (
        "prepare_human_approval"
        if state["recommendation"]["requires_human_approval"]
        else "finalize"
    )


def build_monitoring_graph(
    llm: StructuredMonitoringLLM,
    settings: AgentSettings | None = None,
    *,
    checkpointer: Any | None = None,
) -> Any:
    """Compile the one-orchestrator graph with an in-memory MVP checkpointer."""
    nodes = AgentNodes(llm, settings or AgentSettings())
    builder = StateGraph(AgentState)
    builder.add_node("load_monitoring_case", nodes.load_monitoring_case)
    builder.add_node("select_evidence", nodes.select_evidence)
    builder.add_node("llm_triage", nodes.llm_triage)
    builder.add_node("route_diagnostics", nodes.route_diagnostics)
    builder.add_node(
        "prepare_diagnostic_context",
        nodes.prepare_diagnostic_context,
    )
    builder.add_node("llm_recommendation", nodes.llm_recommendation)
    builder.add_node("verify_recommendation", nodes.verify_recommendation)
    builder.add_node("revise_recommendation", nodes.revise_recommendation)
    builder.add_node("deterministic_fallback", nodes.deterministic_fallback)
    builder.add_node("prepare_human_approval", nodes.prepare_human_approval)
    builder.add_node("human_approval", nodes.human_approval)
    builder.add_node("finalize", nodes.finalize)

    builder.add_edge(START, "load_monitoring_case")
    builder.add_edge("load_monitoring_case", "select_evidence")
    builder.add_edge("select_evidence", "llm_triage")
    builder.add_edge("llm_triage", "route_diagnostics")
    builder.add_edge("route_diagnostics", "prepare_diagnostic_context")
    builder.add_edge("prepare_diagnostic_context", "llm_recommendation")
    builder.add_edge("llm_recommendation", "verify_recommendation")
    builder.add_conditional_edges(
        "verify_recommendation",
        _after_verification,
        {
            "revise_recommendation": "revise_recommendation",
            "deterministic_fallback": "deterministic_fallback",
            "prepare_human_approval": "prepare_human_approval",
            "finalize": "finalize",
        },
    )
    builder.add_edge("revise_recommendation", "llm_recommendation")
    builder.add_conditional_edges(
        "deterministic_fallback",
        _after_fallback,
        {
            "prepare_human_approval": "prepare_human_approval",
            "finalize": "finalize",
        },
    )
    builder.add_edge("prepare_human_approval", "human_approval")
    builder.add_edge("human_approval", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())
