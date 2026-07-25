"""Run deterministic monitoring followed by the bounded LangGraph orchestrator."""

from __future__ import annotations

import argparse
import importlib.util
from typing import Any

from langgraph.types import Command
from rich.console import Console
from rich.table import Table

from monitoring_agent.agent.config import AgentSettings
from monitoring_agent.agent.graph import build_monitoring_graph
from monitoring_agent.agent.llm import GroqStructuredMonitoringLLM
from monitoring_agent.agent.nodes import utc_approval_payload
from monitoring_agent.monitoring.engine import MonitoringEngine
from monitoring_agent.paths import PROJECT_ROOT
from monitoring_agent.scenarios.generator import SUPPORTED_SCENARIOS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run evidence-backed LangGraph monitoring."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Run all replay scenarios.")
    selection.add_argument("--scenario", choices=SUPPORTED_SCENARIOS)
    parser.add_argument(
        "--decision",
        choices=["approve", "reject", "request_revision"],
        help="Non-interactive decision used if the graph interrupts.",
    )
    parser.add_argument("--reviewer", default="cli-reviewer")
    parser.add_argument("--comment")
    parser.add_argument(
        "--use-fake-llm",
        action="store_true",
        help="Use the clearly labelled test-only fake structured LLM.",
    )
    return parser


def _load_fake_llm() -> Any:
    """Load the test-only fake without placing fake behaviour in provider code."""
    fake_path = PROJECT_ROOT / "tests/fakes/fake_structured_llm.py"
    spec = importlib.util.spec_from_file_location("offline_fake_structured_llm", fake_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load offline fake LLM from {fake_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FakeStructuredMonitoringLLM()


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    return interrupts[0].value


def _review_decision(
    console: Console,
    payload: dict[str, Any],
    *,
    decision: str | None,
    reviewer: str,
    comment: str | None,
) -> dict[str, Any]:
    console.print("\n[bold]Human approval required[/bold]")
    console.print_json(data=payload)
    selected = decision
    if selected is None:
        selected = console.input(
            "Decision [approve/reject/request_revision]: "
        ).strip()
        while selected not in {"approve", "reject", "request_revision"}:
            selected = console.input(
                "Enter approve, reject, or request_revision: "
            ).strip()
    if selected == "request_revision" and comment is None:
        comment = (
            "Reviewer requested a future revision."
            if decision is not None
            else console.input("Reviewer comment: ").strip() or None
        )
    return utc_approval_payload(selected, reviewer, comment)


def main() -> int:
    """Run one graph per scenario while preserving each stable thread on resume."""
    args = _parser().parse_args()
    console = Console(width=180)
    settings = AgentSettings()
    try:
        llm = (
            _load_fake_llm()
            if args.use_fake_llm
            else GroqStructuredMonitoringLLM(settings)
        )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        return 2

    scenario_names = list(SUPPORTED_SCENARIOS) if args.all else [args.scenario]
    engine = MonitoringEngine()
    graph = build_monitoring_graph(llm, settings)
    table = Table(show_header=True, header_style="bold")
    for column in (
        "scenario",
        "incident",
        "route",
        "verification",
        "revisions",
        "fallback",
        "final_status",
        "LLM",
        "reports",
    ):
        table.add_column(column)

    for scenario_name in scenario_names:
        try:
            monitoring_result = engine.run_scenario(scenario_name)
        except FileNotFoundError as exc:
            console.print(
                f"[red]Scenario artifact missing:[/red] {exc}\n"
                "Run: python scripts\\generate_monitoring_scenarios.py"
            )
            return 2

        thread_id = f"agentic-{monitoring_result.run_id.lower()}"
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "thread_id": thread_id,
            "monitoring_result": monitoring_result.model_dump(mode="json"),
            "llm_call_metadata": [],
            "execution_errors": [],
            "revision_count": 0,
        }
        result = graph.invoke(initial_state, config=config)
        payload = _interrupt_payload(result)
        if payload is not None:
            approval = _review_decision(
                console,
                payload,
                decision=args.decision,
                reviewer=args.reviewer,
                comment=args.comment,
            )
            result = graph.invoke(Command(resume=approval), config=config)

        fallback_used = any(
            item.get("fallback_used") for item in result["llm_call_metadata"]
        )
        table.add_row(
            scenario_name,
            result["recommendation"]["incident_type"],
            result["triage"]["diagnostic_route"],
            result["verification"]["status"],
            str(result["revision_count"]),
            str(bool(fallback_used)),
            result["final_status"],
            "FAKE" if llm.is_fake else "LIVE",
            ", ".join(result["final_report_paths"]),
        )

    console.print(table)
    if args.use_fake_llm:
        console.print(
            "[yellow]Offline demonstration used fake-structured-llm; no live Groq "
            "result was produced.[/yellow]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
