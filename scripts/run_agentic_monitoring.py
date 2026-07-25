"""Run deterministic monitoring followed by the bounded LangGraph orchestrator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from langgraph.types import Command
from rich.console import Console
from rich.table import Table

from monitoring_agent.agent.checkpointing import create_checkpointer
from monitoring_agent.agent.config import AgentSettings
from monitoring_agent.agent.graph import build_monitoring_graph
from monitoring_agent.agent.llm import (
    GroqStructuredMonitoringLLM,
    StructuredCallResult,
)
from monitoring_agent.agent.nodes import (
    approval_resume_payload,
    utc_approval_payload,
)
from monitoring_agent.monitoring.engine import MonitoringEngine
from monitoring_agent.paths import PROJECT_ROOT
from monitoring_agent.scenarios.generator import SUPPORTED_SCENARIOS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run evidence-backed LangGraph monitoring."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Run all replay scenarios.")
    selection.add_argument("--scenario", choices=SUPPORTED_SCENARIOS)
    parser.add_argument("--model-id", default="credit_default")
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
    parser.add_argument(
        "--checkpoint-backend",
        choices=["memory", "sqlite"],
        help="Override AGENT_CHECKPOINT_BACKEND.",
    )
    parser.add_argument(
        "--checkpoint-db",
        type=Path,
        help="SQLite checkpoint path, relative to the project root unless absolute.",
    )
    parser.add_argument("--thread-id", help="Stable thread ID for a new single run.")
    parser.add_argument(
        "--run-label",
        type=_valid_run_label,
        help=(
            "Write agentic reports under an isolated model/run-label/scenario "
            "directory without changing default report locations."
        ),
    )
    parser.add_argument(
        "--pause-only",
        action="store_true",
        help="Stop normally after persisting the human-approval interrupt.",
    )
    parser.add_argument(
        "--resume-thread",
        help="Resume an approval interrupt stored by the SQLite backend.",
    )
    return parser


def _valid_run_label(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", value) is None:
        raise argparse.ArgumentTypeError(
            "run label must be 1-64 letters, digits, dots, underscores, or hyphens"
        )
    return value


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


def _ascii_json(payload: dict[str, Any]) -> str:
    """Render model text safely on legacy Windows console encodings."""
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _review_decision(
    console: Console,
    payload: dict[str, Any],
    *,
    decision: str | None,
    reviewer: str,
    comment: str | None,
) -> dict[str, Any]:
    console.print("\n[bold]Human approval required[/bold]")
    console.print(_ascii_json(payload), markup=False)
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


class _ApprovalOnlyResumeLLM:
    """Guard against any provider call while resuming an existing approval interrupt."""

    provider_name = "groq"
    is_fake = False

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @staticmethod
    def triage(payload: dict[str, Any]) -> StructuredCallResult:
        raise RuntimeError("Approval-only resume attempted an unexpected triage LLM call.")

    @staticmethod
    def recommend(payload: dict[str, Any]) -> StructuredCallResult:
        raise RuntimeError(
            "Approval-only resume attempted an unexpected recommendation LLM call."
        )


def _validate_options(args: argparse.Namespace, backend: str) -> None:
    if args.resume_thread:
        if args.all or args.scenario:
            raise ValueError("--resume-thread cannot be combined with --all or --scenario.")
        if args.thread_id:
            raise ValueError("--thread-id is only valid when starting a new run.")
        if args.pause_only:
            raise ValueError("--pause-only cannot be combined with --resume-thread.")
        if args.run_label:
            raise ValueError("--run-label cannot be combined with --resume-thread.")
        if backend != "sqlite":
            raise ValueError("Cross-process resume requires --checkpoint-backend sqlite.")
        if args.decision is None:
            raise ValueError("--decision is required when resuming a saved thread.")
        return
    if not args.all and args.scenario is None:
        raise ValueError("Choose --scenario, --all, or --resume-thread.")
    if args.all and (args.thread_id or args.pause_only):
        raise ValueError("--thread-id and --pause-only require one --scenario.")


def _resolved_checkpoint_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _checkpoint_state_path(backend: str, path: Path) -> str | None:
    if backend != "sqlite":
        return None
    resolved = _resolved_checkpoint_path(path)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _add_result_row(
    table: Table,
    result: dict[str, Any],
    *,
    llm_is_fake: bool,
) -> None:
    fallback_used = any(
        item.get("fallback_used") for item in result["llm_call_metadata"]
    )
    table.add_row(
        result["scenario_name"],
        result["recommendation"]["incident_type"],
        result["triage"]["diagnostic_route"],
        result["verification"]["status"],
        str(result["revision_count"]),
        str(bool(fallback_used)),
        result["final_status"],
        "FAKE" if llm_is_fake else "LIVE",
        ", ".join(result["final_report_paths"]),
    )


def _resume_saved_thread(
    graph: Any,
    *,
    thread_id: str,
    decision: str,
    reviewer: str,
    comment: str | None,
) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise ValueError(f"No saved checkpoint exists for thread {thread_id!r}.")
    if "human_approval" not in snapshot.next:
        raise ValueError(f"Thread {thread_id!r} is not awaiting human approval.")

    original_run_id = snapshot.values.get("run_id")
    original_recommendation = snapshot.values.get("recommendation")
    original_llm_metadata = snapshot.values.get("llm_call_metadata", [])
    approval = utc_approval_payload(decision, reviewer, comment)
    result = graph.invoke(
        Command(
            resume=approval_resume_payload(
                approval,
                resumed_from_checkpoint=True,
            )
        ),
        config=config,
    )
    if result.get("run_id") != original_run_id:
        raise RuntimeError("Saved run ID changed during approval-only resume.")
    if result.get("recommendation") != original_recommendation:
        raise RuntimeError("Saved recommendation changed during approval-only resume.")
    if result.get("llm_call_metadata") != original_llm_metadata:
        raise RuntimeError("Approval-only resume unexpectedly changed LLM call metadata.")
    return result


def main() -> int:
    """Run one graph per scenario while preserving each stable thread on resume."""
    args = _parser().parse_args()
    console = Console(width=180)
    settings = AgentSettings()
    backend = args.checkpoint_backend or settings.checkpoint_backend
    checkpoint_path = args.checkpoint_db or settings.checkpoint_db
    try:
        _validate_options(args, backend)
        llm = (
            _ApprovalOnlyResumeLLM(settings.model)
            if args.resume_thread
            else (
                _load_fake_llm()
                if args.use_fake_llm
                else GroqStructuredMonitoringLLM(settings)
            )
        )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        return 2

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

    try:
        with create_checkpointer(
            backend,
            checkpoint_path if backend == "sqlite" else None,
        ) as checkpointer:
            graph = build_monitoring_graph(llm, settings, checkpointer=checkpointer)
            if args.resume_thread:
                result = _resume_saved_thread(
                    graph,
                    thread_id=args.resume_thread,
                    decision=args.decision,
                    reviewer=args.reviewer,
                    comment=args.comment,
                )
                _add_result_row(table, result, llm_is_fake=False)
            else:
                scenario_names = (
                    list(SUPPORTED_SCENARIOS) if args.all else [args.scenario]
                )
                engine = MonitoringEngine(model_id=args.model_id)
                for scenario_name in scenario_names:
                    try:
                        monitoring_result = engine.run_scenario(scenario_name)
                    except FileNotFoundError as exc:
                        console.print(
                            f"[red]Scenario artifact missing:[/red] {exc}\n"
                            "Run the scenario generator for that scenario."
                        )
                        return 2

                    thread_id = (
                        args.thread_id
                        or f"agentic-{monitoring_result.run_id.lower()}"
                    )
                    config = {"configurable": {"thread_id": thread_id}}
                    initial_state = {
                        "thread_id": thread_id,
                        "checkpoint_backend": backend,
                        "checkpoint_database": _checkpoint_state_path(
                            backend,
                            checkpoint_path,
                        ),
                        "run_label": args.run_label,
                        "resumed_from_checkpoint": False,
                        "pause_count": 0,
                        "resume_count": 0,
                        "monitoring_result": monitoring_result.model_dump(mode="json"),
                        "llm_call_metadata": [],
                        "execution_errors": [],
                        "revision_count": 0,
                    }
                    result = graph.invoke(initial_state, config=config)
                    payload = _interrupt_payload(result)
                    if payload is not None and args.pause_only:
                        console.print(
                            f"[green]Paused at human approval.[/green] "
                            f"Thread: {thread_id} | Checkpoint: "
                            f"{_checkpoint_state_path(backend, checkpoint_path) or 'memory'}"
                        )
                        return 0
                    if payload is not None:
                        approval = _review_decision(
                            console,
                            payload,
                            decision=args.decision,
                            reviewer=args.reviewer,
                            comment=args.comment,
                        )
                        result = graph.invoke(
                            Command(
                                resume=approval_resume_payload(
                                    approval,
                                    resumed_from_checkpoint=False,
                                )
                            ),
                            config=config,
                        )
                    elif args.pause_only:
                        console.print(
                            "[red]The selected scenario completed without an approval "
                            "interrupt; no paused checkpoint was created.[/red]"
                        )
                        return 2
                    _add_result_row(
                        table,
                        result,
                        llm_is_fake=bool(llm.is_fake),
                    )
    except (OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Agent workflow error:[/red] {exc}")
        return 2

    console.print(table)
    if args.use_fake_llm:
        console.print(
            "[yellow]Offline demonstration used fake-structured-llm; no live Groq "
            "result was produced.[/yellow]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
