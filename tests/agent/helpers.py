"""Shared fixtures for focused agent tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monitoring_agent.agent.config import AgentSettings
from monitoring_agent.agent.graph import build_monitoring_graph
from monitoring_agent.monitoring.schemas import MonitoringRunResult
from monitoring_agent.paths import GENERATED_REPORTS_DIR
from tests.fakes.fake_structured_llm import FakeStructuredMonitoringLLM


def load_result(scenario_name: str) -> MonitoringRunResult:
    path = GENERATED_REPORTS_DIR / scenario_name / "monitoring_result.json"
    return MonitoringRunResult.model_validate(json.loads(path.read_text(encoding="utf-8")))


def run_until_interrupt_or_end(
    scenario_name: str,
    *,
    failure_mode: str | None = None,
    thread_suffix: str = "default",
) -> tuple[Any, dict[str, Any], dict[str, Any], FakeStructuredMonitoringLLM]:
    result = load_result(scenario_name)
    fake = FakeStructuredMonitoringLLM(failure_mode=failure_mode)
    settings = AgentSettings(max_revision_attempts=1)
    graph = build_monitoring_graph(fake, settings)
    thread_id = f"test-{scenario_name}-{thread_suffix}"
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "thread_id": thread_id,
        "monitoring_result": result.model_dump(mode="json"),
        "llm_call_metadata": [],
        "execution_errors": [],
        "revision_count": 0,
    }
    output = graph.invoke(initial_state, config=config)
    return graph, output, config, fake


def reports_exist(paths: list[str]) -> bool:
    project_root = Path(__file__).resolve().parents[2]
    return all((project_root / path).is_file() for path in paths)
