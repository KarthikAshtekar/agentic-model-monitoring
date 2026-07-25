"""Durable SQLite checkpoint and cross-process approval-resume behavior."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from monitoring_agent.agent.checkpointing import create_checkpointer
from monitoring_agent.agent.config import AgentSettings
from monitoring_agent.agent.graph import build_monitoring_graph
from scripts.run_agentic_monitoring import (
    _parser,
    _resume_saved_thread,
    _validate_options,
)
from tests.agent.helpers import load_result
from tests.fakes.fake_structured_llm import FakeStructuredMonitoringLLM


def _initial_state(thread_id: str, checkpoint_database: str) -> dict[str, Any]:
    result = load_result("feature_drift")
    return {
        "thread_id": thread_id,
        "checkpoint_backend": "sqlite",
        "checkpoint_database": checkpoint_database,
        "resumed_from_checkpoint": False,
        "pause_count": 0,
        "resume_count": 0,
        "monitoring_result": result.model_dump(mode="json"),
        "llm_call_metadata": [],
        "execution_errors": [],
        "revision_count": 0,
    }


def _is_safe_state_value(value: Any) -> bool:
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(_is_safe_state_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_safe_state_value(item)
            for key, item in value.items()
        )
    return False


def test_sqlite_factory_creates_parent_and_database(tmp_path) -> None:
    database_path = tmp_path / "nested/checkpoints/test.sqlite"
    with create_checkpointer("sqlite", database_path) as checkpointer:
        assert isinstance(checkpointer, SqliteSaver)
    assert database_path.is_file()


def test_checkpoint_factory_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported checkpoint backend"):
        with create_checkpointer("postgres"):
            pass


@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [("approve", "approved"), ("reject", "rejected")],
)
def test_new_graph_resumes_saved_interrupt_without_llm_call(
    tmp_path,
    decision: str,
    expected_status: str,
) -> None:
    database_path = tmp_path / f"resume-{decision}.sqlite"
    thread_id = f"durable-{decision}"
    config = {"configurable": {"thread_id": thread_id}}
    first_fake = FakeStructuredMonitoringLLM()

    with create_checkpointer("sqlite", database_path) as first_checkpointer:
        first_graph = build_monitoring_graph(
            first_fake,
            AgentSettings(),
            checkpointer=first_checkpointer,
        )
        paused = first_graph.invoke(
            _initial_state(thread_id, str(database_path)),
            config=config,
        )
        snapshot = first_graph.get_state(config)
        original_run_id = snapshot.values["run_id"]
        original_recommendation = snapshot.values["recommendation"]

        assert "__interrupt__" in paused
        assert snapshot.next == ("human_approval",)
        assert snapshot.values["pause_count"] == 1
        assert first_fake.recommendation_call_count == 1
        assert _is_safe_state_value(snapshot.values)
        serialized = json.dumps(snapshot.values).lower()
        assert "groq_api_key" not in serialized
        assert "fitted model" not in serialized

    second_fake = FakeStructuredMonitoringLLM()
    with create_checkpointer("sqlite", database_path) as second_checkpointer:
        second_graph = build_monitoring_graph(
            second_fake,
            AgentSettings(),
            checkpointer=second_checkpointer,
        )
        resumed = _resume_saved_thread(
            second_graph,
            thread_id=thread_id,
            decision=decision,
            reviewer="pytest-durable",
            comment=None,
        )

    assert resumed["run_id"] == original_run_id
    assert resumed["recommendation"] == original_recommendation
    assert resumed["final_status"] == expected_status
    assert resumed["resumed_from_checkpoint"] is True
    assert resumed["pause_count"] == 1
    assert resumed["resume_count"] == 1
    assert second_fake.triage_call_count == 0
    assert second_fake.recommendation_call_count == 0


def test_missing_sqlite_thread_returns_clear_error(tmp_path) -> None:
    database_path = tmp_path / "missing.sqlite"
    with create_checkpointer("sqlite", database_path) as checkpointer:
        graph = build_monitoring_graph(
            FakeStructuredMonitoringLLM(),
            AgentSettings(),
            checkpointer=checkpointer,
        )
        with pytest.raises(ValueError, match="No saved checkpoint exists"):
            _resume_saved_thread(
                graph,
                thread_id="missing-thread",
                decision="approve",
                reviewer="pytest",
                comment=None,
            )


def test_memory_backend_rejects_cross_process_resume() -> None:
    args = _parser().parse_args(
        [
            "--resume-thread",
            "memory-thread",
            "--checkpoint-backend",
            "memory",
            "--decision",
            "approve",
        ]
    )
    with pytest.raises(ValueError, match="requires --checkpoint-backend sqlite"):
        _validate_options(args, "memory")
