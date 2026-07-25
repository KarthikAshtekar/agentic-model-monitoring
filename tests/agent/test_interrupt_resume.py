"""Human interrupt and same-thread resume behavior."""

from datetime import UTC, datetime

import pytest
from langgraph.types import Command

from tests.agent.helpers import reports_exist, run_until_interrupt_or_end


@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [
        ("approve", "approved"),
        ("reject", "rejected"),
        ("request_revision", "revision_requested"),
    ],
)
def test_non_normal_pause_and_resume(
    decision: str,
    expected_status: str,
) -> None:
    graph, paused, config, _ = run_until_interrupt_or_end(
        "feature_drift",
        thread_suffix=f"resume-{decision}",
    )
    assert "__interrupt__" in paused
    thread_id = config["configurable"]["thread_id"]
    resumed = graph.invoke(
        Command(
            resume={
                "decision": decision,
                "comment": "Focused interrupt-resume test.",
                "reviewer": "pytest-reviewer",
                "reviewed_at_utc": datetime.now(UTC).isoformat(),
            }
        ),
        config=config,
    )
    assert resumed["thread_id"] == thread_id
    assert resumed["final_status"] == expected_status
    assert resumed["approval_decision"]["decision"] == decision
    assert reports_exist(resumed["final_report_paths"])
