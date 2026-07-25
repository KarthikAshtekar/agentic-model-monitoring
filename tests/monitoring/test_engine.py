"""End-to-end deterministic engine behavior for normal replay."""

from monitoring_agent.monitoring.engine import MonitoringEngine


def test_normal_operation_remains_noncritical() -> None:
    """Clean replay is valid and retains the normal-operation candidate."""
    result = MonitoringEngine().run_scenario("normal_operation")

    assert result.batch_valid is True
    assert result.batch_blocked is False
    assert result.incident_candidates[0] == "normal_operation"
    assert not any(
        item.domain == "data_quality" and item.status == "critical"
        for item in result.evidence
    )
    assert not any(
        item.domain == "performance" and item.status == "critical"
        for item in result.evidence
    )


def test_evidence_ids_are_unique_and_present_for_incidents() -> None:
    """All evidence is uniquely addressable, especially warning and critical records."""
    engine = MonitoringEngine()
    for scenario in (
        "normal_operation",
        "feature_drift",
        "data_quality_failure",
        "performance_degradation",
    ):
        result = engine.run_scenario(scenario)
        identifiers = [item.evidence_id for item in result.evidence]
        assert len(identifiers) == len(set(identifiers))
        assert all(
            item.evidence_id
            for item in result.evidence
            if item.status in {"warning", "critical"}
        )
