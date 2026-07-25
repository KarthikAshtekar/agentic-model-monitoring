"""Data-quality behavior against the generated failure scenario."""

from monitoring_agent.monitoring.engine import MonitoringEngine


def test_data_quality_failure_blocks_downstream_evaluation() -> None:
    """Duplicate IDs block inference and suppress performance conclusions."""
    result = MonitoringEngine().run_scenario("data_quality_failure")

    assert result.data_quality.duplicate_record_count > 0
    assert result.batch_blocked is True
    assert "data_quality_failure" in result.incident_candidates
    assert result.performance.evaluated is False
    assert result.drift.evaluated_feature_count == 0
    assert any(
        item.evidence_id == "DQ-DUPLICATE-RECORD-ID"
        and item.status == "critical"
        for item in result.evidence
    )
