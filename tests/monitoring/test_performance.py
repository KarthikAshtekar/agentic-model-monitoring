"""Labelled-performance behavior against synthetic outcome drift."""

from monitoring_agent.monitoring.engine import MonitoringEngine


def test_performance_degradation_is_material_without_critical_feature_drift() -> None:
    """Outcome-only changes degrade recall or PR-AUC while features remain stable."""
    result = MonitoringEngine().run_scenario("performance_degradation")
    deltas = result.performance.metric_deltas

    assert result.batch_blocked is False
    assert result.performance.evaluated is True
    assert result.drift.critical_feature_count == 0
    assert (
        deltas["operating_threshold"]["recall"] >= 0.08
        or deltas["pr_auc_drop"] >= 0.05
    )
    assert "performance_degradation" in result.incident_candidates
