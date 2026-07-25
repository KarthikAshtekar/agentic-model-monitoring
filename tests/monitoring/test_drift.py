"""Feature-drift behavior against the generated covariate-shift scenario."""

from monitoring_agent.monitoring.engine import MonitoringEngine


def test_feature_drift_is_critical_without_data_quality_block() -> None:
    """Configured credit-limit or repayment features produce critical PSI."""
    result = MonitoringEngine().run_scenario("feature_drift")
    intended_features = {"LIMIT_BAL", "PAY_0", "RecentPaymentDelay"}

    assert result.batch_blocked is False
    assert result.drift.critical_feature_count >= 1
    assert any(
        result.drift.feature_metrics[feature]["psi"] >= 0.25
        and result.drift.feature_metrics[feature]["primary_status"] == "critical"
        for feature in intended_features
    )
    assert "feature_drift" in result.incident_candidates
    assert not any(
        item.domain == "data_quality" and item.status == "critical"
        for item in result.evidence
    )
