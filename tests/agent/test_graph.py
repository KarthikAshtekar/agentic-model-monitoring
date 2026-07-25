"""End-to-end graph routing with the injected structured fake."""

from tests.agent.helpers import reports_exist, run_until_interrupt_or_end


def test_normal_operation_completes_without_interrupt() -> None:
    _, output, _, _ = run_until_interrupt_or_end(
        "normal_operation",
        thread_suffix="normal",
    )
    assert "__interrupt__" not in output
    assert output["final_status"] == "completed_no_approval_required"
    assert output["triage"]["diagnostic_route"] == "no_additional_diagnostics"
    assert reports_exist(output["final_report_paths"])


def test_feature_drift_follows_mixed_diagnostics() -> None:
    _, output, _, _ = run_until_interrupt_or_end(
        "feature_drift",
        thread_suffix="feature",
    )
    assert "__interrupt__" in output
    assert output["triage"]["diagnostic_route"] in {
        "drift_diagnostics",
        "mixed_diagnostics",
    }


def test_data_quality_failure_preserves_block_and_reaches_approval() -> None:
    _, output, _, _ = run_until_interrupt_or_end(
        "data_quality_failure",
        thread_suffix="dq",
    )
    assert "__interrupt__" in output
    assert output["monitoring_result"]["batch_blocked"] is True
    assert output["recommendation"]["incident_type"] == "data_quality_failure"


def test_performance_degradation_follows_performance_diagnostics() -> None:
    _, output, _, _ = run_until_interrupt_or_end(
        "performance_degradation",
        thread_suffix="performance",
    )
    assert "__interrupt__" in output
    assert output["triage"]["diagnostic_route"] == "performance_diagnostics"


def test_invalid_first_recommendation_triggers_one_revision() -> None:
    _, output, _, fake = run_until_interrupt_or_end(
        "performance_degradation",
        failure_mode="invalid_evidence_ids",
        thread_suffix="revise-once",
    )
    assert "__interrupt__" in output
    assert output["revision_count"] == 1
    assert output["verification"]["status"] == "pass"
    assert fake.recommendation_call_count == 2


def test_invalid_revised_recommendation_triggers_fallback() -> None:
    _, output, _, fake = run_until_interrupt_or_end(
        "feature_drift",
        failure_mode="invalid_always",
        thread_suffix="fallback-invalid",
    )
    assert "__interrupt__" in output
    assert output["revision_count"] == 1
    assert output["verification"]["status"] == "fallback"
    assert fake.recommendation_call_count == 2
    assert any(item.get("fallback_used") for item in output["llm_call_metadata"])


def test_api_failure_triggers_fallback() -> None:
    _, output, _, _ = run_until_interrupt_or_end(
        "performance_degradation",
        failure_mode="api_failure",
        thread_suffix="fallback-api",
    )
    assert "__interrupt__" in output
    assert output["verification"]["status"] == "fallback"
    assert any(item.get("fallback_used") for item in output["llm_call_metadata"])
    assert output["execution_errors"]
