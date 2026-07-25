"""Focused checks for isolated agentic report locations."""

from pathlib import Path

import monitoring_agent.agent.reporting as reporting


def test_run_label_isolates_reports_without_changing_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(reporting, "GENERATED_REPORTS_DIR", tmp_path)

    default_json, _ = reporting._locations(
        "diabetes_risk",
        "feature_drift",
        "live",
    )
    isolated_json, isolated_markdown = reporting._locations(
        "diabetes_risk",
        "feature_drift",
        "live",
        run_label="reliability_rerun_01",
    )

    assert default_json == (
        tmp_path
        / "diabetes_risk"
        / "feature_drift"
        / "live_groq"
        / "agentic_result.json"
    )
    assert isolated_json == (
        tmp_path
        / "diabetes_risk"
        / "reliability_rerun_01"
        / "feature_drift"
        / "agentic_result.json"
    )
    assert isolated_markdown.name == "agentic_report.md"
