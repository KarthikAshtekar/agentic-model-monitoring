"""Deterministic JSON and Markdown monitoring reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monitoring_agent.monitoring.evidence import SEVERITY_RANK
from monitoring_agent.monitoring.schemas import EvidenceItem, MonitoringRunResult
from monitoring_agent.paths import GENERATED_REPORTS_DIR, PROJECT_ROOT


def _report_locations(result: MonitoringRunResult) -> tuple[Path, Path]:
    report_dir = GENERATED_REPORTS_DIR / result.scenario_name
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "monitoring_result.json"
    markdown_path = report_dir / "monitoring_report.md"
    result.report_paths = [
        json_path.relative_to(PROJECT_ROOT).as_posix(),
        markdown_path.relative_to(PROJECT_ROOT).as_posix(),
    ]
    return json_path, markdown_path


def _ranked_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    status_rank = {"critical": 3, "warning": 2, "not_evaluated": 1, "pass": 0}
    return sorted(
        items,
        key=lambda item: (
            -SEVERITY_RANK[item.severity],
            -status_rank[item.status],
            item.evidence_id,
        ),
    )


def _display(value: Any, limit: int = 120) -> str:
    text = json.dumps(value, default=str, sort_keys=True)
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def write_json_report(result: MonitoringRunResult) -> Path:
    """Write the complete validated monitoring result as JSON."""
    json_path, _ = _report_locations(result)
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(result.model_dump_json(indent=2))
        handle.write("\n")
    return json_path


def write_markdown_report(result: MonitoringRunResult) -> Path:
    """Write a concise evidence-first Markdown monitoring report."""
    _, markdown_path = _report_locations(result)
    evidence = _ranked_evidence(result.evidence)
    critical_count = sum(item.status == "critical" for item in evidence)
    warning_count = sum(item.status == "warning" for item in evidence)

    drift_rows = sorted(
        result.drift.feature_metrics.items(),
        key=lambda item: float(item[1].get("psi", 0.0)),
        reverse=True,
    )[:10]
    drift_table = "\n".join(
        f"| `{feature}` | {metrics.get('feature_class')} | "
        f"{float(metrics.get('psi', 0.0)):.4f} | {metrics.get('primary_status')} |"
        for feature, metrics in drift_rows
    )
    if not drift_table:
        drift_table = "| _Not evaluated_ | - | - | - |"

    dq_findings = [
        item
        for item in evidence
        if item.domain == "data_quality" and item.status != "pass"
    ]
    dq_lines = (
        "\n".join(f"- `{item.evidence_id}`: {item.message}" for item in dq_findings)
        or "- No warning or critical data-quality findings."
    )

    prediction_findings = [
        item for item in evidence if item.domain == "prediction_drift"
    ]
    prediction_lines = (
        "\n".join(
            f"- `{item.evidence_id}` ({item.status}): {item.message}"
            for item in prediction_findings
        )
        or "- Not evaluated because inference was blocked."
    )

    if result.performance.evaluated:
        operating = result.performance.metrics_at_operating_threshold
        performance_lines = "\n".join(
            [
                f"- Operating-threshold accuracy: `{operating['accuracy']:.4f}`",
                f"- Operating-threshold precision: `{operating['precision']:.4f}`",
                f"- Operating-threshold recall: `{operating['recall']:.4f}`",
                f"- Operating-threshold F1: `{operating['f1']:.4f}`",
                f"- ROC-AUC: `{operating['roc_auc']:.4f}`",
                f"- PR-AUC: `{operating['pr_auc']:.4f}`",
                f"- Brier score: `{operating['brier_score']:.4f}`",
            ]
        )
    else:
        performance_lines = f"- Not evaluated: {result.performance.reason_not_evaluated}"

    evidence_table = "\n".join(
        "| `{}` | {} | {} | {} | `{}` | {} |".format(
            item.evidence_id,
            item.domain,
            item.status,
            item.severity,
            item.feature or "-",
            _display(item.observed_value),
        )
        for item in evidence
    )

    content = f"""# Monitoring report: {result.scenario_name}

## Run summary

- Run ID: `{result.run_id}`
- Created: `{result.created_at_utc.isoformat()}`
- Model: `{result.model_name}` version `{result.model_version}`
- Evidence records: `{len(result.evidence)}` (`{critical_count}` critical,
  `{warning_count}` warning)

## Batch validity

- Valid: `{str(result.batch_valid).lower()}`
- Blocked: `{str(result.batch_blocked).lower()}`
- Rows: `{result.data_quality.row_count}`
- Columns: `{result.data_quality.column_count}`
- Labels available: `{str(result.labels_available).lower()}`
- Feature row count: `{result.feature_row_count}`
- Labelled row count: `{result.labelled_row_count}`
- Label coverage rate: `{result.label_coverage_rate:.1%}`
- Minimum labelled sample size: `{result.minimum_labelled_sample_size}`
- Performance evaluation: `{"evaluated" if result.performance.evaluated else "not evaluated"}`
- Reason not evaluated: `{result.performance.reason_not_evaluated or "not applicable"}`

## Incident candidates

{", ".join(f"`{candidate}`" for candidate in result.incident_candidates)}

## Overall severity

`{result.overall_severity}`

## Data-quality findings

{dq_lines}

## Top feature-drift findings

| Feature | Class | PSI | Primary status |
|---|---|---:|---|
{drift_table}

## Prediction-drift findings

{prediction_lines}

## Performance findings

{performance_lines}

## Evidence table

| Evidence ID | Domain | Status | Severity | Feature | Observed |
|---|---|---|---|---|---|
{evidence_table}

## Limitations

- This is deterministic replay evidence, not a live-production observation.
- Observed reference ranges are monitoring baselines, not universal validity constraints.
- PSI and KS are distribution diagnostics and do not establish causality.
- The performance-degradation scenario uses synthetic label/outcome drift.
- No LLM recommendation, LangGraph investigation, retraining, or automated action is included.
"""
    markdown_path.write_text(content, encoding="utf-8", newline="\n")
    return markdown_path
