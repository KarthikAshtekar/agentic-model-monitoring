"""JSON, Markdown, and CSV outputs for live-agent evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

from monitoring_agent.evaluation.schemas import LiveEvaluationSummary
from monitoring_agent.paths import EVALUATIONS_DIR, PROJECT_ROOT

LIVE_EVALUATION_DIR = EVALUATIONS_DIR / "live_groq"
EXTENDED_LIVE_EVALUATION_DIR = EVALUATIONS_DIR / "live_groq_six_scenarios"


def registered_live_evaluation_dir(model_id: str) -> Path:
    return EVALUATIONS_DIR / model_id / "live_groq_six_scenarios"


def readiness_verdict(summary: LiveEvaluationSummary) -> str:
    """Return a factual non-production verdict from deterministic metrics."""
    if summary.scenario_count < 4 or summary.completed_count < summary.scenario_count:
        return "live_evaluation_incomplete"
    required_rates = (
        summary.structured_output_success_rate,
        summary.incident_compatibility_rate,
        summary.route_compatibility_rate,
        summary.evidence_grounding_rate,
        summary.policy_compliance_rate,
        summary.approval_completion_rate,
    )
    if all(rate == 1.0 for rate in required_rates):
        return "live_agent_validated_with_limitations"
    return "live_agent_requires_revision"


def _percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_live_evaluation_outputs(
    summary: LiveEvaluationSummary,
    *,
    defects: list[str],
    output_directory: Path = LIVE_EVALUATION_DIR,
) -> list[Path]:
    """Write all required evaluation artifacts from one validated summary."""
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "live_evaluation_summary.json"
    markdown_path = output_directory / "live_evaluation_report.md"
    csv_path = output_directory / "live_scenario_comparison.csv"

    json_path.write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "scenario_name",
            "completed",
            "incident_type",
            "incident_compatible",
            "diagnostic_route",
            "route_compatible",
            "triage_parse_success",
            "recommendation_parse_success",
            "first_pass_verification",
            "revision_count",
            "used_fallback",
            "all_cited_evidence_valid",
            "all_claims_cited",
            "all_actions_cited",
            "policy_compliant",
            "approval_completed",
            "final_status",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "violations",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in summary.scenario_results:
            row = result.model_dump()
            writer.writerow(
                {
                    key: (
                        " | ".join(row[key])
                        if key == "violations"
                        else row.get(key)
                    )
                    for key in fieldnames
                }
            )

    scenario_rows = "\n".join(
        "| {} | `{}` | `{}` | {} | {} | {} | {} | `{}` |".format(
            result.scenario_name,
            result.incident_type,
            result.diagnostic_route,
            "yes" if result.triage_parse_success and result.recommendation_parse_success else "no",
            "yes"
            if (
                result.all_cited_evidence_valid
                and result.all_claims_cited
                and result.all_actions_cited
            )
            else "no",
            "yes" if result.policy_compliant else "no",
            "yes" if result.used_fallback else "no",
            result.final_status,
        )
        for result in summary.scenario_results
    )
    revision_lines = "\n".join(
        f"- `{item.scenario_name}`: `{item.revision_count}` revision(s), "
        f"final verifier pass `{str(item.final_verification_passed).lower()}`."
        for item in summary.scenario_results
    )
    approval_lines = "\n".join(
        f"- `{item.scenario_name}`: required `{str(item.approval_required).lower()}`, "
        f"completed `{str(item.approval_completed).lower()}`, status `{item.final_status}`."
        for item in summary.scenario_results
    )
    latency_lines = "\n".join(
        f"- `{item.scenario_name}`: `{item.latency_ms}` ms; tokens "
        f"`{item.input_tokens}` input / `{item.output_tokens}` output / "
        f"`{item.total_tokens}` total."
        for item in summary.scenario_results
    )
    scenario_note_lines = "\n".join(
        f"- `{item.scenario_name}`: "
        + " ".join(
            note
            for note in item.notes
            if note.startswith("Live execution error preserved")
        )
        for item in summary.scenario_results
        if any(
            note.startswith("Live execution error preserved")
            for note in item.notes
        )
    ) or "- No live execution errors were recorded."
    defect_lines = "\n".join(f"- {item}" for item in defects) or "- None."
    limitation_lines = "\n".join(f"- {item}" for item in summary.limitations)
    verdict = readiness_verdict(summary)
    evaluation_label = (
        "Extended live Groq robustness evaluation"
        if summary.scenario_count == 6
        else "Live Groq agent evaluation"
    )
    preservation_note = (
        "\n- Original core evaluation: `4` scenarios, preserved under "
        "`reports/evaluations/live_groq/`.\n"
        "- Extended robustness evaluation: `6` scenarios."
        if summary.scenario_count == 6 and summary.model_id == "credit_default"
        else ""
    )
    content = f"""# {evaluation_label}

## Evaluation objective

Evaluate the existing one-orchestrator monitoring agent across {summary.scenario_count}
controlled replay
scenarios using deterministic structured-output, routing, grounding, policy, verification,
fallback, approval, latency, and token checks. No LLM-as-judge was used.

## Provider and model

- Provider: `{summary.provider}`
- Model: `{summary.model}`
- Registered model: `{summary.model_id}`
- Domain: `{summary.domain_id}`
- Execution date: `{summary.created_at_utc.isoformat()}`
- Scenarios completed: `{summary.completed_count}/{summary.scenario_count}`
{preservation_note}

## Scenario-level results

| Scenario | Incident | Route | Structured | Grounded | Policy | Fallback | Final status |
|---|---|---|---|---|---|---|---|
{scenario_rows}

## Structured-output success

`{_percentage(summary.structured_output_success_rate)}`. A fallback is explicitly excluded
from live structured-output success.

## Incident and routing compatibility

- Incident compatibility: `{_percentage(summary.incident_compatibility_rate)}`
- Route compatibility: `{_percentage(summary.route_compatibility_rate)}`

## Evidence grounding

`{_percentage(summary.evidence_grounding_rate)}` of scenarios had only authoritative
deterministic evidence IDs and cited every claim and action.

## Policy compliance

`{_percentage(summary.policy_compliance_rate)}` under the deterministic hard and
scenario-specific policies.

## Verification and revision behaviour

- First-pass verification: `{_percentage(summary.first_pass_verification_rate)}`
{revision_lines}

## Fallback usage

Fallback rate: `{_percentage(summary.fallback_rate)}`.

### Preserved execution errors

{scenario_note_lines}

## Approval behaviour

Approval completion rate: `{_percentage(summary.approval_completion_rate)}`.
{approval_lines}

## Latency and token usage

- Mean scenario LLM latency: `{summary.mean_latency_ms}` ms
- Median scenario LLM latency: `{summary.median_latency_ms}` ms
- Aggregate tokens: `{summary.total_input_tokens}` input /
  `{summary.total_output_tokens}` output / `{summary.total_tokens}` total
{latency_lines}

## Concrete defects found and fixed

{defect_lines}

## Limitations

{limitation_lines}

## Final readiness verdict

`{verdict}`

This verdict is limited to controlled replay validation and is not a production-readiness
or deployment claim. No remediation was executed.
"""
    markdown_path.write_text(content, encoding="utf-8", newline="\n")
    return [json_path, markdown_path, csv_path]


def relative_output_paths(paths: list[Path]) -> list[str]:
    return [path.relative_to(PROJECT_ROOT).as_posix() for path in paths]
