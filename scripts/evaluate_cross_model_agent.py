"""Combine validated model-specific live summaries without rerunning evaluation."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from monitoring_agent.evaluation.schemas import LiveEvaluationSummary
from monitoring_agent.paths import EVALUATIONS_DIR, PROJECT_ROOT

CREDIT_SUMMARY = (
    EVALUATIONS_DIR
    / "live_groq_six_scenarios"
    / "live_evaluation_summary.json"
)
DIABETES_SUMMARY = (
    EVALUATIONS_DIR
    / "diabetes_risk"
    / "live_groq_six_scenarios"
    / "live_evaluation_summary.json"
)
OUTPUT_DIR = EVALUATIONS_DIR / "cross_model"


def _weighted_rate(
    summaries: list[LiveEvaluationSummary],
    field: str,
) -> float:
    count = sum(item.scenario_count for item in summaries)
    if not count:
        return 0.0
    numerator = sum(
        float(getattr(item, field)) * item.scenario_count for item in summaries
    )
    return round(numerator / count, 4)


def _optional_sum(values: list[int | None]) -> int | None:
    available = [item for item in values if item is not None]
    return sum(available) if available else None


def _revision_rate(summary: LiveEvaluationSummary) -> float:
    if not summary.scenario_count:
        return 0.0
    revised = sum(item.revision_count > 0 for item in summary.scenario_results)
    return round(revised / summary.scenario_count, 4)


def _load(path: Path) -> LiveEvaluationSummary:
    if not path.is_file():
        raise FileNotFoundError(f"Validated model summary missing: {path}")
    return LiveEvaluationSummary.model_validate_json(path.read_text(encoding="utf-8"))


def build_cross_model_summary(
    summaries: list[LiveEvaluationSummary],
) -> dict[str, Any]:
    rate_fields = (
        "structured_output_success_rate",
        "incident_compatibility_rate",
        "route_compatibility_rate",
        "evidence_grounding_rate",
        "policy_compliance_rate",
        "first_pass_verification_rate",
        "fallback_rate",
        "approval_completion_rate",
    )
    scenario_count = sum(item.scenario_count for item in summaries)
    latency_pairs = [
        (result.latency_ms, result.model_id)
        for summary in summaries
        for result in summary.scenario_results
        if result.latency_ms is not None
    ]
    latencies = [item[0] for item in latency_pairs]
    revised_count = sum(
        result.revision_count > 0
        for summary in summaries
        for result in summary.scenario_results
    )
    aggregate = {
        field: _weighted_rate(summaries, field) for field in rate_fields
    }
    return {
        "model_count": len(summaries),
        "domain_count": len({item.domain_id for item in summaries}),
        "scenario_count": scenario_count,
        "completed_count": sum(item.completed_count for item in summaries),
        **aggregate,
        "revision_rate": (
            round(revised_count / scenario_count, 4) if scenario_count else 0.0
        ),
        "mean_latency_ms": (
            round(sum(latencies) / len(latencies), 2) if latencies else None
        ),
        "total_input_tokens": _optional_sum(
            [item.total_input_tokens for item in summaries]
        ),
        "total_output_tokens": _optional_sum(
            [item.total_output_tokens for item in summaries]
        ),
        "total_tokens": _optional_sum([item.total_tokens for item in summaries]),
        "models": [
            {
                "model_id": item.model_id,
                "domain_id": item.domain_id,
                "provider": item.provider,
                "model": item.model,
                "scenario_count": item.scenario_count,
                "completed_count": item.completed_count,
                **{field: getattr(item, field) for field in rate_fields},
                "revision_rate": _revision_rate(item),
                "mean_latency_ms": item.mean_latency_ms,
                "median_latency_ms": item.median_latency_ms,
                "total_input_tokens": item.total_input_tokens,
                "total_output_tokens": item.total_output_tokens,
                "total_tokens": item.total_tokens,
            }
            for item in summaries
        ],
        "limitations": [
            "The comparison contains controlled replay scenarios, not production traffic.",
            "Credit and diabetes performance metrics are not averaged across domains.",
            "Both model evaluations use internal historical reference splits.",
            "No deployment, diagnosis, credit decision, or remediation was executed.",
        ],
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def write_outputs(
    payload: dict[str, Any],
    summaries: list[LiveEvaluationSummary],
) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "cross_model_summary.json"
    report_path = OUTPUT_DIR / "cross_model_report.md"
    csv_path = OUTPUT_DIR / "scenario_comparison.csv"
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    fieldnames = [
        "model_id",
        "domain_id",
        "scenario_name",
        "incident_type",
        "diagnostic_route",
        "completed",
        "structured_output_success",
        "incident_compatible",
        "route_compatible",
        "evidence_grounded",
        "policy_compliant",
        "first_pass_verification",
        "revision_count",
        "used_fallback",
        "approval_completed",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            for item in summary.scenario_results:
                writer.writerow(
                    {
                        "model_id": item.model_id,
                        "domain_id": item.domain_id,
                        "scenario_name": item.scenario_name,
                        "incident_type": item.incident_type,
                        "diagnostic_route": item.diagnostic_route,
                        "completed": item.completed,
                        "structured_output_success": (
                            item.triage_parse_success
                            and item.recommendation_parse_success
                        ),
                        "incident_compatible": item.incident_compatible,
                        "route_compatible": item.route_compatible,
                        "evidence_grounded": (
                            item.all_cited_evidence_valid
                            and item.all_claims_cited
                            and item.all_actions_cited
                        ),
                        "policy_compliant": item.policy_compliant,
                        "first_pass_verification": item.first_pass_verification,
                        "revision_count": item.revision_count,
                        "used_fallback": item.used_fallback,
                        "approval_completed": item.approval_completed,
                        "latency_ms": item.latency_ms,
                        "input_tokens": item.input_tokens,
                        "output_tokens": item.output_tokens,
                        "total_tokens": item.total_tokens,
                    }
                )
    model_rows = "\n".join(
        "| `{model_id}` | `{domain_id}` | {completed_count}/{scenario_count} | "
        "{structured_output_success_rate:.1%} | {evidence_grounding_rate:.1%} | "
        "{policy_compliance_rate:.1%} | {first_pass_verification_rate:.1%} | "
        "{revision_rate:.1%} | {fallback_rate:.1%} |".format(**item)
        for item in payload["models"]
    )
    limitations = "\n".join(f"- {item}" for item in payload["limitations"])
    verdict = (
        "cross_model_agent_validated_with_limitations"
        if payload["completed_count"] == payload["scenario_count"]
        and all(
            payload[field] == 1.0
            for field in (
                "structured_output_success_rate",
                "incident_compatibility_rate",
                "route_compatibility_rate",
                "evidence_grounding_rate",
                "policy_compliance_rate",
                "approval_completion_rate",
            )
        )
        else "cross_model_agent_requires_revision"
    )
    report_path.write_text(
        f"""# Cross-model live agent evaluation

This report combines already validated model-specific summaries. It does not rerun or
rejudge either model.

| Model | Domain | Complete | Structured | Grounded | Policy | First pass | Revision | Fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{model_rows}

## Combined controlled-replay metrics

- Models/domains: `{payload["model_count"]}` / `{payload["domain_count"]}`
- Scenarios completed: `{payload["completed_count"]}/{payload["scenario_count"]}`
- Structured output: `{payload["structured_output_success_rate"]:.1%}`
- Incident compatibility: `{payload["incident_compatibility_rate"]:.1%}`
- Route compatibility: `{payload["route_compatibility_rate"]:.1%}`
- Evidence grounding: `{payload["evidence_grounding_rate"]:.1%}`
- Policy compliance: `{payload["policy_compliance_rate"]:.1%}`
- First-pass verification: `{payload["first_pass_verification_rate"]:.1%}`
- Revision: `{payload["revision_rate"]:.1%}`
- Fallback: `{payload["fallback_rate"]:.1%}`
- Approval completion: `{payload["approval_completion_rate"]:.1%}`

Model-level ROC-AUC, PR-AUC, recall, or other performance metrics are deliberately not
averaged across domains.

## Limitations

{limitations}

## Verdict

`{verdict}`

This is not a production-readiness or deployment claim.
""",
        encoding="utf-8",
        newline="\n",
    )
    return [json_path, report_path, csv_path]


def main() -> int:
    summaries = [_load(CREDIT_SUMMARY), _load(DIABETES_SUMMARY)]
    payload = build_cross_model_summary(summaries)
    paths = write_outputs(payload, summaries)
    print(
        f"Combined {payload['model_count']} models and "
        f"{payload['scenario_count']} controlled scenarios."
    )
    for path in paths:
        print(f"  {path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
