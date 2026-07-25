"""Compare two isolated diabetes reliability reruns with preserved first runs."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from monitoring_agent.evaluation.evaluator import evaluate_scenario
from monitoring_agent.evaluation.schemas import ScenarioEvaluation
from monitoring_agent.paths import EVALUATIONS_DIR, GENERATED_REPORTS_DIR, PROJECT_ROOT

MODEL_ID = "diabetes_risk"
PROVIDER = "groq"
PROVIDER_MODEL = "openai/gpt-oss-20b"
RUN_LABEL = "reliability_rerun_01"
SCENARIOS = ("feature_drift", "unlabelled_drift")
OUTPUT_DIR = EVALUATIONS_DIR / MODEL_ID / "reliability_rerun"
HASH_MANIFEST = OUTPUT_DIR / "original_artifact_hashes.json"


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _original_path(scenario: str) -> Path:
    return (
        GENERATED_REPORTS_DIR
        / MODEL_ID
        / scenario
        / "live_groq"
        / "agentic_result.json"
    )


def _repeat_path(scenario: str) -> Path:
    return (
        GENERATED_REPORTS_DIR
        / MODEL_ID
        / RUN_LABEL
        / scenario
        / "agentic_result.json"
    )


def _monitoring_path(scenario: str) -> Path:
    return GENERATED_REPORTS_DIR / MODEL_ID / scenario / "monitoring_result.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required reliability input missing: {_relative(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_original_hashes() -> list[dict[str, Any]]:
    """Fail before comparison if any authoritative first-run artifact changed."""
    manifest = _load_json(HASH_MANIFEST)
    verified: list[dict[str, Any]] = []
    for record in manifest["artifacts"]:
        path = PROJECT_ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Original artifact from hash manifest is missing: {record['path']}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size_bytes = path.stat().st_size
        if digest != record["sha256"] or size_bytes != record["size_bytes"]:
            raise RuntimeError(
                f"Original artifact changed after its hash was recorded: {record['path']}"
            )
        verified.append(
            {
                "path": record["path"],
                "sha256": digest,
                "size_bytes": size_bytes,
                "verified": True,
            }
        )
    return verified


def _structured(result: ScenarioEvaluation) -> bool:
    return result.triage_parse_success and result.recommendation_parse_success


def _grounded(result: ScenarioEvaluation) -> bool:
    return (
        result.all_cited_evidence_valid
        and result.all_claims_cited
        and result.all_actions_cited
    )


def _optional_sum(values: list[int | None]) -> int | None:
    available = [value for value in values if value is not None]
    return sum(available) if available else None


def _provider_errors(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "stage": str(item.get("stage", "")),
            "error_type": str(item.get("error_type", "")),
            "error_message": str(item.get("error_message", "")),
        }
        for item in payload.get("execution_errors", [])
        if item.get("error_type") in {"RateLimitError", "APIConnectionError"}
        or "HTTP 429" in str(item.get("error_message", ""))
    ]


def _output_verification_errors(payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "stage": str(item.get("stage", "")),
            "error_type": str(item.get("error_type", "")),
            "error_message": str(item.get("error_message", "")),
        }
        for item in payload.get("execution_errors", [])
        if item.get("error_type") == "TriageVerificationError"
    ]


def _classify(
    scenario: str,
    repeat_payload: dict[str, Any],
    repeat_evaluation: ScenarioEvaluation,
) -> str:
    provider_reproduced = bool(_provider_errors(repeat_payload))
    output_reproduced = bool(_output_verification_errors(repeat_payload))
    if scenario == "feature_drift":
        if provider_reproduced:
            return "provider_failure_reproduced"
        if _structured(repeat_evaluation) and not repeat_evaluation.used_fallback:
            return "provider_failure_not_reproduced"
        return "mixed_repeat_outcome"
    if provider_reproduced and output_reproduced:
        return "mixed_repeat_outcome"
    if provider_reproduced:
        return "provider_failure_reproduced"
    if output_reproduced:
        return "model_output_failure_reproduced"
    if (
        _structured(repeat_evaluation)
        and not repeat_evaluation.used_fallback
        and repeat_evaluation.revision_count == 0
    ):
        return "successful_repeat"
    return "mixed_repeat_outcome"


def _evaluation_fields(result: ScenarioEvaluation) -> dict[str, Any]:
    return {
        "structured_output_success": _structured(result),
        "incident_type": result.incident_type,
        "incident_compatible": result.incident_compatible,
        "diagnostic_route": result.diagnostic_route,
        "route_compatible": result.route_compatible,
        "evidence_grounding": _grounded(result),
        "policy_compliant": result.policy_compliant,
        "first_pass_verification": result.first_pass_verification,
        "final_verification_passed": result.final_verification_passed,
        "revision_count": result.revision_count,
        "fallback_used": result.used_fallback,
        "approval_completed": result.approval_completed,
        "llm_latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
        "violations": result.violations,
        "notes": result.notes,
    }


def _scenario_comparison(scenario: str) -> dict[str, Any]:
    original_path = _original_path(scenario)
    repeat_path = _repeat_path(scenario)
    monitoring = _load_json(_monitoring_path(scenario))
    original_payload = _load_json(original_path)
    repeat_payload = _load_json(repeat_path)
    original_evaluation = evaluate_scenario(original_payload, monitoring)
    repeat_evaluation = evaluate_scenario(repeat_payload, monitoring)
    repeat_provider_errors = _provider_errors(repeat_payload)
    repeat_output_errors = _output_verification_errors(repeat_payload)
    original_issue = (
        "Groq recommendation HTTP 429"
        if scenario == "feature_drift"
        else (
            "Incompatible initial triage rejected by deterministic verification; "
            "subsequent Groq recommendation HTTP 429"
        )
    )
    return {
        "evaluation_type": "repeat_reliability_run",
        "replaces_original_evaluation": False,
        "original_result_preserved": True,
        "model_id": MODEL_ID,
        "scenario": scenario,
        "provider": str(repeat_payload["provider"]),
        "provider_model": str(repeat_payload["model"]),
        "execution_timestamp": str(repeat_payload["created_at_utc"]),
        "original_result_path": _relative(original_path),
        "repeat_result_path": _relative(repeat_path),
        "original_issue": original_issue,
        "classification": _classify(scenario, repeat_payload, repeat_evaluation),
        "original_provider_failure": bool(_provider_errors(original_payload)),
        "repeat_provider_failure": bool(repeat_provider_errors),
        "original_model_output_failure": bool(
            _output_verification_errors(original_payload)
        ),
        "repeat_model_output_failure": bool(repeat_output_errors),
        "repeat_revision_required": repeat_evaluation.revision_count > 0,
        "repeat_provider_error": repeat_provider_errors,
        "repeat_output_verification_error": repeat_output_errors,
        "original": _evaluation_fields(original_evaluation),
        "repeat": _evaluation_fields(repeat_evaluation),
    }


def build_summary() -> dict[str, Any]:
    verified_hashes = verify_original_hashes()
    results = [_scenario_comparison(scenario) for scenario in SCENARIOS]
    execution_timestamp = datetime.now(UTC).isoformat()
    repeat_evaluations = [item["repeat"] for item in results]

    def rate(field: str) -> float:
        return round(
            sum(bool(item[field]) for item in repeat_evaluations)
            / len(repeat_evaluations),
            4,
        )

    latencies = [
        float(item["llm_latency_ms"])
        for item in repeat_evaluations
        if item["llm_latency_ms"] is not None
    ]
    return {
        "evaluation_type": "repeat_reliability_run",
        "replaces_original_evaluation": False,
        "original_result_preserved": True,
        "model_id": MODEL_ID,
        "scenario": list(SCENARIOS),
        "provider": PROVIDER,
        "provider_model": PROVIDER_MODEL,
        "execution_timestamp": execution_timestamp,
        "original_result_path": [
            _relative(_original_path(scenario)) for scenario in SCENARIOS
        ],
        "repeat_result_path": [
            _relative(_repeat_path(scenario)) for scenario in SCENARIOS
        ],
        "original_hashes_verified": True,
        "verified_original_artifacts": verified_hashes,
        "repeat_scenario_count": len(results),
        "repeat_structured_output_success_rate": rate(
            "structured_output_success"
        ),
        "repeat_incident_compatibility_rate": rate("incident_compatible"),
        "repeat_route_compatibility_rate": rate("route_compatible"),
        "repeat_evidence_grounding_rate": rate("evidence_grounding"),
        "repeat_policy_compliance_rate": rate("policy_compliant"),
        "repeat_first_pass_verification_rate": rate("first_pass_verification"),
        "repeat_fallback_rate": rate("fallback_used"),
        "repeat_approval_completion_rate": rate("approval_completed"),
        "mean_repeat_latency_ms": (
            round(sum(latencies) / len(latencies), 2) if latencies else None
        ),
        "total_repeat_input_tokens": _optional_sum(
            [item["input_tokens"] for item in repeat_evaluations]
        ),
        "total_repeat_output_tokens": _optional_sum(
            [item["output_tokens"] for item in repeat_evaluations]
        ),
        "total_repeat_tokens": _optional_sum(
            [item["total_tokens"] for item in repeat_evaluations]
        ),
        "original_headline_metrics_unchanged": {
            "diabetes_completed": "6/6",
            "diabetes_structured_output_success": 0.6667,
            "diabetes_fallback_rate": 0.3333,
            "cross_model_completed": "12/12",
            "cross_model_structured_output_success": 0.8333,
            "cross_model_fallback_rate": 0.1666,
        },
        "scenario_results": results,
        "limitations": [
            "This is a two-case repeat reliability check, not a replacement evaluation.",
            "The original 12-case first-run evaluation remains authoritative.",
            "A successful repeat does not erase an original provider or output failure.",
            "Two repeat calls do not estimate general or production reliability.",
            "No prompts, evidence, routing, verification, revision, or approval policies changed.",
        ],
    }


def write_outputs(summary: dict[str, Any]) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "reliability_summary.json"
    markdown_path = OUTPUT_DIR / "reliability_report.md"
    csv_path = OUTPUT_DIR / "scenario_comparison.csv"
    json_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    fieldnames = [
        "evaluation_type",
        "replaces_original_evaluation",
        "original_result_preserved",
        "model_id",
        "scenario",
        "provider",
        "provider_model",
        "execution_timestamp",
        "original_result_path",
        "repeat_result_path",
        "original_issue",
        "classification",
        "original_structured_output_success",
        "repeat_structured_output_success",
        "repeat_incident_compatible",
        "repeat_route_compatible",
        "repeat_evidence_grounding",
        "repeat_policy_compliant",
        "repeat_first_pass_verification",
        "repeat_revision_count",
        "repeat_fallback_used",
        "repeat_approval_completed",
        "repeat_llm_latency_ms",
        "repeat_input_tokens",
        "repeat_output_tokens",
        "repeat_total_tokens",
        "repeat_provider_failure",
        "repeat_model_output_failure",
        "repeat_revision_required",
        "repeat_provider_error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in summary["scenario_results"]:
            writer.writerow(
                {
                    "evaluation_type": item["evaluation_type"],
                    "replaces_original_evaluation": item[
                        "replaces_original_evaluation"
                    ],
                    "original_result_preserved": item[
                        "original_result_preserved"
                    ],
                    "model_id": item["model_id"],
                    "scenario": item["scenario"],
                    "provider": item["provider"],
                    "provider_model": item["provider_model"],
                    "execution_timestamp": item["execution_timestamp"],
                    "original_result_path": item["original_result_path"],
                    "repeat_result_path": item["repeat_result_path"],
                    "original_issue": item["original_issue"],
                    "classification": item["classification"],
                    "original_structured_output_success": item["original"][
                        "structured_output_success"
                    ],
                    "repeat_structured_output_success": item["repeat"][
                        "structured_output_success"
                    ],
                    "repeat_incident_compatible": item["repeat"][
                        "incident_compatible"
                    ],
                    "repeat_route_compatible": item["repeat"][
                        "route_compatible"
                    ],
                    "repeat_evidence_grounding": item["repeat"][
                        "evidence_grounding"
                    ],
                    "repeat_policy_compliant": item["repeat"]["policy_compliant"],
                    "repeat_first_pass_verification": item["repeat"][
                        "first_pass_verification"
                    ],
                    "repeat_revision_count": item["repeat"]["revision_count"],
                    "repeat_fallback_used": item["repeat"]["fallback_used"],
                    "repeat_approval_completed": item["repeat"][
                        "approval_completed"
                    ],
                    "repeat_llm_latency_ms": item["repeat"]["llm_latency_ms"],
                    "repeat_input_tokens": item["repeat"]["input_tokens"],
                    "repeat_output_tokens": item["repeat"]["output_tokens"],
                    "repeat_total_tokens": item["repeat"]["total_tokens"],
                    "repeat_provider_failure": item["repeat_provider_failure"],
                    "repeat_model_output_failure": item[
                        "repeat_model_output_failure"
                    ],
                    "repeat_revision_required": item[
                        "repeat_revision_required"
                    ],
                    "repeat_provider_error": json.dumps(
                        item["repeat_provider_error"],
                        sort_keys=True,
                    ),
                }
            )

    rows = "\n".join(
        "| `{scenario}` | `{classification}` | {structured} | {verification} | "
        "{fallback} | {provider_failure} | {output_failure} | {latency} |".format(
            scenario=item["scenario"],
            classification=item["classification"],
            structured="yes"
            if item["repeat"]["structured_output_success"]
            else "no",
            verification="yes"
            if item["repeat"]["final_verification_passed"]
            else "no",
            fallback="yes" if item["repeat"]["fallback_used"] else "no",
            provider_failure="yes" if item["repeat_provider_failure"] else "no",
            output_failure="yes"
            if item["repeat_model_output_failure"]
            else "no",
            latency=item["repeat"]["llm_latency_ms"],
        )
        for item in summary["scenario_results"]
    )
    details = "\n".join(
        f"""### `{item["scenario"]}`

- Original issue: {item["original_issue"]}.
- Classification: `{item["classification"]}`.
- Original provider failure reproduced: `{str(item["repeat_provider_failure"]).lower()}`.
- Original model-output verification failure reproduced: `{str(item["repeat_model_output_failure"]).lower()}`.
- Different repeat recommendation revision required: `{str(item["repeat_revision_required"]).lower()}`.
- Repeat incident/route: `{item["repeat"]["incident_type"]}` /
  `{item["repeat"]["diagnostic_route"]}`.
- Repeat grounding/policy/approval: `{str(item["repeat"]["evidence_grounding"]).lower()}` /
  `{str(item["repeat"]["policy_compliant"]).lower()}` /
  `{str(item["repeat"]["approval_completed"]).lower()}`.
- Repeat revisions/fallback: `{item["repeat"]["revision_count"]}` /
  `{str(item["repeat"]["fallback_used"]).lower()}`.
- Repeat tokens: `{item["repeat"]["input_tokens"]}` input /
  `{item["repeat"]["output_tokens"]}` output / `{item["repeat"]["total_tokens"]}` total.
- Original result: `{item["original_result_path"]}`.
- Repeat result: `{item["repeat_result_path"]}`.
"""
        for item in summary["scenario_results"]
    )
    limitations = "\n".join(f"- {item}" for item in summary["limitations"])
    markdown_path.write_text(
        f"""# Repeat reliability check

- Evaluation type: `repeat_reliability_run`
- Replaces original evaluation: `false`
- Original result preserved: `true`
- Model ID: `{summary["model_id"]}`
- Scenario: `{", ".join(summary["scenario"])}`
- Provider: `{summary["provider"]}`
- Provider model: `{summary["provider_model"]}`
- Execution timestamp: `{summary["execution_timestamp"]}`
- Original result path: `{", ".join(summary["original_result_path"])}`
- Repeat result path: `{", ".join(summary["repeat_result_path"])}`
- Original hashes verified: `true`

## Objective

Determine whether the two first-run diabetes fallbacks reproduced under the same model
bundle, scenario inputs, prompts, evidence, schemas, and deterministic policies. This is
supplementary to, and does not replace, the authoritative first-run evaluation.

## Repeat results

| Scenario | Classification | Structured | Final verifier | Fallback | Provider error | Output error | LLM latency ms |
|---|---|---:|---:|---:|---:|---:|---:|
{rows}

{details}
## Aggregate repeat-only metrics

- Structured-output success: `{summary["repeat_structured_output_success_rate"]:.1%}`
- Incident compatibility: `{summary["repeat_incident_compatibility_rate"]:.1%}`
- Route compatibility: `{summary["repeat_route_compatibility_rate"]:.1%}`
- Evidence grounding: `{summary["repeat_evidence_grounding_rate"]:.1%}`
- Policy compliance: `{summary["repeat_policy_compliance_rate"]:.1%}`
- First-pass verification: `{summary["repeat_first_pass_verification_rate"]:.1%}`
- Fallback: `{summary["repeat_fallback_rate"]:.1%}`
- Approval completion: `{summary["repeat_approval_completion_rate"]:.1%}`
- Mean repeat LLM latency: `{summary["mean_repeat_latency_ms"]}` ms
- Repeat tokens: `{summary["total_repeat_input_tokens"]}` input /
  `{summary["total_repeat_output_tokens"]}` output /
  `{summary["total_repeat_tokens"]}` total

## Authoritative metrics remain unchanged

The original diabetes evaluation remains `6/6` complete with `66.67%` structured-output
success and `33.33%` fallback. The original cross-model evaluation remains `12/12`
complete with `83.33%` structured-output success and `16.66%` fallback. These first-run
figures remain the headline evaluation and CV metrics.

## Interpretation

The deterministic fallback is part of the intended safety architecture. A successful
repeat does not erase the first-run failure, and a failed repeat is not automatically an
implementation defect. No universal or production reliability claim is made.

## Limitations

{limitations}
""",
        encoding="utf-8",
        newline="\n",
    )
    return [json_path, markdown_path, csv_path]


def main() -> int:
    try:
        summary = build_summary()
        paths = write_outputs(summary)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"Reliability evaluation could not run: {exc}")
        return 2
    print("Repeat reliability check complete:")
    for item in summary["scenario_results"]:
        print(
            f"  {item['scenario']}: {item['classification']} | "
            f"structured={item['repeat']['structured_output_success']} | "
            f"fallback={item['repeat']['fallback_used']}"
        )
    for path in paths:
        print(f"  {_relative(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
