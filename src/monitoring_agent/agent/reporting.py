"""JSON and Markdown reporting for completed agentic monitoring runs."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from monitoring_agent.paths import GENERATED_REPORTS_DIR, PROJECT_ROOT


def _execution_identity(
    metadata: list[dict[str, Any]],
) -> tuple[str, str, str, bool]:
    for item in metadata:
        provider = item.get("provider")
        model = item.get("model")
        if provider == "fake":
            return "fake", str(model), "fake", True
        if provider == "groq":
            return "groq", str(model), "live", False
    raise ValueError("Agentic report has no identifiable fake or live LLM provider call.")


def _locations(
    model_id: str,
    scenario_name: str,
    execution_mode: str,
    *,
    checkpoint_backend: str = "memory",
    thread_id: str | None = None,
    run_label: str | None = None,
) -> tuple[Path, Path]:
    mode_directory = "fake" if execution_mode == "fake" else "live_groq"
    report_dir = (
        GENERATED_REPORTS_DIR / model_id / run_label / scenario_name
        if run_label
        else GENERATED_REPORTS_DIR / model_id / scenario_name / mode_directory
    )
    if checkpoint_backend == "sqlite" and not run_label:
        safe_thread_id = re.sub(r"[^A-Za-z0-9._-]", "_", thread_id or "unknown-thread")
        report_dir = (
            GENERATED_REPORTS_DIR
            / model_id
            / scenario_name
            / f"{mode_directory}_persistent"
            / safe_thread_id
        )
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / "agentic_result.json", report_dir / "agentic_report.md"


def _fallback_used(state: dict[str, Any]) -> bool:
    return any(
        bool(item.get("fallback_used")) for item in state.get("llm_call_metadata", [])
    )


def _llm_summary(metadata: list[dict[str, Any]]) -> str:
    calls = [
        item
        for item in metadata
        if item.get("schema") in {"AgentTriage", "AgentRecommendation"}
    ]
    if not calls:
        return "- No successful live or fake structured-model call was recorded."
    return "\n".join(
        "- `{provider}` / `{model}` / `{schema}`: latency `{latency}` ms, "
        "parse success `{parsed}`, tokens `{tokens}`, fake `{fake}`".format(
            provider=item.get("provider"),
            model=item.get("model"),
            schema=item.get("schema"),
            latency=item.get("latency_ms"),
            parsed=item.get("parse_success"),
            tokens=json.dumps(item.get("token_usage"), sort_keys=True),
            fake=str(bool(item.get("is_fake"))).lower(),
        )
        for item in calls
    )


def write_agentic_reports(state: dict[str, Any]) -> list[str]:
    """Write agentic artifacts without altering deterministic reports."""
    scenario_name = state["scenario_name"]
    monitoring = state["monitoring_result"]
    model_id = monitoring.get("model_id", "credit_default")
    metadata = state.get("llm_call_metadata", [])
    provider, model, execution_mode, is_fake_llm = _execution_identity(metadata)
    created_at_utc = datetime.now(UTC).isoformat()
    checkpoint_backend = str(state.get("checkpoint_backend", "memory"))
    checkpoint_database = state.get("checkpoint_database")
    resumed_from_checkpoint = bool(state.get("resumed_from_checkpoint", False))
    pause_count = int(state.get("pause_count", 0))
    resume_count = int(state.get("resume_count", 0))
    run_label = state.get("run_label")
    json_path, markdown_path = _locations(
        model_id,
        scenario_name,
        execution_mode,
        checkpoint_backend=checkpoint_backend,
        thread_id=state["thread_id"],
        run_label=run_label,
    )
    original_json_path = (
        GENERATED_REPORTS_DIR
        / model_id
        / scenario_name
        / ("fake" if execution_mode == "fake" else "live_groq")
        / "agentic_result.json"
    )
    reliability_metadata = (
        {
            "evaluation_type": "repeat_reliability_run",
            "replaces_original_evaluation": False,
            "original_result_preserved": True,
            "model_id": model_id,
            "scenario": scenario_name,
            "provider": provider,
            "provider_model": model,
            "execution_timestamp": created_at_utc,
            "original_result_path": original_json_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "repeat_result_path": json_path.relative_to(PROJECT_ROOT).as_posix(),
        }
        if run_label
        else {}
    )
    payload = {
        "provider": provider,
        "model": model,
        "execution_mode": execution_mode,
        "is_fake_llm": is_fake_llm,
        "run_id": state["run_id"],
        "thread_id": state["thread_id"],
        "created_at_utc": created_at_utc,
        "checkpoint_backend": checkpoint_backend,
        "checkpoint_database": checkpoint_database,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "pause_count": pause_count,
        "resume_count": resume_count,
        "scenario_name": scenario_name,
        "model_id": model_id,
        "display_name": monitoring.get("display_name"),
        "domain_id": monitoring.get("domain_id"),
        "task_type": monitoring.get("task_type"),
        "bundle_mode": monitoring.get("bundle_mode"),
        "deterministic_incident_candidates": state["monitoring_result"][
            "incident_candidates"
        ],
        "selected_evidence": state["selected_evidence"],
        "triage": state["triage"],
        "diagnostic_context": state["diagnostic_context"],
        "recommendation": state["recommendation"],
        "verification": state["verification"],
        "revision_count": state["revision_count"],
        "fallback_used": _fallback_used(state),
        "approval_required": state["approval_required"],
        "approval_decision": state.get("approval_decision"),
        "llm_call_metadata": metadata,
        "execution_errors": state.get("execution_errors", []),
        "final_status": state["final_status"],
        **reliability_metadata,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    triage = state["triage"]
    recommendation = state["recommendation"]
    verification = state["verification"]
    claims = "\n".join(
        f"- {claim['claim']} "
        f"({', '.join(f'`{item}`' for item in claim['evidence_ids'])})"
        for claim in recommendation["claims"]
    )
    actions = "\n".join(
        "- `{}` [{}]: {} Evidence: {}. Human approval: `{}`.".format(
            action["action_type"],
            action["priority"],
            action["action"],
            ", ".join(f"`{item}`" for item in action["evidence_ids"]),
            str(action["requires_human_approval"]).lower(),
        )
        for action in recommendation["recommended_actions"]
    )
    uncertainties = (
        "\n".join(f"- {item}" for item in recommendation["uncertainties"])
        or "- None stated."
    )
    approval = state.get("approval_decision")
    approval_text = (
        "- Not required for normal operation."
        if not state["approval_required"]
        else (
            f"- Decision: `{approval['decision']}` by `{approval['reviewer']}` at "
            f"`{approval['reviewed_at_utc']}`. Comment: "
            f"{approval.get('comment') or '_None_'}"
            if approval
            else "- Pending human review."
        )
    )
    hypothesis = (
        recommendation["root_cause_hypothesis"]
        or "No detailed root-cause hypothesis was asserted."
    )
    reliability_markdown = (
        "\n".join(
            [
                "## Repeat reliability provenance",
                "",
                "- Evaluation type: `repeat_reliability_run`",
                "- Replaces original evaluation: `false`",
                "- Original result preserved: `true`",
                f"- Model ID: `{model_id}`",
                f"- Scenario: `{scenario_name}`",
                f"- Provider: `{provider}`",
                f"- Provider model: `{model}`",
                f"- Execution timestamp: `{created_at_utc}`",
                (
                    "- Original result path: "
                    f"`{reliability_metadata['original_result_path']}`"
                ),
                (
                    "- Repeat result path: "
                    f"`{reliability_metadata['repeat_result_path']}`"
                ),
                "",
            ]
        )
        if run_label
        else ""
    )
    content = f"""# Agentic monitoring report: {scenario_name}

## Execution provenance

- Provider: `{provider}`
- Model: `{model}`
- Execution mode: `{execution_mode}`
- Fake LLM: `{str(is_fake_llm).lower()}`
- Run ID: `{state["run_id"]}`
- Thread ID: `{state["thread_id"]}`
- Created at UTC: `{created_at_utc}`
- Registered model: `{model_id}`
- Model: `{monitoring.get("display_name")}`
- Domain: `{monitoring.get("domain_id")}`
- Task: `{monitoring.get("task_type")}`
- Bundle mode: `{monitoring.get("bundle_mode")}`

{reliability_markdown}
## Scenario and run

- Run ID: `{state["run_id"]}`
- Thread ID: `{state["thread_id"]}`
- Final status: `{state["final_status"]}`
- Deterministic candidates: {", ".join(
        f"`{item}`" for item in state["monitoring_result"]["incident_candidates"]
    )}

## Workflow persistence

- Backend: `{checkpoint_backend}`
- Checkpoint database: `{checkpoint_database or "not applicable"}`
- Thread ID: `{state["thread_id"]}`
- Resumed from checkpoint: `{str(resumed_from_checkpoint).lower()}`
- Pause count: `{pause_count}`
- Resume count: `{resume_count}`

## Agent triage

- Incident: `{triage["incident_type"]}`
- Severity: `{triage["severity"]}`
- Reason: {triage["reason"]}
- Evidence: {", ".join(f"`{item}`" for item in triage["selected_evidence_ids"])}

## Diagnostic route

`{triage["diagnostic_route"]}` using `{len(
        state["diagnostic_context"].get("evidence", [])
    )}` already-calculated evidence records.

## Evidence-backed claims

{claims}

## Root-cause hypothesis

{hypothesis}

## Recommended actions

{actions}

No action was automatically executed.

## Uncertainties

{uncertainties}

## Verification

- Status before finalisation: `{verification["status"]}`
- Violations: {json.dumps(verification["violations"])}
- Unsupported evidence IDs: {json.dumps(
        verification["unsupported_evidence_ids"]
    )}
- Policy checks: {", ".join(
        f"`{item}`" for item in verification["policy_checks"]
    )}

## Fallback

- Used: `{str(_fallback_used(state)).lower()}`
- Revision attempts: `{state["revision_count"]}`

## Human decision

{approval_text}

## Structured LLM calls

{_llm_summary(state.get("llm_call_metadata", []))}

## Limitations

- This is a controlled replay, not a live-production observation.
- Deterministic monitoring evidence remains authoritative; the LLM did not recalculate it.
- Evidence associations do not establish causality.
- The selected checkpoint backend supports controlled local workflows only; SQLite is
  not presented as production-grade durability.
- Recommendations require external human execution; this workflow performs no remediation.
"""
    markdown_path.write_text(content, encoding="utf-8", newline="\n")
    return [
        json_path.relative_to(PROJECT_ROOT).as_posix(),
        markdown_path.relative_to(PROJECT_ROOT).as_posix(),
    ]


def migrate_legacy_fake_reports() -> list[str]:
    """Move legacy root-level fake reports into fake/ and add provenance metadata."""
    migrated: list[str] = []
    for scenario_dir in sorted(GENERATED_REPORTS_DIR.iterdir()):
        if not scenario_dir.is_dir():
            continue
        legacy_json = scenario_dir / "agentic_result.json"
        legacy_markdown = scenario_dir / "agentic_report.md"
        if not legacy_json.is_file() or not legacy_markdown.is_file():
            continue
        payload = json.loads(legacy_json.read_text(encoding="utf-8"))
        provider, model, execution_mode, is_fake_llm = _execution_identity(
            payload.get("llm_call_metadata", [])
        )
        if execution_mode != "fake":
            raise ValueError(
                f"Refusing to migrate non-fake legacy report: {legacy_json}"
            )
        created_at_utc = datetime.fromtimestamp(
            legacy_json.stat().st_mtime,
            tz=UTC,
        ).isoformat()
        enriched = {
            "provider": provider,
            "model": model,
            "execution_mode": execution_mode,
            "is_fake_llm": is_fake_llm,
            "run_id": payload["run_id"],
            "thread_id": payload["thread_id"],
            "created_at_utc": created_at_utc,
            **{
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "provider",
                    "model",
                    "execution_mode",
                    "is_fake_llm",
                    "run_id",
                    "thread_id",
                    "created_at_utc",
                }
            },
        }
        target_json, target_markdown = _locations(
            payload.get("model_id", "credit_default"),
            payload["scenario_name"],
            execution_mode,
        )
        target_json.write_text(
            json.dumps(enriched, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        markdown = legacy_markdown.read_text(encoding="utf-8")
        title, remainder = markdown.split("\n", maxsplit=1)
        provenance = f"""

## Execution provenance

- Provider: `{provider}`
- Model: `{model}`
- Execution mode: `{execution_mode}`
- Fake LLM: `{str(is_fake_llm).lower()}`
- Run ID: `{payload["run_id"]}`
- Thread ID: `{payload["thread_id"]}`
- Created at UTC: `{created_at_utc}`
"""
        target_markdown.write_text(
            f"{title}{provenance}\n{remainder.lstrip()}",
            encoding="utf-8",
            newline="\n",
        )
        legacy_json.unlink()
        legacy_markdown.unlink()
        migrated.extend(
            [
                target_json.relative_to(PROJECT_ROOT).as_posix(),
                target_markdown.relative_to(PROJECT_ROOT).as_posix(),
            ]
        )
    return migrated
