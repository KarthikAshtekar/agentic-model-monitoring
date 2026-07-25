"""JSON and Markdown reporting for completed agentic monitoring runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from monitoring_agent.paths import GENERATED_REPORTS_DIR, PROJECT_ROOT


def _locations(scenario_name: str) -> tuple[Path, Path]:
    report_dir = GENERATED_REPORTS_DIR / scenario_name
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
    json_path, markdown_path = _locations(scenario_name)
    payload = {
        "run_id": state["run_id"],
        "thread_id": state["thread_id"],
        "scenario_name": scenario_name,
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
        "llm_call_metadata": state.get("llm_call_metadata", []),
        "execution_errors": state.get("execution_errors", []),
        "final_status": state["final_status"],
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
    content = f"""# Agentic monitoring report: {scenario_name}

## Scenario and run

- Run ID: `{state["run_id"]}`
- Thread ID: `{state["thread_id"]}`
- Final status: `{state["final_status"]}`
- Deterministic candidates: {", ".join(
        f"`{item}`" for item in state["monitoring_result"]["incident_candidates"]
    )}

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
- The in-memory checkpointer is suitable for this MVP, not durable production recovery.
- Recommendations require external human execution; this workflow performs no remediation.
"""
    markdown_path.write_text(content, encoding="utf-8", newline="\n")
    return [
        json_path.relative_to(PROJECT_ROOT).as_posix(),
        markdown_path.relative_to(PROJECT_ROOT).as_posix(),
    ]
