"""Evaluate completed core or extended live Groq agent runs deterministically."""

from __future__ import annotations

import argparse
import json

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from monitoring_agent.evaluation.evaluator import (
    CORE_SCENARIOS,
    EXTENDED_SCENARIOS,
    evaluate_live_runs,
)
from monitoring_agent.evaluation.reporting import (
    EXTENDED_LIVE_EVALUATION_DIR,
    LIVE_EVALUATION_DIR,
    readiness_verdict,
    relative_output_paths,
    write_live_evaluation_outputs,
)
from monitoring_agent.monitoring.schemas import MonitoringRunResult
from monitoring_agent.paths import GENERATED_REPORTS_DIR

EVALUATION_FIXES = [
    "Pre-live grounding audit removed synthetic citable SYSTEM IDs; run-level batch and "
    "label facts remain in structured case context, while citations now resolve only to "
    "the authoritative monitoring evidence registry.",
    "Agentic report paths were separated into fake/ and live_groq/ with explicit provider, "
    "model, execution-mode, fake-LLM, run, thread, and UTC provenance fields.",
    "Initial non-normal recommendation parsing could bypass the bounded verifier because "
    "semantic root-cause checks ran inside Pydantic parsing. Those checks now run in the "
    "deterministic verifier, where one revision remains available.",
    "Initial non-normal packets included ten irrelevant pass records and exceeded the "
    "provider's token-per-minute budget across sequential triage and recommendation calls. "
    "Pass padding is now limited to normal-operation stability evidence.",
    "A live feature-drift approval display initially failed on a non-breaking hyphen under "
    "the Windows console encoding. Approval payloads are now rendered as ASCII-safe escaped "
    "JSON without changing the structured payload.",
    "Initial data-quality and performance triage attempts were rate-limited and safely "
    "fell back; their targeted reruns were retained as the final live evaluation outputs.",
    "Robustness-extension offline validation found that the insufficient-label fallback "
    "uncertainty was too generic for the explicit evidence-sufficiency rule; it now states "
    "the label-coverage limitation.",
    "A legacy byte-reproducibility test overwrote the reviewed normal-operation scenario "
    "directory; regeneration now runs only in an isolated temporary test directory.",
    "The unlabelled-drift manifest initially retained a hard-coded PAY_0 link description "
    "from the shared transformation helper; its provenance now correctly records PAY_2. "
    "The transformation data and monitoring evidence were unchanged.",
    "The two new live Groq runs exposed no implementation defect and both passed structured "
    "parsing, grounding, policy, and first-pass verification without fallback.",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate preserved live Groq scenario reports."
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Evaluate all six scenarios without overwriting the four-scenario baseline.",
    )
    return parser


def _load_runs(
    scenario_names: tuple[str, ...],
) -> list[tuple[dict, MonitoringRunResult]]:
    runs: list[tuple[dict, MonitoringRunResult]] = []
    for scenario_name in scenario_names:
        scenario_dir = GENERATED_REPORTS_DIR / scenario_name
        agent_path = scenario_dir / "live_groq/agentic_result.json"
        monitoring_path = scenario_dir / "monitoring_result.json"
        missing = [
            str(path)
            for path in (agent_path, monitoring_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Required evaluation file(s) missing: " + ", ".join(missing)
            )
        agent = json.loads(agent_path.read_text(encoding="utf-8"))
        monitoring = MonitoringRunResult.model_validate_json(
            monitoring_path.read_text(encoding="utf-8")
        )
        runs.append((agent, monitoring))
    return runs


def main() -> int:
    """Generate deterministic evaluation artifacts and one compact summary."""
    args = _parser().parse_args()
    console = Console(width=180)
    scenario_names = EXTENDED_SCENARIOS if args.extended else CORE_SCENARIOS
    output_directory = (
        EXTENDED_LIVE_EVALUATION_DIR if args.extended else LIVE_EVALUATION_DIR
    )
    try:
        summary = evaluate_live_runs(_load_runs(scenario_names))
        paths = write_live_evaluation_outputs(
            summary,
            defects=EVALUATION_FIXES,
            output_directory=output_directory,
        )
    except (
        FileNotFoundError,
        ValueError,
        ValidationError,
        json.JSONDecodeError,
    ) as exc:
        console.print(f"[red]Live evaluation could not run:[/red] {exc}")
        return 2

    table = Table(show_header=True, header_style="bold")
    for column in (
        "scenario",
        "incident",
        "route",
        "structured",
        "grounded",
        "policy",
        "revisions",
        "fallback",
        "approval",
        "latency_ms",
    ):
        table.add_column(column)
    for result in summary.scenario_results:
        grounded = (
            result.all_cited_evidence_valid
            and result.all_claims_cited
            and result.all_actions_cited
        )
        table.add_row(
            result.scenario_name,
            result.incident_type,
            result.diagnostic_route,
            str(result.triage_parse_success and result.recommendation_parse_success),
            str(grounded),
            str(result.policy_compliant),
            str(result.revision_count),
            str(result.used_fallback),
            str(result.approval_completed),
            str(result.latency_ms),
        )
    console.print(table)
    console.print(
        f"Verdict: {readiness_verdict(summary)} | "
        f"Structured: {summary.structured_output_success_rate:.1%} | "
        f"Grounding: {summary.evidence_grounding_rate:.1%} | "
        f"Policy: {summary.policy_compliance_rate:.1%}"
    )
    console.print("Outputs: " + ", ".join(relative_output_paths(paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
