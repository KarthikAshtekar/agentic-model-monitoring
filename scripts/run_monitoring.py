"""Run deterministic monitoring for one or all replay scenarios."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from monitoring_agent.monitoring.engine import MonitoringEngine
from monitoring_agent.monitoring.reporting import (
    write_json_report,
    write_markdown_report,
)
from monitoring_agent.scenarios.generator import SUPPORTED_SCENARIOS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic scenario monitoring.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Monitor all MVP scenarios.")
    selection.add_argument("--scenario", choices=SUPPORTED_SCENARIOS)
    return parser


def main() -> int:
    """Run monitoring and report execution status without failing on intended incidents."""
    args = _parser().parse_args()
    scenario_names = list(SUPPORTED_SCENARIOS) if args.all else [args.scenario]
    engine = MonitoringEngine()
    table = Table(show_header=True, header_style="bold")
    for column in (
        "scenario",
        "batch_valid",
        "batch_blocked",
        "overall_severity",
        "incident_candidates",
        "critical_evidence_count",
        "warning_evidence_count",
        "json_report",
        "markdown_report",
    ):
        table.add_column(column)

    for scenario_name in scenario_names:
        result = engine.run_scenario(scenario_name)
        markdown_path = write_markdown_report(result)
        json_path = write_json_report(result)
        table.add_row(
            scenario_name,
            str(result.batch_valid),
            str(result.batch_blocked),
            result.overall_severity,
            ",".join(result.incident_candidates),
            str(sum(item.status == "critical" for item in result.evidence)),
            str(sum(item.status == "warning" for item in result.evidence)),
            str(json_path),
            str(markdown_path),
        )

    Console(width=500).print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
