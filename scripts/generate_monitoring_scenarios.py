"""Generate one or all deterministic monitoring replay scenarios."""

from __future__ import annotations

import argparse

from monitoring_agent.scenarios.generator import (
    SUPPORTED_SCENARIOS,
    generate_all_scenarios,
    generate_scenario,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic monitoring scenarios.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Generate every enabled scenario.")
    selection.add_argument("--scenario", choices=SUPPORTED_SCENARIOS)
    parser.add_argument("--model-id", default="credit_default")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    """Generate selected scenarios and print compact manifest summaries."""
    args = _parser().parse_args()
    manifests = (
        generate_all_scenarios(model_id=args.model_id, overwrite=args.overwrite)
        if args.all
        else [
            generate_scenario(
                args.scenario,
                model_id=args.model_id,
                overwrite=args.overwrite,
            )
        ]
    )
    print(f"Generated {len(manifests)} monitoring scenario(s).")
    for manifest in manifests:
        candidates = ",".join(manifest.expected_incident_candidates)
        print(
            f"  {manifest.model_id}/{manifest.scenario_name}: "
            f"rows={manifest.generated_sample_count}, "
            f"seed={manifest.random_seed}, expected={candidates}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
