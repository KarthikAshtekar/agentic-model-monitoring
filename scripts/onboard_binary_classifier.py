"""Onboard a validated fitted binary classifier without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

from monitoring_agent.onboarding.bundle_builder import onboard_binary_classifier


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Onboard a binary classifier.")
    parser.add_argument("--source-project", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = onboard_binary_classifier(
        args.source_project,
        args.manifest,
        args.model_id,
        overwrite=args.overwrite,
    )
    print(
        f"Onboarded {result.model_id}: valid={result.valid}, "
        f"mode={result.bundle_mode}, rows={result.reference_sample_count}, "
        f"features={result.raw_feature_count}, "
        f"max_probability_diff={result.maximum_probability_absolute_difference}"
    )
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
