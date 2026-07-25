"""Inspect a candidate model project without executing its artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from monitoring_agent.onboarding.inspector import (
    inspect_model_project,
    write_inspection_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a binary model project.")
    parser.add_argument("--source-project", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    inspection = inspect_model_project(args.source_project)
    paths = write_inspection_outputs(inspection, args.output)
    print(
        f"Inspected {inspection.scanned_file_count} candidate files; "
        f"wrote {len(paths)} outputs."
    )
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
