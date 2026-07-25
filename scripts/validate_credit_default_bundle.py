"""Validate the complete exported credit-default monitoring bundle."""

from __future__ import annotations

from monitoring_agent.bundle.validation import validate_bundle


def main() -> int:
    """Print a concise structured pass/fail summary."""
    result = validate_bundle()
    status = "PASS" if result.valid else "FAIL"
    print(f"Credit-default bundle validation: {status}")
    print(f"  checks passed: {sum(result.checks.values())}/{len(result.checks)}")
    for name, value in result.details.items():
        print(f"  {name}: {value}")
    for issue in result.warnings:
        print(f"  warning [{issue.check}]: {issue.message}")
    for issue in result.errors:
        print(f"  error [{issue.check}]: {issue.message}")
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
