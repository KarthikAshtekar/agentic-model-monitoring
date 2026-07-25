"""Verify that the repository scaffold and initial configuration are usable."""

from __future__ import annotations

from pathlib import Path

import yaml

import monitoring_agent
from monitoring_agent.paths import (
    CONFIG_DIR,
    ensure_runtime_directories,
    print_project_paths,
)

CONFIG_FILES = (CONFIG_DIR / "monitoring.yaml", CONFIG_DIR / "scenarios.yaml")


def _load_yaml(path: Path) -> object:
    """Load a YAML document with the safe parser and require non-empty content."""
    with path.open(encoding="utf-8") as file:
        document = yaml.safe_load(file)
    if document is None:
        raise ValueError(f"Configuration is empty: {path}")
    return document


def main() -> int:
    """Run scaffold checks and return a process exit code."""
    try:
        print(f"monitoring_agent version: {monitoring_agent.__version__}")
        print_project_paths()

        for config_path in CONFIG_FILES:
            if not config_path.is_file():
                raise FileNotFoundError(f"Missing configuration file: {config_path}")
            _load_yaml(config_path)
            print(f"Validated YAML: {config_path}")

        ensure_runtime_directories()
        print("Runtime directories are available.")
    except Exception as exc:
        print(f"Setup verification failed: {exc}")
        return 1

    print("Setup verification completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
