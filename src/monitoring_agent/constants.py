"""Foundational project constants."""

from pathlib import Path
from typing import Final

PROJECT_NAME: Final[str] = "agentic-model-monitoring"
PACKAGE_NAME: Final[str] = "monitoring_agent"
DEFAULT_CONFIG_PATH: Final[Path] = Path("configs/monitoring.yaml")
DEFAULT_SCENARIO_CONFIG_PATH: Final[Path] = Path("configs/scenarios.yaml")
