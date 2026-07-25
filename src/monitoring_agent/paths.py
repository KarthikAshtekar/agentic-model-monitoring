"""Project paths resolved independently of the current working directory."""

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
SRC_DIR: Final[Path] = PROJECT_ROOT / "src"
CONFIG_DIR: Final[Path] = PROJECT_ROOT / "configs"
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
REFERENCE_DATA_DIR: Final[Path] = DATA_DIR / "reference"
PRODUCTION_DATA_DIR: Final[Path] = DATA_DIR / "production"
SCENARIO_DATA_DIR: Final[Path] = DATA_DIR / "scenarios"
ARTIFACTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
MODELS_DIR: Final[Path] = ARTIFACTS_DIR / "models"
BASELINES_DIR: Final[Path] = ARTIFACTS_DIR / "baselines"
METADATA_DIR: Final[Path] = ARTIFACTS_DIR / "metadata"
CHECKPOINTS_DIR: Final[Path] = ARTIFACTS_DIR / "checkpoints"
DEFAULT_CHECKPOINT_DB: Final[Path] = CHECKPOINTS_DIR / "agent_checkpoints.sqlite"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
GENERATED_REPORTS_DIR: Final[Path] = REPORTS_DIR / "generated"
EVALUATIONS_DIR: Final[Path] = REPORTS_DIR / "evaluations"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"

_RUNTIME_DIRECTORIES: Final[tuple[Path, ...]] = (
    REFERENCE_DATA_DIR,
    PRODUCTION_DATA_DIR,
    SCENARIO_DATA_DIR,
    MODELS_DIR,
    BASELINES_DIR,
    METADATA_DIR,
    CHECKPOINTS_DIR,
    GENERATED_REPORTS_DIR,
    EVALUATIONS_DIR,
    FIGURES_DIR,
)

_DISPLAY_PATHS: Final[tuple[tuple[str, Path], ...]] = (
    ("Project root", PROJECT_ROOT),
    ("Source", SRC_DIR),
    ("Configuration", CONFIG_DIR),
    ("Reference data", REFERENCE_DATA_DIR),
    ("Production data", PRODUCTION_DATA_DIR),
    ("Scenario data", SCENARIO_DATA_DIR),
    ("Models", MODELS_DIR),
    ("Baselines", BASELINES_DIR),
    ("Metadata", METADATA_DIR),
    ("Checkpoints", CHECKPOINTS_DIR),
    ("Generated reports", GENERATED_REPORTS_DIR),
    ("Evaluations", EVALUATIONS_DIR),
    ("Figures", FIGURES_DIR),
)


def ensure_runtime_directories() -> None:
    """Create the approved runtime input and output directories when absent."""
    for directory in _RUNTIME_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def print_project_paths() -> None:
    """Print important resolved project paths for command-line verification."""
    print("Resolved project paths:")
    for label, path in _DISPLAY_PATHS:
        print(f"  {label:<20} {path}")
