"""Tests for project path resolution and runtime directory creation."""

from monitoring_agent.paths import (
    BASELINES_DIR,
    CONFIG_DIR,
    EVALUATIONS_DIR,
    FIGURES_DIR,
    GENERATED_REPORTS_DIR,
    METADATA_DIR,
    MODELS_DIR,
    PRODUCTION_DATA_DIR,
    PROJECT_ROOT,
    REFERENCE_DATA_DIR,
    SCENARIO_DATA_DIR,
    SRC_DIR,
    ensure_runtime_directories,
)


def test_static_project_paths_exist() -> None:
    """Required scaffold directories resolve inside the project root."""
    assert PROJECT_ROOT.is_dir()
    assert CONFIG_DIR.is_dir()
    assert SRC_DIR.is_dir()
    assert (PROJECT_ROOT / "tests").is_dir()
    assert (CONFIG_DIR / "monitoring.yaml").is_file()
    assert (CONFIG_DIR / "scenarios.yaml").is_file()


def test_ensure_runtime_directories() -> None:
    """Runtime directories can be created repeatedly without error."""
    ensure_runtime_directories()

    for directory in (
        REFERENCE_DATA_DIR,
        PRODUCTION_DATA_DIR,
        SCENARIO_DATA_DIR,
        MODELS_DIR,
        BASELINES_DIR,
        METADATA_DIR,
        GENERATED_REPORTS_DIR,
        EVALUATIONS_DIR,
        FIGURES_DIR,
    ):
        assert directory.is_dir()
