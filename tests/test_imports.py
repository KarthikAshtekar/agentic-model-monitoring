"""Package import smoke tests."""

import monitoring_agent


def test_package_version() -> None:
    """The package imports and exposes its scaffold version."""
    assert monitoring_agent.__version__ == "0.1.0"
