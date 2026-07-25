"""Complete validation tests against the actual exported bundle."""

from monitoring_agent.bundle.loader import CreditDefaultBundle
from monitoring_agent.bundle.validation import validate_bundle


def test_complete_bundle_validation_passes() -> None:
    """Integrity, alignment, schema, threshold, and inference checks all pass."""
    result = validate_bundle(CreditDefaultBundle())

    assert result.valid, [issue.message for issue in result.errors]
    assert all(result.checks.values())
    assert result.details["reference_sample_count"] == 6002
    assert result.details["feature_count"] == 36
