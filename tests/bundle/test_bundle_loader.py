"""Loader tests against the actual exported credit-default bundle."""

import numpy as np
import pandas as pd
import pytest

from monitoring_agent.bundle.loader import BundleInputError, CreditDefaultBundle


@pytest.fixture(scope="module")
def bundle() -> CreditDefaultBundle:
    """Return the real project bundle."""
    return CreditDefaultBundle()


def _inference_frame(bundle: CreditDefaultBundle) -> pd.DataFrame:
    features = bundle.load_reference_features()
    return features[bundle.ordered_inference_features()]


def test_metadata_and_schemas_load(bundle: CreditDefaultBundle) -> None:
    """All JSON contracts parse and agree on core identity."""
    metadata = bundle.load_metadata()
    schema = bundle.load_feature_schema()
    manifest = bundle.load_manifest()
    metrics = bundle.load_reference_metrics()

    assert metadata.model_name == "xgboost_public"
    assert metadata.feature_count == len(schema.ordered_features) == 36
    assert metadata.reference_sample_count == metrics.sample_count
    assert manifest.inference_available is True


def test_reference_files_align(bundle: CreditDefaultBundle) -> None:
    """Features, labels, and predictions share unique ordered record IDs."""
    features = bundle.load_reference_features()
    labels = bundle.load_reference_labels()
    predictions = bundle.load_reference_predictions()

    assert features["record_id"].is_unique
    assert labels["record_id"].is_unique
    assert predictions["record_id"].is_unique
    assert (
        features["record_id"].tolist()
        == labels["record_id"].tolist()
        == predictions["record_id"].tolist()
    )


def test_feature_order_is_enforced(bundle: CreditDefaultBundle) -> None:
    """The loader rejects a frame with correct names in the wrong order."""
    frame = _inference_frame(bundle).head(2).copy()
    columns = frame.columns.tolist()
    columns[0], columns[1] = columns[1], columns[0]

    with pytest.raises(BundleInputError, match="feature order"):
        bundle.predict_probabilities(frame[columns])


def test_target_leakage_guard(bundle: CreditDefaultBundle) -> None:
    """The loader rejects a target column in inference predictors."""
    frame = _inference_frame(bundle).head(2).copy()
    frame["Default_Flag"] = [0, 1]

    with pytest.raises(BundleInputError, match="contains target data"):
        bundle.predict_probabilities(frame)


def test_probabilities_and_threshold_predictions(bundle: CreditDefaultBundle) -> None:
    """Stored probabilities and both thresholded decisions are reproducible."""
    metadata = bundle.load_metadata()
    predictions = bundle.load_reference_predictions()
    probabilities = predictions["predicted_probability"].to_numpy(dtype=float)

    assert np.isfinite(probabilities).all()
    assert np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all()
    assert np.array_equal(
        (probabilities >= metadata.default_threshold).astype("int8"),
        predictions["predicted_class_default_threshold"].to_numpy(dtype="int8"),
    )
    assert np.array_equal(
        (probabilities >= metadata.operating_threshold).astype("int8"),
        predictions["predicted_class_operating_threshold"].to_numpy(dtype="int8"),
    )


def test_pipeline_reproduces_exported_probabilities(bundle: CreditDefaultBundle) -> None:
    """The fitted exported pipeline reproduces its stored reference probabilities."""
    stored = bundle.load_reference_predictions()["predicted_probability"].to_numpy()
    reproduced = bundle.predict_probabilities(_inference_frame(bundle))

    assert np.allclose(reproduced, stored, atol=1e-7, rtol=1e-7)
