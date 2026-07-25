"""Deterministic project inspection without model execution."""

from __future__ import annotations

from monitoring_agent.onboarding.inspector import (
    inspect_model_project,
    write_inspection_outputs,
)


def test_inspector_classifies_candidates_and_preserves_business_unknowns(
    tmp_path,
) -> None:
    (tmp_path / "final_binary_model.joblib").touch()
    (tmp_path / "test_predictions.csv").touch()
    (tmp_path / "metrics.json").write_text("{}")
    (tmp_path / "MODEL_CARD.md").write_text("# Card")
    (tmp_path / "requirements.txt").write_text("pandas")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")

    result = inspect_model_project(tmp_path)

    assert result.fields["model_artifact"].status == "inferred"
    assert result.fields["prediction_files"].status == "confirmed"
    assert "pyproject.toml" in result.fields["environment"].candidate_paths
    assert "operating threshold" in result.unresolved_business_fields
    paths = write_inspection_outputs(result, tmp_path / "inspection")
    assert {item.name for item in paths} == {
        "inspection_result.json",
        "inspection_report.md",
        "candidate_manifest.yaml",
    }
    assert "UNRESOLVED" in paths[-1].read_text()
