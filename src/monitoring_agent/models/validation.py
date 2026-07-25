"""Validation helpers for registered-model manifests and paths."""

from __future__ import annotations

from pathlib import Path

from monitoring_agent.models.manifest import RegisteredModelManifest


def validate_manifest_paths(
    manifest: RegisteredModelManifest,
    project_root: Path,
    *,
    require_generated_bundle: bool = True,
) -> list[str]:
    """Return missing bundle paths without permitting repository escape."""
    missing: list[str] = []
    for field_name, relative in manifest.bundle_paths.model_dump().items():
        if relative is None:
            continue
        candidate = (project_root / relative).resolve()
        try:
            candidate.relative_to(project_root.resolve())
        except ValueError as exc:
            raise ValueError(f"{field_name} escapes the project root.") from exc
        if require_generated_bundle and not candidate.is_file():
            missing.append(relative)
    return missing
