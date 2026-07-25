"""Read-only deterministic discovery for candidate model projects."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from monitoring_agent.onboarding.schemas import InspectionField, ProjectInspection

SUPPORTED_SUFFIXES = {
    ".joblib",
    ".pkl",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".parquet",
    ".html",
    ".md",
    ".txt",
    ".ipynb",
    ".toml",
}
SKIP_DIRECTORIES = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
UNRESOLVED_BUSINESS_FIELDS = [
    "business use",
    "positive-outcome meaning",
    "operating threshold",
    "cost of false positives or negatives",
    "protected attributes",
    "permitted remediation",
    "approval requirements",
]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not set(path.relative_to(root).parts).intersection(SKIP_DIRECTORIES)
    )


def _field(
    status: str,
    candidates: list[str],
    rationale: str,
    value: Any = None,
) -> InspectionField:
    return InspectionField(
        status=status,
        value=value,
        candidate_paths=candidates,
        rationale=rationale,
    )


def inspect_model_project(source_project: Path | str) -> ProjectInspection:
    """Inspect file names only; never execute notebooks or deserialize models."""
    root = Path(source_project).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Source project not found: {root}")
    files = _files(root)
    relative = [_relative(path, root) for path in files]
    lowered = {item: item.lower() for item in relative}
    model_paths = [
        item for item in relative if Path(item).suffix.lower() in {".joblib", ".pkl"}
    ]
    binary_models = [
        item
        for item in model_paths
        if "binary" in lowered[item] and "multiclass" not in lowered[item]
    ]
    prediction_paths = [
        item for item in relative if "prediction" in lowered[item]
    ]
    metric_paths = [
        item
        for item in relative
        if any(token in lowered[item] for token in ("metric", "run_summary"))
    ]
    data_paths = [
        item
        for item in relative
        if Path(item).suffix.lower() in {".csv", ".parquet"}
        and "prediction" not in lowered[item]
    ]
    report_paths = [
        item
        for item in relative
        if Path(item).suffix.lower() in {".html", ".md"}
    ]
    requirements = [
        item
        for item in relative
        if "requirements" in Path(item).name.lower()
        or Path(item).name.lower() in {"pyproject.toml", "environment.yml"}
    ]
    fields = {
        "model_artifact": _field(
            "inferred" if len(binary_models) == 1 else (
                "ambiguous" if binary_models or model_paths else "missing"
            ),
            binary_models or model_paths,
            "Binary-named fitted artifacts are preferred; contents were not executed.",
            binary_models[0] if len(binary_models) == 1 else None,
        ),
        "target_column": _field(
            "ambiguous",
            [
                item
                for item in relative
                if any(token in lowered[item] for token in ("config", "prediction"))
            ][:20],
            "A target name cannot be confirmed safely from file names alone.",
        ),
        "feature_schema": _field(
            "inferred" if data_paths else "missing",
            data_paths[:30],
            "Tabular data may expose a candidate feature schema after explicit review.",
        ),
        "reference_data": _field(
            "ambiguous" if len(data_paths) > 1 else (
                "inferred" if data_paths else "missing"
            ),
            data_paths[:30],
            "Several data tables may represent raw, processed, train, or held-out data.",
        ),
        "prediction_files": _field(
            "confirmed" if prediction_paths else "missing",
            prediction_paths[:30],
            "File names explicitly identify prediction artifacts.",
        ),
        "metric_files": _field(
            "confirmed" if metric_paths else "missing",
            metric_paths[:30],
            "File names explicitly identify metric or run-summary artifacts.",
        ),
        "reports": _field(
            "confirmed" if report_paths else "missing",
            report_paths[:30],
            "Reader-facing Markdown and HTML reports were discovered.",
        ),
        "environment": _field(
            "confirmed" if requirements else "missing",
            requirements,
            "Dependency declaration files were discovered.",
        ),
    }
    counts = Counter(path.suffix.lower() for path in files)
    return ProjectInspection(
        source_project=str(root),
        scanned_file_count=len(files),
        artifact_counts=dict(sorted(counts.items())),
        fields=fields,
        unresolved_business_fields=UNRESOLVED_BUSINESS_FIELDS,
        created_at_utc=datetime.now(UTC),
    )


def write_inspection_outputs(
    inspection: ProjectInspection,
    output_directory: Path | str,
) -> list[Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "inspection_result.json"
    report_path = output / "inspection_report.md"
    candidate_path = output / "candidate_manifest.yaml"
    json_path.write_text(
        inspection.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    field_lines = "\n".join(
        f"| `{name}` | `{field.status}` | "
        f"{', '.join(f'`{item}`' for item in field.candidate_paths[:5]) or '-'} | "
        f"{field.rationale} |"
        for name, field in inspection.fields.items()
    )
    unresolved = "\n".join(
        f"- {item}: `UNRESOLVED — must be supplied explicitly`"
        for item in inspection.unresolved_business_fields
    )
    report_path.write_text(
        f"""# Candidate model-project inspection

- Source project: `{inspection.source_project}`
- Files scanned: `{inspection.scanned_file_count}`
- Created: `{inspection.created_at_utc.isoformat()}`

| Field | Status | Candidates | Rationale |
|---|---|---|---|
{field_lines}

## Fields deliberately not inferred

{unresolved}

This inspection is deterministic discovery, not an onboarding decision. Model files were
not deserialized and notebooks were not executed.
""",
        encoding="utf-8",
        newline="\n",
    )
    candidate: dict[str, Any] = {
        "identity": {
            "model_id": "UNRESOLVED",
            "display_name": "UNRESOLVED",
            "model_version": "UNRESOLVED",
            "task_type": "binary_classification",
            "adapter_type": "sklearn_binary_classifier",
            "domain_id": "UNRESOLVED",
        },
        "business_context": {
            item.replace(" ", "_"): "UNRESOLVED — must be supplied explicitly"
            for item in inspection.unresolved_business_fields
        },
        "inspection_candidates": {
            name: field.model_dump(mode="json")
            for name, field in inspection.fields.items()
        },
    }
    candidate_path.write_text(
        yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    return [json_path, report_path, candidate_path]
