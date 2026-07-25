"""List enabled registered models without exposing source-machine paths."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from monitoring_agent.models.bundle import RegisteredModelBundle
from monitoring_agent.models.registry import ModelRegistry


def main() -> int:
    registry = ModelRegistry()
    table = Table(show_header=True, header_style="bold")
    columns = (
        "model_id",
        "display_name",
        "task_type",
        "domain_id",
        "bundle_mode",
        "inference_available",
        "reference_sample_count",
        "feature_count",
        "operating_threshold",
    )
    for column in columns:
        table.add_column(column)
    for manifest in registry.list_enabled_models():
        try:
            sample_count = str(
                RegisteredModelBundle(
                    manifest=manifest,
                    registry=registry,
                ).load_reference_metrics().sample_count
            )
        except FileNotFoundError:
            sample_count = "bundle_not_created"
        table.add_row(
            manifest.model_id,
            manifest.identity.display_name,
            manifest.identity.task_type,
            manifest.identity.domain_id,
            manifest.provenance.bundle_mode,
            str(manifest.inference_available),
            sample_count,
            str(len(manifest.data_contract.ordered_features)),
            str(manifest.prediction_contract.operating_threshold),
        )
    Console(width=180).print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
