"""YAML-backed registered-model registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from monitoring_agent.models.manifest import RegisteredModelManifest
from monitoring_agent.paths import PROJECT_ROOT


class RegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: str
    enabled: bool


class RegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_model_id: str
    models: dict[str, RegistryEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def default_must_exist_and_be_enabled(self) -> RegistryConfig:
        entry = self.models.get(self.default_model_id)
        if entry is None:
            raise ValueError("default_model_id is not present in models.")
        if not entry.enabled:
            raise ValueError("default_model_id must be enabled.")
        return self


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=_UniqueKeyLoader)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")
    return payload


class ModelRegistry:
    """Resolve enabled model manifests without hidden path assumptions."""

    def __init__(
        self,
        registry_path: Path | str | None = None,
        *,
        project_root: Path | str = PROJECT_ROOT,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.registry_path = Path(
            registry_path or self.project_root / "configs/model_registry.yaml"
        ).resolve()
        self.config = RegistryConfig.model_validate(_load_yaml(self.registry_path))
        self._manifests: dict[str, RegisteredModelManifest] = {}
        self._validate_entries()

    def _resolve_manifest_path(self, value: str) -> Path:
        candidate = (self.project_root / value).resolve()
        try:
            candidate.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError("Manifest path escapes the monitoring repository.") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Registered manifest not found: {candidate}")
        return candidate

    def _validate_entries(self) -> None:
        seen_manifest_paths: dict[Path, str] = {}
        for model_id, entry in self.config.models.items():
            path = self._resolve_manifest_path(entry.manifest)
            if path in seen_manifest_paths:
                other = seen_manifest_paths[path]
                raise ValueError(
                    f"Models {other!r} and {model_id!r} use the same manifest path."
                )
            seen_manifest_paths[path] = model_id
            manifest = RegisteredModelManifest.model_validate(_load_yaml(path))
            if manifest.model_id != model_id:
                raise ValueError(
                    f"Registry key {model_id!r} does not match manifest model_id "
                    f"{manifest.model_id!r}."
                )
            self._manifests[model_id] = manifest

    @property
    def default_model_id(self) -> str:
        return self.config.default_model_id

    def list_enabled_models(self) -> list[RegisteredModelManifest]:
        return [
            self._manifests[model_id]
            for model_id, entry in self.config.models.items()
            if entry.enabled
        ]

    def load_manifest(self, model_id: str | None = None) -> RegisteredModelManifest:
        resolved = model_id or self.default_model_id
        entry = self.config.models.get(resolved)
        if entry is None:
            choices = ", ".join(sorted(self.config.models))
            raise KeyError(f"Unknown model_id {resolved!r}. Registered models: {choices}.")
        if not entry.enabled:
            raise ValueError(f"Registered model {resolved!r} is disabled.")
        return self._manifests[resolved]
